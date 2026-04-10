"""
modeling/tracker.py — Rastreador MLflow para experimentação MLOps.

Responsabilidade única: encapsular toda interação com MLflow:
  - Configuração de tracking URI e experimento
  - Logging de parâmetros, métricas e artefatos
  - Registro do modelo no Model Registry
  - Geração do resumo JSON do experimento

Usa SQLite como backend de armazenamento (configurável em modeling.yaml):
    tracking_uri: "sqlite:///mlruns.db"

Para inspecionar os resultados, execute no terminal:
    mlflow ui --backend-store-uri sqlite:///mlruns.db
e acesse http://localhost:5000
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn


class MLflowTracker:
    """
    Gerencia o ciclo de vida de tracking MLflow para o pipeline de modelagem.

    Parâmetros
    ----------
    tracking_uri     : URI do backend MLflow (sqlite:///mlruns.db ou caminho de pasta)
    experiment_name  : nome do experimento MLflow
    root_dir         : diretório raiz do projeto (para resolver caminhos relativos)
    logger           : Logger opcional
    """

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        root_dir: Path,
        logger: logging.Logger | None = None,
    ) -> None:
        self.tracking_uri    = tracking_uri
        self.experiment_name = experiment_name
        self.root_dir        = root_dir
        self.logger          = logger
        self._configurar()

    def _configurar(self) -> None:
        """Configura o tracking URI e o experimento MLflow."""
        # Suporte a URI absoluta (sqlite://) ou caminho relativo (pasta mlruns)
        uri = self.tracking_uri
        if not uri.startswith('sqlite:') and not uri.startswith('http'):
            # Caminho relativo → converte para URI absoluta de pasta
            uri = (self.root_dir / uri).as_uri()

        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(self.experiment_name)

        if self.logger:
            self.logger.info('MLFlow tracking URI  : %s', uri)
            self.logger.info('MLFlow experiment    : %s', self.experiment_name)

    # ── Logging de baseline ───────────────────────────────────────────────────

    def logar_baseline(
        self,
        model_name: str,
        params: dict,
        fold_metrics: list[dict],
        agg_metrics: dict,
        training_time: float,
        model_class: str,
        reducer_method: str,
    ) -> None:
        """Abre um run de baseline e registra parâmetros + métricas."""
        with mlflow.start_run(
            run_name=f'baseline_{model_name}',
            tags={'stage': 'baseline', 'model': model_name},
        ):
            params_log = {str(k): (str(v) if v is None else v) for k, v in params.items()}
            params_log['reducer_method'] = reducer_method
            mlflow.log_params(params_log)
            mlflow.set_tag('model_class', model_class)
            mlflow.set_tag('reducer_method', reducer_method)

            for fm in fold_metrics:
                step = fm['fold']
                mlflow.log_metric('fold_rmse', fm['rmse'], step=step)
                mlflow.log_metric('fold_mae',  fm['mae'],  step=step)
                mlflow.log_metric('fold_r2',   fm['r2'],   step=step)
                mlflow.log_metric('fold_mape', fm['mape'], step=step)

            mlflow.log_metrics(agg_metrics)
            mlflow.log_metric('training_time_s', training_time)

    # ── Logging de otimização Optuna ──────────────────────────────────────────

    def contexto_otimizacao(self, model_name: str, stage: str = 'optuna'):
        """
        Context manager para o run pai de otimização Optuna.

        Uso:
            with tracker.contexto_otimizacao('ridge') as _:
                # trials aninhados são abertos dentro do optimizer
        """
        return mlflow.start_run(
            run_name=f'optuna_{model_name}',
            tags={'stage': stage, 'model': model_name},
        )

    def logar_melhor_optuna(
        self,
        best_params: dict,
        best_cv_rmse: float,
        n_trials: int,
        study: Any | None = None,
        artifact_paths: list[Path] | None = None,
    ) -> None:
        """Registra os melhores params e artefatos Optuna no run ativo."""
        params_log = {
            f'best_{k}': (str(v) if v is None else v)
            for k, v in best_params.items()
        }
        mlflow.log_params(params_log)
        mlflow.log_metrics({'best_cv_rmse': best_cv_rmse, 'n_trials': n_trials})

        if artifact_paths:
            for p in artifact_paths:
                if p and Path(p).exists():
                    mlflow.log_artifact(str(p), artifact_path='optuna')

    # ── Logging do melhor modelo ──────────────────────────────────────────────

    def logar_melhor_modelo(
        self,
        model_name: str,
        model: Any,
        best_params: dict,
        reducer_params: dict,
        cv_metrics: dict,
        train_metrics: dict,
        fold_metrics: list[dict],
        plot_paths: dict,
        tuned: bool,
    ) -> str:
        """
        Abre um run para o melhor modelo, loga tudo e retorna o run_id.

        Retorna
        -------
        str : run_id do run criado
        """
        with mlflow.start_run(
            run_name=f'best_model_{model_name}',
            tags={
                'stage' : 'best_model',
                'model' : model_name,
                'tuned' : str(tuned),
            },
        ) as run:
            # Parâmetros — estimador + reducer
            params_log = {k: (str(v) if v is None else v) for k, v in best_params.items()}
            params_log.update({
                f'reducer_{k}': (str(v) if v is None else v)
                for k, v in reducer_params.items()
            })
            mlflow.log_params(params_log)

            # Métricas de CV
            mlflow.log_metrics(cv_metrics)

            # Métricas de treino completo
            mlflow.log_metrics(train_metrics)

            # Métricas por fold
            for fm in fold_metrics:
                step = fm['fold']
                mlflow.log_metric('fold_rmse', fm['rmse'], step=step)
                mlflow.log_metric('fold_mae',  fm['mae'],  step=step)
                mlflow.log_metric('fold_r2',   fm['r2'],   step=step)

            # Artefatos — plots diagnósticos
            for caminho in plot_paths.values():
                if caminho and Path(caminho).exists():
                    mlflow.log_artifact(str(caminho), artifact_path='plots')

            # Modelo serializado
            mlflow.sklearn.log_model(model, artifact_path='model')

            return run.info.run_id

    def logar_holdout(
        self,
        run_id: str,
        holdout_metrics: dict,
        delta_pct: float,
        holdout_plot_path: Path | None = None,
    ) -> None:
        """Adiciona métricas do holdout ao run do melhor modelo."""
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({
                'holdout_rmse'            : holdout_metrics['rmse'],
                'holdout_mae'             : holdout_metrics['mae'],
                'holdout_r2'              : holdout_metrics['r2'],
                'holdout_mape'            : holdout_metrics['mape'],
                'cv_vs_holdout_delta_pct' : delta_pct,
            })
            if holdout_plot_path and Path(holdout_plot_path).exists():
                mlflow.log_artifact(str(holdout_plot_path), artifact_path='plots')

    def registrar_modelo(self, run_id: str, registry_name: str) -> None:
        """Registra o modelo no MLflow Model Registry."""
        model_uri = f'runs:/{run_id}/model'
        try:
            resultado = mlflow.register_model(model_uri=model_uri, name=registry_name)
            if self.logger:
                self.logger.info(
                    'Modelo registrado: %s (versão %s)', registry_name, resultado.version
                )
        except Exception as exc:
            if self.logger:
                self.logger.warning('Model Registry não disponível: %s', exc)

    def salvar_resumo_json(
        self,
        output_dir: Path,
        best_model_name: str,
        best_run_id: str,
        best_result: dict,
        holdout_metrics: dict,
        top_n_names: list[str],
        full_ranking_records: list[dict],
    ) -> Path:
        """
        Salva um resumo JSON do experimento em disco.

        Retorna
        -------
        Path do arquivo salvo
        """
        best_params   = best_result.get('best_params', {})
        reducer_params = best_result.get('reducer_params', {})

        resumo = {
            'best_model'          : best_model_name,
            'best_run_id'         : best_run_id,
            'experiment_name'     : self.experiment_name,
            'cv_rmse_mean'        : round(best_result['cv_rmse_mean'], 4),
            'cv_rmse_std'         : round(best_result['cv_rmse_std'], 4),
            'cv_r2_mean'          : round(best_result['cv_r2_mean'], 4),
            'holdout_rmse'        : round(holdout_metrics['rmse'], 4),
            'holdout_mae'         : round(holdout_metrics['mae'], 4),
            'holdout_r2'          : round(holdout_metrics['r2'], 4),
            'holdout_mape'        : round(holdout_metrics['mape'], 4),
            'best_params'         : {k: (str(v) if v is None else v) for k, v in best_params.items()},
            'reducer_params'      : {k: (str(v) if v is None else v) for k, v in reducer_params.items()},
            'top3_base_models'    : top_n_names,
            'all_models_ranked'   : full_ranking_records,
        }

        caminho = output_dir / 'experiment_summary.json'
        with open(caminho, 'w', encoding='utf-8') as f:
            json.dump(resumo, f, indent=2, ensure_ascii=False, default=str)

        if self.logger:
            self.logger.info('Resumo salvo: %s', caminho)

        return caminho
