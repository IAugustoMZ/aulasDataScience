"""
modeling/ensemble.py — Construtor de modelos ensemble (Stacking e Voting).

Responsabilidade única: construir, otimizar e avaliar ensembles a partir
dos top-N modelos individuais selecionados após a otimização.

StackingRegressor : usa previsões out-of-fold como features para um meta-learner
VotingRegressor  : média ponderada das previsões dos modelos base

Ambos aceitam o CVRunner e o otimizador Optuna para busca de hiperparâmetros.
"""
from __future__ import annotations

import logging
from typing import Any

import mlflow
import optuna
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor, VotingRegressor

from src.modeling.model_factory import construir_pipeline
from src.modeling.cross_validation import CVRunner
from src.modeling.metrics import agregar_metricas_folds
from src.modeling.optimizer import _params_reducer_padrao


class EnsembleBuilder:
    """
    Constrói e otimiza ensembles Stacking e Voting.

    Parâmetros
    ----------
    ensembles_cfg : dict da seção ensembles em modeling.yaml
    cv_runner     : CVRunner configurado
    pipe_cfg      : dict da seção pipeline em modeling.yaml
    feat_red_cfg  : dict da seção feature_reduction em modeling.yaml
    n_trials_global: número de trials padrão para Optuna
    seed          : semente aleatória global
    logger        : Logger opcional
    """

    def __init__(
        self,
        ensembles_cfg: dict,
        cv_runner: CVRunner,
        pipe_cfg: dict,
        feat_red_cfg: dict,
        n_trials_global: int = 10,
        seed: int = 42,
        logger: logging.Logger | None = None,
    ) -> None:
        self.ensembles_cfg   = ensembles_cfg
        self.cv_runner       = cv_runner
        self.pipe_cfg        = pipe_cfg
        self.feat_red_cfg    = feat_red_cfg
        self.n_trials_global = n_trials_global
        self.seed            = seed
        self.logger          = logger

    def _construir_estimadores_base(self, top_n_entries: list[tuple]) -> list[tuple]:
        """
        Cria Pipelines NÃO treinados dos top-N modelos com os melhores parâmetros.

        Parâmetros
        ----------
        top_n_entries : lista de (nome, resultado) ordenada por CV RMSE

        Retorna
        -------
        Lista de (nome, pipeline) para usar em StackingRegressor e VotingRegressor
        """
        return [
            (nome, construir_pipeline(
                model_cfg     =resultado['model_cfg'],
                params_modelo =resultado['best_params'],
                params_reducer=resultado.get('reducer_params', _params_reducer_padrao(self.feat_red_cfg)),
                pipe_cfg      =self.pipe_cfg,
            ))
            for nome, resultado in top_n_entries
        ]

    def construir_stacking(
        self,
        top_n_entries: list[tuple],
        X_train: Any,
        y_train: Any,
    ) -> dict | None:
        """
        Otimiza e avalia StackingRegressor via Optuna.

        O Optuna busca o melhor alpha do Ridge meta-learner.
        passthrough=False é sempre utilizado para evitar que NaN do X bruto
        contaminem as meta-features e causem falha no Ridge meta-learner.

        Parâmetros
        ----------
        top_n_entries : lista de (nome, resultado) dos modelos base
        X_train       : features de treino
        y_train       : target de treino

        Retorna
        -------
        dict com resultados ou None se todos os trials falharem
        """
        cfg_stacking = self.ensembles_cfg.get('stacking', {})
        if not cfg_stacking.get('enabled', True):
            if self.logger:
                self.logger.info('  [SKIP] Stacking desabilitado no config')
            return None

        n_trials  = cfg_stacking.get('optuna_trials', self.n_trials_global)
        inner_cv  = cfg_stacking.get('inner_cv_folds', 3)
        nomes_base = [nome for nome, _ in top_n_entries]

        if self.logger:
            self.logger.info('  [OPTUNA] stacking  (%d trials) ...', n_trials)

        with mlflow.start_run(
            run_name='optuna_stacking',
            tags={'stage': 'optuna', 'model': 'stacking', 'base_models': str(nomes_base)},
        ):
            def _objetivo_stacking(trial: optuna.Trial) -> float:
                meta_alpha = trial.suggest_float('meta_alpha', 1e-4, 1e4, log=True)
                stacking = StackingRegressor(
                    estimators=self._construir_estimadores_base(top_n_entries),
                    final_estimator=Ridge(alpha=meta_alpha),
                    passthrough=False,
                    cv=inner_cv,
                    n_jobs=1,
                )
                fold_mets = self.cv_runner.executar(stacking, X_train, y_train)
                agg = agregar_metricas_folds(fold_mets)

                with mlflow.start_run(
                    run_name=f'stacking_trial_{trial.number}',
                    nested=True,
                    tags={'stage': 'stacking_trial'},
                ):
                    mlflow.log_params({'meta_alpha': meta_alpha, 'passthrough': 'False'})
                    mlflow.log_metrics({
                        'cv_rmse_mean': agg['cv_rmse_mean'],
                        'cv_r2_mean'  : agg['cv_r2_mean'],
                    })
                return agg['cv_rmse_mean']

            estudo = optuna.create_study(
                direction='minimize',
                study_name='stacking',
                sampler=optuna.samplers.TPESampler(seed=self.seed),
            )
            estudo.optimize(_objetivo_stacking, n_trials=n_trials, catch=(Exception,))

            try:
                mlflow.log_params({f'best_{k}': v for k, v in estudo.best_params.items()})
                mlflow.log_metric('best_cv_rmse', estudo.best_value)
            except ValueError:
                if self.logger:
                    self.logger.warning('  [SKIP] stacking — todos os trials falharam')
                return None
            mlflow.log_param('base_models', str(nomes_base))

        # Constrói e avalia o melhor Stacking com CV completo
        melhor_meta_alpha = estudo.best_params['meta_alpha']
        melhor_stacking = StackingRegressor(
            estimators=self._construir_estimadores_base(top_n_entries),
            final_estimator=Ridge(alpha=melhor_meta_alpha),
            passthrough=False,
            cv=inner_cv,
            n_jobs=1,
        )
        fold_mets_stacking = self.cv_runner.executar(melhor_stacking, X_train, y_train)
        agg_stacking       = agregar_metricas_folds(fold_mets_stacking)

        if self.logger:
            self.logger.info(
                '    Stacking → CV RMSE: %.2f ± %.2f  (meta_alpha=%.4f)',
                agg_stacking['cv_rmse_mean'], agg_stacking['cv_rmse_std'], melhor_meta_alpha,
            )

        return {
            **agg_stacking,
            'fold_metrics': fold_mets_stacking,
            'model_cfg'   : {'module': 'sklearn.ensemble', 'class': 'StackingRegressor', 'default_params': {}},
            'best_params' : estudo.best_params,
            'tuned'       : True,
            '_instance'   : melhor_stacking,
        }

    def construir_voting(
        self,
        top_n_entries: list[tuple],
        X_train: Any,
        y_train: Any,
    ) -> dict | None:
        """
        Otimiza e avalia VotingRegressor via Optuna.

        O Optuna busca os pesos ótimos (inteiros) para cada modelo base.

        Parâmetros
        ----------
        top_n_entries : lista de (nome, resultado) dos modelos base
        X_train       : features de treino
        y_train       : target de treino

        Retorna
        -------
        dict com resultados ou None se todos os trials falharem
        """
        cfg_voting = self.ensembles_cfg.get('voting', {})
        if not cfg_voting.get('enabled', True):
            if self.logger:
                self.logger.info('  [SKIP] Voting desabilitado no config')
            return None

        n_trials = cfg_voting.get('optuna_trials', self.n_trials_global)
        w_low    = cfg_voting.get('weight_low', 1)
        w_high   = cfg_voting.get('weight_high', 10)
        nomes_base = [nome for nome, _ in top_n_entries]

        if self.logger:
            self.logger.info('  [OPTUNA] voting    (%d trials) ...', n_trials)

        with mlflow.start_run(
            run_name='optuna_voting',
            tags={'stage': 'optuna', 'model': 'voting', 'base_models': str(nomes_base)},
        ):
            def _objetivo_voting(trial: optuna.Trial) -> float:
                pesos = [
                    trial.suggest_int(f'w_{nome}', w_low, w_high)
                    for nome in nomes_base
                ]
                voting = VotingRegressor(
                    estimators=self._construir_estimadores_base(top_n_entries),
                    weights=pesos,
                    n_jobs=1,
                )
                fold_mets = self.cv_runner.executar(voting, X_train, y_train)
                agg = agregar_metricas_folds(fold_mets)

                with mlflow.start_run(
                    run_name=f'voting_trial_{trial.number}',
                    nested=True,
                    tags={'stage': 'voting_trial'},
                ):
                    mlflow.log_params({f'w_{n}': w for n, w in zip(nomes_base, pesos)})
                    mlflow.log_metrics({
                        'cv_rmse_mean': agg['cv_rmse_mean'],
                        'cv_r2_mean'  : agg['cv_r2_mean'],
                    })
                return agg['cv_rmse_mean']

            estudo = optuna.create_study(
                direction='minimize',
                study_name='voting',
                sampler=optuna.samplers.TPESampler(seed=self.seed),
            )
            estudo.optimize(_objetivo_voting, n_trials=n_trials, catch=(Exception,))

            try:
                mlflow.log_params({f'best_{k}': v for k, v in estudo.best_params.items()})
                mlflow.log_metric('best_cv_rmse', estudo.best_value)
            except ValueError:
                if self.logger:
                    self.logger.warning('  [SKIP] voting — todos os trials falharam')
                return None
            mlflow.log_param('base_models', str(nomes_base))

        # Constrói e avalia o melhor Voting com CV completo
        melhores_pesos = [estudo.best_params[f'w_{nome}'] for nome in nomes_base]
        melhor_voting  = VotingRegressor(
            estimators=self._construir_estimadores_base(top_n_entries),
            weights=melhores_pesos,
            n_jobs=1,
        )
        fold_mets_voting = self.cv_runner.executar(melhor_voting, X_train, y_train)
        agg_voting       = agregar_metricas_folds(fold_mets_voting)

        if self.logger:
            self.logger.info(
                '    Voting → CV RMSE: %.2f ± %.2f  (pesos=%s)',
                agg_voting['cv_rmse_mean'], agg_voting['cv_rmse_std'], melhores_pesos,
            )

        return {
            **agg_voting,
            'fold_metrics': fold_mets_voting,
            'model_cfg'   : {'module': 'sklearn.ensemble', 'class': 'VotingRegressor', 'default_params': {}},
            'best_params' : estudo.best_params,
            'tuned'       : True,
            '_instance'   : melhor_voting,
        }
