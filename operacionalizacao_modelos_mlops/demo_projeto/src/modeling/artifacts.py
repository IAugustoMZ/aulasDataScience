"""
modeling/artifacts.py — Gerador de artefatos diagnósticos do melhor modelo.

Responsabilidade única: salvar plots PNG em disco e retornar seus caminhos.
O MLflowTracker é responsável por logá-los no MLflow — separação de concerns.

Plots gerados (configuráveis em modeling.yaml → artifacts.plots):
  residuals             — resíduos vs valores ajustados
  predicted_vs_actual   — previstos vs reais com linha de referência perfeita
  error_distribution    — histograma + CDF dos resíduos
  feature_importance    — top-20 features por importância (tree, linear ou permutation)
  cv_fold_comparison    — RMSE e R² por fold (robustez)
  learning_curve        — viés-variância por tamanho do treino
  optuna_history        — histórico de otimização por trial (Optuna)
  optuna_param_importances — importância dos hiperparâmetros no Optuna
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')   # backend não-interativo: salva em arquivo sem abrir janela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline as SklearnPipeline


class ArtifactGenerator:
    """
    Gera e salva plots diagnósticos do melhor modelo.

    Parâmetros
    ----------
    output_dir   : diretório onde os PNG serão salvos
    artifacts_cfg: dict da seção artifacts em modeling.yaml
    logger       : Logger opcional
    """

    def __init__(
        self,
        output_dir: Path,
        artifacts_cfg: dict,
        logger: logging.Logger | None = None,
    ) -> None:
        self.output_dir   = Path(output_dir)
        self.artifacts_cfg = artifacts_cfg
        self.logger = logger
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Helpers internos ──────────────────────────────────────────────────────

    def _salvar(self, fig: Any, nome: str) -> Path | None:
        """Salva a figura em disco e fecha. Retorna o Path ou None em caso de erro."""
        caminho = self.output_dir / nome
        try:
            fig.savefig(caminho, dpi=120, bbox_inches='tight')
        except Exception as exc:
            if self.logger:
                self.logger.warning('Falha ao salvar plot %s: %s', nome, exc)
            return None
        finally:
            plt.close(fig)
        if self.logger:
            self.logger.info('Plot salvo: %s', caminho)
        return caminho

    def _extrair_importancia_features(
        self,
        model: Any,
        feature_names: list[str],
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> pd.Series:
        """
        Extrai importância de features do modelo treinado.

        Prioridade:
          1. feature_importances_ (árvores, ensembles baseados em árvores)
          2. coef_                (modelos lineares — usa valor absoluto)
          3. permutation_importance (fallback model-agnóstico: SVR, KNN, ensembles)
        """
        if isinstance(model, SklearnPipeline):
            estimador = model.named_steps['estimator']
            reducer   = model.named_steps.get('reducer')
            if reducer is not None and reducer.selected_features is not None:
                nomes_imp = reducer.selected_features
            else:
                nomes_imp = feature_names
        else:
            estimador = model
            nomes_imp = feature_names

        if hasattr(estimador, 'feature_importances_'):
            return pd.Series(estimador.feature_importances_, index=nomes_imp)
        elif hasattr(estimador, 'coef_'):
            coef = np.abs(estimador.coef_)
            if coef.ndim > 1:
                coef = coef.flatten()
            return pd.Series(coef, index=nomes_imp)
        else:
            amostra = min(2000, len(X_val))
            idx = np.random.default_rng(42).choice(len(X_val), amostra, replace=False)
            r = permutation_importance(
                model, X_val.iloc[idx], y_val.iloc[idx],
                n_repeats=5, random_state=42, n_jobs=-1,
            )
            return pd.Series(r.importances_mean, index=feature_names)

    # ── Plots individuais ─────────────────────────────────────────────────────

    def plot_residuals(
        self, y_pred_train: np.ndarray, residuos: np.ndarray, model_name: str
    ) -> Path | None:
        """Resíduos vs valores ajustados — detecta heterocedasticidade."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_pred_train, residuos, alpha=0.25, s=6, color='steelblue', edgecolors='none')
        ax.axhline(0, color='crimson', linewidth=1.5, linestyle='--', label='Resíduo = 0')
        ax.set_xlabel('Valores Previstos ($)')
        ax.set_ylabel('Resíduos ($)')
        ax.set_title(f'Resíduos vs Valores Ajustados — {model_name}')
        ax.legend()
        plt.tight_layout()
        return self._salvar(fig, 'residuals.png')

    def plot_predicted_vs_actual(
        self, y_train: pd.Series, y_pred_train: np.ndarray, r2: float, model_name: str
    ) -> Path | None:
        """Previstos vs reais com linha de referência perfeita."""
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(y_train, y_pred_train, alpha=0.25, s=6, color='steelblue', edgecolors='none')
        lim = [min(y_train.min(), y_pred_train.min()), max(y_train.max(), y_pred_train.max())]
        ax.plot(lim, lim, 'r--', linewidth=1.5, label='Previsão Perfeita')
        ax.set_xlabel('Valores Reais ($)')
        ax.set_ylabel('Valores Previstos ($)')
        ax.set_title(f'Previstos vs Reais — {model_name}\nR² = {r2:.4f}')
        ax.legend()
        plt.tight_layout()
        return self._salvar(fig, 'pred_vs_actual.png')

    def plot_error_distribution(self, residuos: np.ndarray, model_name: str) -> Path | None:
        """Histograma + CDF dos resíduos."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(residuos, bins=60, color='steelblue', edgecolor='white', alpha=0.8)
        axes[0].axvline(0, color='crimson', linewidth=1.5, linestyle='--', label='0')
        axes[0].axvline(float(np.mean(residuos)), color='orange', linewidth=1.5,
                        linestyle='-.', label=f'Média: {np.mean(residuos):.0f}')
        axes[0].set_xlabel('Resíduo ($)')
        axes[0].set_ylabel('Frequência')
        axes[0].set_title('Distribuição dos Resíduos')
        axes[0].legend()

        sorted_res = np.sort(residuos)
        cum_prob   = np.arange(1, len(sorted_res) + 1) / len(sorted_res)
        axes[1].plot(sorted_res, cum_prob, color='steelblue', linewidth=1.5)
        axes[1].axvline(0, color='crimson', linewidth=1.5, linestyle='--')
        axes[1].set_xlabel('Resíduo ($)')
        axes[1].set_ylabel('Probabilidade Acumulada')
        axes[1].set_title('CDF dos Resíduos')
        axes[1].grid(True, alpha=0.3)
        fig.suptitle(f'Análise de Erros — {model_name}', fontsize=13, fontweight='bold')
        plt.tight_layout()
        return self._salvar(fig, 'error_dist.png')

    def plot_feature_importance(
        self,
        model: Any,
        feature_names: list[str],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_name: str,
    ) -> Path | None:
        """Top-20 features por importância."""
        try:
            importancia = self._extrair_importancia_features(model, feature_names, X_train, y_train)
            top20 = importancia.nlargest(20).sort_values()
            fig, ax = plt.subplots(figsize=(10, 8))
            top20.plot(kind='barh', ax=ax, color='steelblue', edgecolor='white')
            ax.set_xlabel('Importância')
            ax.set_title(f'Top-20 Features — {model_name}')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            if self.logger:
                self.logger.info('Top-5 features: %s', importancia.nlargest(5).index.tolist())
            return self._salvar(fig, 'feature_importance.png')
        except Exception as exc:
            if self.logger:
                self.logger.warning('Feature importance falhou: %s', exc)
            return None

    def plot_cv_fold_comparison(
        self, fold_metrics: list[dict], model_name: str
    ) -> Path | None:
        """RMSE e R² por fold — avalia robustez do modelo."""
        fold_rmse = [fm['rmse'] for fm in fold_metrics]
        fold_r2   = [fm['r2']   for fm in fold_metrics]
        labels    = [f'Fold {fm["fold"]}' for fm in fold_metrics]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].bar(labels, fold_rmse, color='steelblue', edgecolor='white')
        axes[0].axhline(float(np.mean(fold_rmse)), color='crimson', linewidth=1.5,
                        linestyle='--', label=f'Média: {np.mean(fold_rmse):.0f}')
        axes[0].set_ylabel('RMSE ($)')
        axes[0].set_title('RMSE por Fold')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

        axes[1].bar(labels, fold_r2, color='teal', edgecolor='white')
        axes[1].axhline(float(np.mean(fold_r2)), color='crimson', linewidth=1.5,
                        linestyle='--', label=f'Média: {np.mean(fold_r2):.4f}')
        axes[1].set_ylabel('R²')
        axes[1].set_title('R² por Fold')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        fig.suptitle(f'Robustez por Fold — {model_name}', fontsize=13, fontweight='bold')
        plt.tight_layout()
        return self._salvar(fig, 'cv_fold_comparison.png')

    def plot_learning_curve(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series, model_name: str) -> Path | None:
        """Curva de aprendizado — diagnóstica viés-variância."""
        eh_ensemble_complexo = isinstance(model, (StackingRegressor, VotingRegressor))
        if eh_ensemble_complexo:
            if self.logger:
                self.logger.warning('Learning curve ignorada para ensemble (custo computacional alto).')
            return None
        try:
            tamanhos, scores_treino, scores_val = learning_curve(
                clone(model), X_train, y_train,
                cv=5,
                scoring='neg_root_mean_squared_error',
                train_sizes=np.linspace(0.1, 1.0, 8),
                n_jobs=-1,
            )
            media_treino = -scores_treino.mean(axis=1)
            std_treino   = scores_treino.std(axis=1)
            media_val    = -scores_val.mean(axis=1)
            std_val      = scores_val.std(axis=1)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(tamanhos, media_treino, 'o-', color='steelblue', label='Treino', linewidth=2)
            ax.fill_between(tamanhos, media_treino - std_treino, media_treino + std_treino, alpha=0.15, color='steelblue')
            ax.plot(tamanhos, media_val, 's-', color='crimson', label='Validação (CV)', linewidth=2)
            ax.fill_between(tamanhos, media_val - std_val, media_val + std_val, alpha=0.15, color='crimson')
            ax.set_xlabel('Tamanho do conjunto de treino')
            ax.set_ylabel('RMSE ($)')
            ax.set_title(f'Curva de Aprendizado — {model_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            return self._salvar(fig, 'learning_curve.png')
        except Exception as exc:
            if self.logger:
                self.logger.warning('Learning curve falhou: %s', exc)
            return None

    def plot_holdout_evaluation(
        self,
        y_holdout: pd.Series,
        y_pred_holdout: np.ndarray,
        holdout_metrics: dict,
        model_name: str,
    ) -> Path | None:
        """Previstos vs reais e distribuição de resíduos no holdout."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes[0].scatter(y_holdout, y_pred_holdout, alpha=0.3, s=6, color='teal', edgecolors='none')
        lim_h = [min(y_holdout.min(), y_pred_holdout.min()), max(y_holdout.max(), y_pred_holdout.max())]
        axes[0].plot(lim_h, lim_h, 'r--', linewidth=1.5, label='Perfeito')
        axes[0].set_xlabel('Real ($)')
        axes[0].set_ylabel('Previsto ($)')
        axes[0].set_title(f'Holdout — Previstos vs Reais\nR² = {holdout_metrics["r2"]:.4f}')
        axes[0].legend()

        res_holdout = y_holdout.values - y_pred_holdout
        axes[1].hist(res_holdout, bins=50, color='teal', edgecolor='white', alpha=0.8)
        axes[1].axvline(0, color='crimson', linewidth=1.5, linestyle='--', label='0')
        axes[1].set_xlabel('Resíduo ($)')
        axes[1].set_ylabel('Frequência')
        axes[1].set_title(f'Distribuição dos Resíduos — Holdout\nRMSE = {holdout_metrics["rmse"]:.0f}')
        axes[1].legend()
        fig.suptitle(f'Avaliação Holdout — {model_name}', fontsize=13, fontweight='bold')
        plt.tight_layout()
        return self._salvar(fig, 'holdout_evaluation.png')

    def plot_optuna_history(self, study: Any, model_name: str) -> Path | None:
        """Histórico de otimização por trial do Optuna."""
        if len(study.trials) <= 1:
            return None
        try:
            import optuna
            fig_ax = optuna.visualization.matplotlib.plot_optimization_history(study)
            fig_ax.figure.set_size_inches(10, 5)
            return self._salvar(fig_ax.figure, f'optuna_history_{model_name}.png')
        except Exception as exc:
            if self.logger:
                self.logger.warning('Plot optuna_history falhou para %s: %s', model_name, exc)
            return None

    def plot_optuna_param_importances(self, study: Any, model_name: str) -> Path | None:
        """Importância dos hiperparâmetros segundo o Optuna."""
        if len(study.trials) <= 1:
            return None
        try:
            import optuna
            fig_ax = optuna.visualization.matplotlib.plot_param_importances(study)
            fig_ax.figure.set_size_inches(10, 5)
            return self._salvar(fig_ax.figure, f'optuna_params_{model_name}.png')
        except Exception as exc:
            if self.logger:
                self.logger.warning('Plot optuna_params falhou para %s: %s', model_name, exc)
            return None

    # ── API de alto nível ─────────────────────────────────────────────────────

    def gerar_diagnosticos_modelo(
        self,
        model: Any,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        fold_metrics: list[dict],
    ) -> dict[str, Path | None]:
        """
        Gera todos os plots diagnósticos do modelo treinado no conjunto de treino.

        Retorna
        -------
        dict mapeando nome_do_plot → Path (ou None se o plot falhou)
        """
        y_pred_train = model.predict(X_train)
        residuos = y_train.values - y_pred_train
        from src.modeling.metrics import calcular_metricas
        metricas_treino = calcular_metricas(y_train.values, y_pred_train)

        plots_habilitados = self.artifacts_cfg.get('plots', [])
        caminhos: dict[str, Path | None] = {}

        if 'residuals' in plots_habilitados:
            caminhos['residuals'] = self.plot_residuals(y_pred_train, residuos, model_name)

        if 'predicted_vs_actual' in plots_habilitados:
            caminhos['predicted_vs_actual'] = self.plot_predicted_vs_actual(
                y_train, y_pred_train, metricas_treino['r2'], model_name
            )

        if 'error_distribution' in plots_habilitados:
            caminhos['error_distribution'] = self.plot_error_distribution(residuos, model_name)

        if 'feature_importance' in plots_habilitados:
            caminhos['feature_importance'] = self.plot_feature_importance(
                model, list(X_train.columns), X_train, y_train, model_name
            )

        if 'cv_fold_comparison' in plots_habilitados:
            caminhos['cv_fold_comparison'] = self.plot_cv_fold_comparison(fold_metrics, model_name)

        if 'learning_curve' in plots_habilitados:
            caminhos['learning_curve'] = self.plot_learning_curve(model, X_train, y_train, model_name)

        return caminhos, metricas_treino
