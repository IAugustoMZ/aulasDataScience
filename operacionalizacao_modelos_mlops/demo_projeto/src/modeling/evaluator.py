"""
modeling/evaluator.py — Avaliador do modelo no conjunto holdout.

Responsabilidade única: executar a avaliação final em dados nunca vistos
durante o treinamento ou seleção de hiperparâmetros.

O holdout é o "cofre selado" — nunca entrou em nenhum fold de CV,
não influenciou a seleção de hiperparâmetros e não foi visto na escolha
do melhor modelo. É a estimativa mais honesta da performance em produção.
"""
from __future__ import annotations

import logging
from typing import Any

from src.modeling.base import BaseEvaluator
from src.modeling.metrics import calcular_metricas


class HoldoutEvaluator(BaseEvaluator):
    """
    Avalia o modelo final no conjunto holdout.

    Análise de robustez:
      • Holdout RMSE ≈ CV RMSE     → modelo generaliza bem
      • Holdout RMSE >> CV RMSE    → possível overfitting ou data leakage
      • Diferença < 10% é considerada aceitável em regressão tabular

    Parâmetros
    ----------
    logger : Logger opcional para diagnósticos de robustez
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger

    def avaliar(self, model: Any, X: Any, y: Any) -> dict:
        """
        Calcula métricas do modelo no conjunto holdout.

        Parâmetros
        ----------
        model : modelo treinado (sklearn Pipeline ou estimador)
        X     : features do holdout
        y     : target do holdout

        Retorna
        -------
        dict com rmse, mae, r2, mape
        """
        y_previsto = model.predict(X)
        return calcular_metricas(y.values, y_previsto)

    def diagnosticar_robustez(self, cv_rmse: float, holdout_rmse: float) -> str:
        """
        Compara o CV RMSE com o Holdout RMSE e emite diagnóstico de robustez.

        Parâmetros
        ----------
        cv_rmse      : RMSE médio de cross-validation
        holdout_rmse : RMSE no conjunto holdout

        Retorna
        -------
        str com o diagnóstico ('BOA', 'MODERADA' ou 'RUIM')
        """
        delta_pct = abs(holdout_rmse - cv_rmse) / cv_rmse * 100

        if self.logger:
            self.logger.info('── Análise de Robustez ──')
            self.logger.info('  CV RMSE (média)  : %.2f', cv_rmse)
            self.logger.info('  Holdout RMSE     : %.2f', holdout_rmse)
            self.logger.info('  Δ                : %.2f (%.1f%%)', holdout_rmse - cv_rmse, delta_pct)

        if delta_pct < 10:
            diagnostico = 'BOA'
            if self.logger:
                self.logger.info('  Diagnóstico      : ✓ Generalização BOA (Δ < 10%%)')
        elif delta_pct < 20:
            diagnostico = 'MODERADA'
            if self.logger:
                self.logger.info('  Diagnóstico      : ⚠ Generalização MODERADA (10%% ≤ Δ < 20%%)')
        else:
            diagnostico = 'RUIM'
            if self.logger:
                self.logger.warning('  Diagnóstico      : ✗ Generalização RUIM (Δ ≥ 20%%) — risco de overfitting!')

        return diagnostico
