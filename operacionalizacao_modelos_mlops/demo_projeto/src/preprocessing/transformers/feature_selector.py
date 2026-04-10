"""
transformers/feature_selector.py — Seletor de Features Final.

Seleciona o subconjunto de colunas configurado para modelagem, descartando
colunas brutas que foram substituídas por features engineered.
Stateless: fit() valida as colunas; transform() seleciona.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.preprocessing.base import BaseFeatureTransformer


class FeatureSelector(BaseFeatureTransformer):
    """
    Seleciona o conjunto final de features para modelagem.

    Por que seleção explícita?
    - Após as transformações, o DataFrame tem 40+ colunas (originais + engineered).
    - Colunas brutas de contagem (total_rooms, etc.) são substituídas por razões.
    - Manter apenas o que vai para o modelo previne vazamento acidental de dados.

    Comportamento tolerante:
    - Colunas ausentes geram WARNING (não exceção) — permite que o pipeline
      continue mesmo que uma transformação anterior tenha sido pulada.
    - Apenas as colunas disponíveis são selecionadas.

    Config (preprocessing.yaml → feature_selection.features_to_keep):
        - "median_income"
        - "bedrooms_per_room"
        - ...

    Exemplo:
        selector = FeatureSelector(
            features_to_keep=config['feature_selection']['features_to_keep'],
            logger=logger
        )
        df = selector.fit_transform(df)
    """

    def __init__(self, features_to_keep: list[str], logger: Any = None) -> None:
        self.features_to_keep = features_to_keep
        self.logger = logger

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureSelector":
        """Valida que todas as features configuradas existem no DataFrame."""
        ausentes = [c for c in self.features_to_keep if c not in X.columns]
        if ausentes:
            self._warn(
                "FeatureSelector.fit: %d colunas da config ausentes no DataFrame "
                "(serão ignoradas): %s",
                len(ausentes), ausentes,
            )
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Seleciona apenas as colunas disponíveis da lista configurada."""
        disponiveis = [c for c in self.features_to_keep if c in X.columns]
        descartadas = len(self.features_to_keep) - len(disponiveis)

        if descartadas > 0:
            self._warn(
                "FeatureSelector.transform: %d/%d colunas solicitadas ausentes — ignoradas.",
                descartadas, len(self.features_to_keep),
            )

        self._log(
            "FeatureSelector.transform: %d/%d features selecionadas | shape: %s → (%d, %d)",
            len(disponiveis), len(self.features_to_keep),
            X.shape, len(X), len(disponiveis),
        )
        return X[disponiveis].copy()
