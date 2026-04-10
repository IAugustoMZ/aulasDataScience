"""
transformers/log_transform.py — Transformador Logarítmico (log1p).

Reduz a assimetria de distribuições com cauda longa criando colunas log_<nome>.
Stateless: fit() é no-op; transform() não aprende parâmetros dos dados.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing.base import BaseFeatureTransformer


class LogTransformer(BaseFeatureTransformer):
    """
    Aplica log1p(x) nas colunas especificadas, criando novas colunas com prefixo 'log_'.

    Por que log1p e não log?
    - log1p(x) = log(1+x) — seguro para x=0 (sem -Inf).
    - x é clipado em 0 antes da transformação (protege contra negativos).

    Colunas originais são mantidas. Novas colunas têm prefixo 'log_'.
    Exemplo: total_rooms → log_total_rooms.

    Config (preprocessing.yaml → log_transform.columns):
        - "total_rooms"
        - "population"

    Exemplo:
        transformer = LogTransformer(columns=config['log_transform']['columns'], logger=logger)
        df = transformer.fit_transform(df)
    """

    def __init__(self, columns: list[str], logger: Any = None) -> None:
        self.columns = columns
        self.logger = logger

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Aplica log1p em cada coluna configurada; emite warning para colunas ausentes."""
        X = X.copy()
        criadas: list[str] = []
        ausentes: list[str] = []

        for col in self.columns:
            if col not in X.columns:
                ausentes.append(col)
                continue

            log_col = f"log_{col}"
            skew_antes = float(X[col].dropna().skew())
            X[log_col] = np.log1p(X[col].clip(lower=0))
            skew_depois = float(X[log_col].dropna().skew())
            criadas.append(log_col)

            self._log(
                "LogTransformer: '%s' → '%s'  |  assimetria: %.2f → %.2f",
                col, log_col, skew_antes, skew_depois,
            )

        if ausentes:
            self._warn("LogTransformer: colunas não encontradas (ignoradas): %s", ausentes)

        return X
