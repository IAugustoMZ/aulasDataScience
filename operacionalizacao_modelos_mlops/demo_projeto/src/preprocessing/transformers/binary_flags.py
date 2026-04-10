"""
transformers/binary_flags.py — Transformador de Flags Binárias.

Adiciona colunas 0/1 para valores capados ou censurados no dataset.
Stateless: fit() é no-op; transform() não aprende parâmetros dos dados.

⚠  Atenção MLOps — Inferência:
   Flags baseadas em colunas disponíveis em produção (ex: housing_median_age)
   são seguras. Flags baseadas no TARGET (ex: median_house_value) NÃO devem
   ser incluídas em features_to_keep — o target não existe em inferência.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.preprocessing.base import BaseFeatureTransformer


class BinaryFlagTransformer(BaseFeatureTransformer):
    """
    Cria colunas binárias (0/1) marcando valores capados ou de borda.

    Por que flags binárias?
    - housing_median_age == 52: limite máximo de coleta — blocos com esse valor
      são sistematicamente mais antigos; o modelo deve aprender esse padrão.
    - Flags baseadas em colunas de entrada são seguras para inferência.

    Config (preprocessing.yaml → binary_flags):
        - column: "housing_median_age"
          value: 52
          new_column: "age_at_cap"
          inference_safe: true

    Exemplo:
        transformer = BinaryFlagTransformer(flags=config['binary_flags'], logger=logger)
        df = transformer.fit_transform(df)
    """

    def __init__(self, flags: list[dict], logger: Any = None) -> None:
        self.flags = flags
        self.logger = logger

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Aplica cada flag configurada; ignora colunas ausentes com warning."""
        X = X.copy()
        for spec in self.flags:
            col = spec["column"]
            val = spec["value"]
            new_col = spec["new_column"]

            if col not in X.columns:
                self._warn(
                    "BinaryFlagTransformer: coluna '%s' não encontrada — flag '%s' ignorada.",
                    col, new_col,
                )
                continue

            X[new_col] = (X[col] == val).astype(int)
            n_flagged = int(X[new_col].sum())
            self._log(
                "BinaryFlagTransformer: '%s' == %s → '%s': %d linhas flagadas (%.2f%%)",
                col, val, new_col, n_flagged, 100 * n_flagged / len(X),
            )
        return X
