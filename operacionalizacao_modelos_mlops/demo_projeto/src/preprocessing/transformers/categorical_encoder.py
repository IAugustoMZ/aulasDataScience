"""
transformers/categorical_encoder.py — Encoder de ocean_proximity.

Aplica encoding ordinal e one-hot na variável categórica ocean_proximity.
Stateless: fit() é no-op; transform() não aprende parâmetros dos dados.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.preprocessing.base import BaseFeatureTransformer


class OceanProximityEncoder(BaseFeatureTransformer):
    """
    Codifica ocean_proximity em duas representações complementares:

    1. Encoding ordinal (ocean_proximity_encoded):
       Mapa configurado por distância ao oceano: ISLAND=0 ... INLAND=4.
       Útil para modelos baseados em árvore e correlações ordinais.

    2. One-hot dummies (op_INLAND, op_NEAR BAY, etc.):
       Necessárias para regressão linear (sem assumir ordem).
       drop_first=False mantém todas as categorias para máxima transparência.

    Por que dual encoding?
    - ANOVA η²=0.238: ocean_proximity sozinha explica 23.8% da variância.
    - op_INLAND é a dummy mais preditiva: r=-0.485 com o target.

    Config (preprocessing.yaml → categorical_encoding):
        column: "ocean_proximity"
        ordinal_column: "ocean_proximity_encoded"
        ordinal_map: {ISLAND: 0, NEAR BAY: 1, NEAR OCEAN: 2, "<1H OCEAN": 3, INLAND: 4}
        one_hot_prefix: "op"
        drop_first: false

    Exemplo:
        encoder = OceanProximityEncoder(enc_config=config['categorical_encoding'], logger=logger)
        df = encoder.fit_transform(df)
    """

    def __init__(self, enc_config: dict, logger: Any = None) -> None:
        self.enc_config = enc_config
        self.logger = logger

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Aplica encoding ordinal e one-hot; retorna X inalterado se a coluna estiver ausente."""
        column = self.enc_config.get("column", "ocean_proximity")
        ordinal_column = self.enc_config.get("ordinal_column", "ocean_proximity_encoded")
        ordinal_map: dict = self.enc_config.get("ordinal_map", {})
        prefix = self.enc_config.get("one_hot_prefix", "op")
        drop_first: bool = self.enc_config.get("drop_first", False)

        if column not in X.columns:
            self._warn(
                "OceanProximityEncoder: coluna '%s' não encontrada — encoding ignorado.",
                column,
            )
            return X

        X = X.copy()

        # ── Encoding ordinal ──────────────────────────────────────────────────
        X[ordinal_column] = X[column].map(ordinal_map)
        n_desconhecidos = int(X[ordinal_column].isna().sum())
        if n_desconhecidos > 0:
            self._warn(
                "OceanProximityEncoder: %d linhas com valores de '%s' não mapeados → NaN",
                n_desconhecidos, column,
            )
        self._log(
            "OceanProximityEncoder: ordinal '%s' criado — mapa: %s",
            ordinal_column, ordinal_map,
        )

        # ── One-hot dummies ───────────────────────────────────────────────────
        dummies = pd.get_dummies(X[column], prefix=prefix, drop_first=drop_first).astype(int)
        X = pd.concat([X, dummies], axis=1)
        self._log("OceanProximityEncoder: dummies criadas: %s", list(dummies.columns))

        return X
