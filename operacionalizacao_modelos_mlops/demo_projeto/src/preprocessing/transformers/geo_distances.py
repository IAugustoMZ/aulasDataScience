"""
transformers/geo_distances.py — Transformador de Distâncias Geográficas.

Calcula a distância euclidiana de cada bloco a cidades de referência da Califórnia.
Stateless: fit() é no-op; transform() não aprende parâmetros dos dados.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing.base import BaseFeatureTransformer


class GeoDistanceTransformer(BaseFeatureTransformer):
    """
    Calcula a distância euclidiana (em graus) de cada bloco a cidades de referência.

    Por que distância euclidiana em graus?
    - O dataset usa graus diretamente; distância euclidiana é uma boa
      aproximação local (Califórnia abrange ~10° lat × ~10° lon).
    - EDA: nearest_city_distance tem r=-0.384 com o target.

    Colunas criadas:
    - dist_<city_name>      para cada cidade configurada
    - nearest_city_distance = mínimo de todas as dist_*

    Config (preprocessing.yaml → geo_distances):
        lat_col: "latitude"
        lon_col: "longitude"
        nearest_city_column: "nearest_city_distance"
        cities:
          - name: "san_francisco"
            lat: 37.7749
            lon: -122.4194

    Exemplo:
        transformer = GeoDistanceTransformer(geo_config=config['geo_distances'], logger=logger)
        df = transformer.fit_transform(df)
    """

    def __init__(self, geo_config: dict, logger: Any = None) -> None:
        self.geo_config = geo_config
        self.logger = logger

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Calcula distâncias e adiciona nearest_city_distance; retorna X inalterado se faltar lat/lon."""
        cidades = self.geo_config.get("cities", [])
        lat_col = self.geo_config.get("lat_col", "latitude")
        lon_col = self.geo_config.get("lon_col", "longitude")
        nearest_col = self.geo_config.get("nearest_city_column", "nearest_city_distance")

        if lat_col not in X.columns or lon_col not in X.columns:
            self._warn(
                "GeoDistanceTransformer: colunas '%s'/'%s' ausentes — transformação ignorada.",
                lat_col, lon_col,
            )
            return X

        X = X.copy()
        colunas_dist: list[str] = []

        for cidade in cidades:
            nome = cidade["name"]
            col_nome = f"dist_{nome}"
            X[col_nome] = np.sqrt(
                (X[lat_col] - cidade["lat"]) ** 2 +
                (X[lon_col] - cidade["lon"]) ** 2
            )
            colunas_dist.append(col_nome)

        if colunas_dist:
            X[nearest_col] = X[colunas_dist].min(axis=1)
            self._log(
                "GeoDistanceTransformer: %d distâncias calculadas: %s | '%s' adicionado.",
                len(colunas_dist), colunas_dist, nearest_col,
            )

        return X
