"""
transformers/polynomial_features.py — Transformador de Features Polinomiais e Interações.

Cria features quadráticas (x²) e termos de interação (x₁ × x₂).
Stateless: fit() é no-op; transform() não aprende parâmetros dos dados.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.preprocessing.base import BaseFeatureTransformer


class PolynomialFeatureTransformer(BaseFeatureTransformer):
    """
    Cria features quadráticas e termos de interação.

    Tipos suportados (inferidos pelo número de colunas):
    - 1 coluna  → quadrado:  name = columns[0]²
    - 2 colunas → interação: name = columns[0] × columns[1]

    Por que features polinomiais?
    - A relação entre renda e preço não é perfeitamente linear.
    - median_income_squared captura retorno decrescente em alta renda.
    - median_income_x_housing_median_age (r=0.589): bairros ricos E antigos são premium.

    Config (preprocessing.yaml → polynomial_features):
        - name: "median_income_squared"
          columns: ["median_income"]
        - name: "median_income_x_housing_median_age"
          columns: ["median_income", "housing_median_age"]

    Exemplo:
        transformer = PolynomialFeatureTransformer(
            poly_config=config['polynomial_features'], logger=logger
        )
        df = transformer.fit_transform(df)
    """

    def __init__(self, poly_config: list[dict], logger: Any = None) -> None:
        self.poly_config = poly_config
        self.logger = logger

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Cria cada feature polinomial configurada; ignora specs com colunas ausentes."""
        X = X.copy()
        criadas: list[str] = []

        for spec in self.poly_config:
            name = spec["name"]
            cols = spec["columns"]
            ausentes = [c for c in cols if c not in X.columns]

            if ausentes:
                self._warn(
                    "PolynomialFeatureTransformer: colunas ausentes %s — '%s' ignorada.",
                    ausentes, name,
                )
                continue

            if len(cols) == 1:
                X[name] = X[cols[0]] ** 2
            elif len(cols) == 2:
                X[name] = X[cols[0]] * X[cols[1]]
            else:
                self._warn(
                    "PolynomialFeatureTransformer: '%s' tem %d colunas — apenas 1 ou 2 suportadas.",
                    name, len(cols),
                )
                continue

            criadas.append(name)

        self._log("PolynomialFeatureTransformer: features criadas: %s", criadas)
        return X
