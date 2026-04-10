"""
transformers/stateful.py — Transformadores Stateful (aprendem parâmetros do treino).

⚠  AVISO MLOps — Data Leakage
   Estes transformadores aprendem estatísticas dos dados de treino no fit().
   NUNCA aplique fit() em todo o dataset antes do split treino/holdout.

   Fluxo correto:
       imputer.fit(X_treino).transform(X_treino)   # aprende no treino
       imputer.transform(X_holdout)                  # aplica no holdout

   Integração no pipeline de modelagem (modelagem.py):
       pipe = Pipeline([
           ('imputer', GroupMedianImputer(...)),    # ← aprende no treino
           ('scaler',  StandardScalerTransformer(...)),
           ('modelo',  Ridge()),
       ])
       pipe.fit(X_treino, y_treino)

   NÃO use estes transformadores no preprocessamento.py — esse script roda
   antes do split e aplicaria fit() em dados de teste, causando data leakage.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.preprocessing.base import BaseFeatureTransformer


class GroupMedianImputer(BaseFeatureTransformer):
    """
    Imputa valores ausentes usando a mediana do grupo (estratificada).

    Por que mediana por grupo?
    - total_bedrooms tem distribuição assimétrica (skew=3.46) → mediana > média.
    - O número de quartos varia sistematicamente por ocean_proximity.
    - Imputar com a mediana global ignora essa heterogeneidade regional.

    Atributos aprendidos no fit (APENAS no conjunto de treino):
        medians_       (dict): {valor_do_grupo → mediana}
        global_median_ (float): fallback para grupos não vistos no fit

    Raises:
        KeyError:    Se group_col ou target_col não existirem no DataFrame.
        RuntimeError: Se transform() for chamado antes de fit().
    """

    def __init__(
        self,
        group_col: str,
        target_col: str,
        logger: Any = None,
    ) -> None:
        self.group_col = group_col
        self.target_col = target_col
        self.logger = logger

    def fit(self, X: pd.DataFrame, y=None) -> "GroupMedianImputer":
        """Aprende a mediana de target_col para cada valor de group_col."""
        cols_ausentes = [c for c in [self.group_col, self.target_col] if c not in X.columns]
        if cols_ausentes:
            raise KeyError(
                f"GroupMedianImputer.fit: colunas ausentes no DataFrame: {cols_ausentes}"
            )

        self.medians_ = (
            X.groupby(self.group_col)[self.target_col]
            .median()
            .to_dict()
        )
        self.global_median_ = float(X[self.target_col].median())

        self._log(
            "GroupMedianImputer.fit: medianas aprendidas por '%s' para '%s': %s",
            self.group_col, self.target_col,
            {k: round(v, 1) for k, v in self.medians_.items()},
        )
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Preenche NaN em target_col com a mediana do grupo correspondente."""
        if not hasattr(self, "medians_"):
            raise RuntimeError(
                "GroupMedianImputer não foi ajustado. Chame fit() antes de transform()."
            )

        X = X.copy()
        n_antes = int(X[self.target_col].isna().sum())

        def _preencher(row: pd.Series) -> float:
            if pd.isna(row[self.target_col]):
                valor = self.medians_.get(row[self.group_col], self.global_median_)
                # Fallback para mediana global quando a mediana do grupo também é NaN
                # (ocorre quando todos os valores do grupo são nulos no conjunto de treino)
                return valor if not pd.isna(valor) else self.global_median_
            return row[self.target_col]

        X[self.target_col] = X.apply(_preencher, axis=1)
        n_depois = int(X[self.target_col].isna().sum())

        self._log(
            "GroupMedianImputer.transform: '%s' — NaN antes=%d, depois=%d",
            self.target_col, n_antes, n_depois,
        )
        return X


class StandardScalerTransformer(BaseFeatureTransformer):
    """
    Aplica Z-score normalization: z = (x − μ) / σ.

    Por que StandardScaler?
    - Regressão linear e SVM são sensíveis à escala das features.
    - Gradient boosting e Random Forest NÃO precisam de escalonamento.

    Parâmetros aprendidos no fit (APENAS no conjunto de treino):
        mean_  (dict): {coluna: média}
        std_   (dict): {coluna: desvio padrão}

    Colunas com std=0 são ignoradas (constantes — sem informação).

    Raises:
        RuntimeError: Se transform() for chamado antes de fit().
    """

    def __init__(self, columns: list[str], logger: Any = None) -> None:
        self.columns = columns
        self.logger = logger

    def fit(self, X: pd.DataFrame, y=None) -> "StandardScalerTransformer":
        """Aprende média e desvio padrão das colunas especificadas."""
        self.mean_: dict[str, float] = {}
        self.std_: dict[str, float] = {}
        ausentes: list[str] = []

        for col in self.columns:
            if col not in X.columns:
                ausentes.append(col)
                continue

            mu = float(X[col].mean())
            sigma = float(X[col].std())

            if sigma == 0:
                self._warn(
                    "StandardScalerTransformer.fit: '%s' tem std=0 (constante) — ignorada.", col
                )
                continue

            self.mean_[col] = mu
            self.std_[col] = sigma

        if ausentes:
            self._warn(
                "StandardScalerTransformer.fit: colunas ausentes ignoradas: %s", ausentes
            )

        self._log(
            "StandardScalerTransformer.fit: parâmetros aprendidos para %d colunas.",
            len(self.mean_),
        )
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Aplica Z-score nas colunas ajustadas no fit."""
        if not hasattr(self, "mean_"):
            raise RuntimeError(
                "StandardScalerTransformer não foi ajustado. Chame fit() antes de transform()."
            )

        X = X.copy()
        escalonadas: list[str] = []

        for col, mu in self.mean_.items():
            if col not in X.columns:
                continue
            X[col] = (X[col] - mu) / self.std_[col]
            escalonadas.append(col)

        self._log(
            "StandardScalerTransformer.transform: %d colunas escalonadas (z-score).",
            len(escalonadas),
        )
        return X

    @property
    def scale_params(self) -> pd.DataFrame:
        """Retorna DataFrame com média e desvio padrão aprendidos (útil para auditoria)."""
        return pd.DataFrame({"mean": self.mean_, "std": self.std_}).rename_axis("feature")
