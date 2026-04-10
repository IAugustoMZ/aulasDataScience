"""
preprocessing/base.py — Classe abstrata base para todos os transformadores de feature engineering.

Princípio de design:
  Centraliza o comportamento comum (logging, interface sklearn) em um único lugar,
  eliminando a duplicação do método _log() e do fit() no-op em cada transformador.

Interface herdada por todas as subclasses:
  - _log(msg, *args)  : emite logger.info se o logger estiver presente
  - fit(X, y)         : implementação padrão (retorna self) para transformadores stateless
  - transform(X, y)   : abstrato — cada subclasse implementa sua lógica específica
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BaseFeatureTransformer(BaseEstimator, TransformerMixin, ABC):
    """
    Classe base para todos os transformadores de pré-processamento do pipeline.

    Herança:
        BaseEstimator    → fornece get_params() / set_params() via introspecção do __init__
        TransformerMixin → fornece fit_transform(X) como self.fit(X).transform(X)
        ABC              → torna a classe abstrata; obriga subclasses a implementar transform()

    Uso:
        class MeuTransformador(BaseFeatureTransformer):
            def __init__(self, meu_param, logger=None):
                self.meu_param = meu_param
                self.logger = logger

            def transform(self, X, y=None):
                X = X.copy()
                # ... lógica de transformação ...
                self._log("MeuTransformador: concluído.")
                return X
    """

    logger: logging.Logger | None  # declarado para tipagem; definido no __init__ da subclasse

    def _log(self, msg: str, *args: Any) -> None:
        """Emite logger.info se o logger estiver configurado; caso contrário, é no-op."""
        if getattr(self, "logger", None) is not None:
            self.logger.info(msg, *args)  # type: ignore[union-attr]

    def _warn(self, msg: str, *args: Any) -> None:
        """Emite logger.warning se o logger estiver configurado; caso contrário, é no-op."""
        if getattr(self, "logger", None) is not None:
            self.logger.warning(msg, *args)  # type: ignore[union-attr]

    def fit(self, X: pd.DataFrame, y=None) -> "BaseFeatureTransformer":
        """
        Implementação padrão para transformadores stateless — retorna self sem aprender nada.

        Transformadores stateful (GroupMedianImputer, StandardScalerTransformer) devem
        sobrescrever este método para aprender parâmetros do conjunto de treino.
        """
        return self

    @abstractmethod
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Aplica a transformação ao DataFrame e retorna uma cópia modificada."""
