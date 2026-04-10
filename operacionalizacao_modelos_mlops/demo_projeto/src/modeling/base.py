"""
modeling/base.py — Classes abstratas base do módulo de modelagem.

Define os contratos principais:
  - BaseOptimizer    : qualquer estratégia de otimização de hiperparâmetros
  - BaseEvaluator    : qualquer avaliador de modelo (holdout, CV externo, etc.)

Princípio SOLID:
  - Single Responsibility: cada ABC define um único contrato
  - Open/Closed: novas estratégias (ex: HyperBand) são adicionadas sem mudar o código existente
  - Liskov Substitution: qualquer implementação concreta é intercambiável
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseOptimizer(ABC):
    """
    Contrato para otimizadores de hiperparâmetros.

    Implementações concretas:
      - OptunaOptimizer         : busca bayesiana (TPE)
      - GridSearchOptimizer     : busca exaustiva em grade
      - RandomizedSearchOptimizer: busca aleatória amostrada
    """

    @abstractmethod
    def otimizar(
        self,
        model_name: str,
        model_cfg: dict,
        X_tune: Any,
        y_tune: Any,
        pipe_cfg: dict,
        feat_red_cfg: dict,
    ) -> dict:
        """
        Executa a otimização e retorna um dicionário com os melhores parâmetros.

        Args:
            model_name   : nome do modelo (chave no models_cfg)
            model_cfg    : configuração do modelo (module, class, search_space, etc.)
            X_tune       : features de treino (ou subsample para modelos lentos)
            y_tune       : target de treino
            pipe_cfg     : configuração do pipeline (imputation, scaling)
            feat_red_cfg : configuração de redução de features

        Returns:
            dict com chaves 'estimator_params' e 'reducer_params'
        """


class BaseEvaluator(ABC):
    """
    Contrato para avaliadores de modelo.

    Implementações concretas:
      - HoldoutEvaluator: avalia no conjunto holdout separado antes de qualquer treino
    """

    @abstractmethod
    def avaliar(self, model: Any, X: Any, y: Any) -> dict:
        """
        Avalia o modelo e retorna dicionário de métricas.

        Args:
            model : modelo treinado (sklearn Pipeline ou estimador)
            X     : features de avaliação
            y     : target de avaliação

        Returns:
            dict com rmse, mae, r2, mape
        """
