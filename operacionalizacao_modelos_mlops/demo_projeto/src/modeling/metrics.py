"""
modeling/metrics.py — Funções de cálculo e agregação de métricas de regressão.

Responsabilidade única: computar e agregar métricas de avaliação.
Nenhuma dependência de MLflow, Optuna ou sklearn.Pipeline — funções puras.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)


def calcular_metricas(y_verdadeiro: np.ndarray, y_previsto: np.ndarray) -> dict:
    """
    Calcula as quatro métricas de regressão usadas no pipeline.

    Parâmetros
    ----------
    y_verdadeiro : array-like com os valores reais do target
    y_previsto   : array-like com as previsões do modelo

    Retorna
    -------
    dict com chaves: rmse, mae, r2, mape
    """
    rmse = float(np.sqrt(mean_squared_error(y_verdadeiro, y_previsto)))
    mae  = float(mean_absolute_error(y_verdadeiro, y_previsto))
    r2   = float(r2_score(y_verdadeiro, y_previsto))
    mape = float(mean_absolute_percentage_error(y_verdadeiro, y_previsto) * 100)
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}


def agregar_metricas_folds(fold_metrics: list[dict]) -> dict:
    """
    Agrega métricas de todos os folds em média ± desvio padrão.

    Parâmetros
    ----------
    fold_metrics : lista de dicts com chaves rmse, mae, r2, mape (e fold)

    Retorna
    -------
    dict com cv_{metrica}_mean e cv_{metrica}_std para cada métrica
    """
    df = pd.DataFrame(fold_metrics)
    resultado = {}
    for col in ['rmse', 'mae', 'r2', 'mape']:
        resultado[f'cv_{col}_mean'] = float(df[col].mean())
        resultado[f'cv_{col}_std']  = float(df[col].std())
    return resultado
