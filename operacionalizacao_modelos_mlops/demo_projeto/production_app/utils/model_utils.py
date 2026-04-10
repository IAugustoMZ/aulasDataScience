"""
production_app/utils/model_utils.py — Cliente MLflow local e auxiliares de IC.

Diferença-chave em relação ao ref_projeto:
  Este módulo NÃO faz chamadas REST a servidores externos.
  O modelo é carregado diretamente do banco SQLite via API Python do MLflow,
  eliminando a necessidade de `mlflow models serve` ou `mlflow server`.

Responsabilidades:
1. carregar_modelo(db_uri)
   Define o tracking URI do MLflow como SQLite e carrega o modelo registrado
   via mlflow.pyfunc.load_model().

2. prever_individual(features_df, modelo)
   Executa predição para uma única linha de features.

3. prever_lote(features_df, modelo)
   Executa predição em lote para múltiplas linhas de features.

4. obter_params_ic(db_uri)
   Consulta o MlflowClient para recuperar cv_rmse_std e holdout_rmse
   da versão mais recente do modelo registrado.

5. calcular_intervalo_confianca(y_hat, cv_rmse_std, n_folds, z)
   Calcula o intervalo de confiança de 95%:
       IC = y_hat ± 1,96 × (cv_rmse_std / √n_folds)

Nomenclatura do modelo (modeling.yaml → modeling.registry_name):
   "california-housing-best"
"""
from __future__ import annotations

import math
from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

_NOME_MODELO: str = "california-housing-best"
_N_FOLDS_CV: int  = 3      # modeling.yaml → cv.n_splits
_Z_95: float      = 1.96   # z-score para IC de 95%


# ─────────────────────────────────────────────────────────────────────────────
# 1. Carregamento do modelo
# ─────────────────────────────────────────────────────────────────────────────

def carregar_modelo(db_uri: str) -> mlflow.pyfunc.PyFuncModel:
    """
    Carrega o modelo registrado diretamente do banco SQLite do MLflow.

    Define o tracking URI para o SQLite especificado e carrega a versão
    mais recente do modelo registrado como 'california-housing-best'.

    Parâmetros
    ----------
    db_uri : str
        URI do banco SQLite do MLflow. Exemplos:
            "sqlite:///mlruns.db"
            "sqlite:////caminho/absoluto/mlruns.db"

    Retorna
    -------
    mlflow.pyfunc.PyFuncModel
        Modelo carregado, pronto para chamar .predict().

    Lança
    -----
    mlflow.exceptions.MlflowException
        Se o modelo não estiver registrado no banco especificado.
    """
    mlflow.set_tracking_uri(db_uri)
    return mlflow.pyfunc.load_model(f"models:/{_NOME_MODELO}/latest")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Predição
# ─────────────────────────────────────────────────────────────────────────────

def prever_individual(
    features_df: pd.DataFrame,
    modelo: mlflow.pyfunc.PyFuncModel,
) -> float:
    """
    Realiza predição para uma única linha de features.

    Parâmetros
    ----------
    features_df : pd.DataFrame
        DataFrame de uma linha com as colunas exatas esperadas pelo modelo.
    modelo : mlflow.pyfunc.PyFuncModel
        Modelo carregado via carregar_modelo().

    Retorna
    -------
    float
        Predição do valor médio imobiliário (em USD).
    """
    resultado = modelo.predict(features_df)
    return float(resultado[0])


def prever_lote(
    features_df: pd.DataFrame,
    modelo: mlflow.pyfunc.PyFuncModel,
) -> list[float]:
    """
    Realiza predição em lote para múltiplas linhas de features.

    Parâmetros
    ----------
    features_df : pd.DataFrame
        DataFrame com N linhas de features pré-processadas.
    modelo : mlflow.pyfunc.PyFuncModel
        Modelo carregado via carregar_modelo().

    Retorna
    -------
    list[float]
        Lista com as predições, uma por linha, na ordem de entrada.
    """
    resultado = modelo.predict(features_df)
    return [float(v) for v in resultado]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Parâmetros de IC via MLflow Python API
# ─────────────────────────────────────────────────────────────────────────────

def obter_params_ic(db_uri: str) -> dict[str, Any]:
    """
    Recupera cv_rmse_std e holdout_rmse do melhor run registrado no MLflow.

    Consulta o MlflowClient diretamente (sem chamadas HTTP), usando o
    banco SQLite local como backend de tracking.

    Passos:
        1. Obtém as versões registradas de 'california-housing-best'.
        2. Prefere versão em estágio 'Production'; se ausente, usa a mais recente.
        3. Busca as métricas do run associado a essa versão.

    Parâmetros
    ----------
    db_uri : str
        URI do banco SQLite do MLflow (mesmo usado em carregar_modelo).

    Retorna
    -------
    dict com chaves:
        cv_rmse_std   (float) — desvio padrão do RMSE entre os folds de CV
        holdout_rmse  (float) — RMSE no conjunto holdout
        run_id        (str)   — identificador do run no MLflow
        versao_modelo (str)   — número da versão no registry

    Lança
    -----
    ValueError
        Se nenhuma versão estiver registrada para o modelo.
    mlflow.exceptions.MlflowException
        Se o banco SQLite não existir ou não for acessível.
    """
    cliente = mlflow.MlflowClient(tracking_uri=db_uri)

    # ── Passo 1: listar versões registradas ───────────────────────────────────
    # Usa search_model_versions (API estável no MLflow ≥ 2.9.0) em vez de
    # get_latest_versions, que foi depreciado na versão 2.9.0.
    versoes = cliente.search_model_versions(f"name='{_NOME_MODELO}'")

    if not versoes:
        raise ValueError(
            f"Nenhuma versão registrada encontrada para o modelo '{_NOME_MODELO}'.\n"
            f"Execute notebooks/modelagem.py para treinar e registrar o modelo."
        )

    # ── Passo 2: selecionar versão (mais recente por número de versão) ────────
    melhor_versao = max(versoes, key=lambda v: int(v.version))

    run_id = melhor_versao.run_id
    versao = melhor_versao.version

    # ── Passo 3: buscar métricas do run ───────────────────────────────────────
    run = cliente.get_run(run_id)
    metricas = run.data.metrics

    cv_rmse_std  = float(metricas.get("cv_rmse_std", 0.0))
    holdout_rmse = float(metricas.get("holdout_rmse", metricas.get("rmse", 0.0)))

    return {
        "cv_rmse_std":   cv_rmse_std,
        "holdout_rmse":  holdout_rmse,
        "run_id":        run_id,
        "versao_modelo": versao,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Cálculo do Intervalo de Confiança
# ─────────────────────────────────────────────────────────────────────────────

def calcular_intervalo_confianca(
    y_hat: float,
    cv_rmse_std: float,
    n_folds: int = _N_FOLDS_CV,
    z: float = _Z_95,
) -> tuple[float, float]:
    """
    Calcula o intervalo de confiança de 95% para uma predição.

    Fórmula:
         SE  = cv_rmse_std / √n_folds
         IC  = y_hat ± z × SE

    Justificativa:
        cv_rmse_std / √n_folds é o erro padrão da estimativa média do RMSE
        entre os folds — uma medida princípiada da incerteza do modelo.

    Parâmetros
    ----------
    y_hat : float
        Predição pontual do modelo (USD).
    cv_rmse_std : float
        Desvio padrão do RMSE entre os folds de CV (do MLflow).
    n_folds : int
        Número de folds de CV usados no treino (padrão: 3, de modeling.yaml).
    z : float
        z-score para o nível de confiança desejado (padrão: 1.96 → 95%).

    Retorna
    -------
    (inferior, superior) : tuple[float, float]
        Limites inferior e superior do intervalo (USD).
    """
    se = cv_rmse_std / math.sqrt(n_folds)
    margem = z * se
    return (y_hat - margem, y_hat + margem)
