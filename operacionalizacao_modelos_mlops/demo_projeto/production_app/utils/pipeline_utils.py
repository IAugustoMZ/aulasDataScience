"""
production_app/utils/pipeline_utils.py — Cadeia completa de pré-processamento para inferência.

Replica exatamente as transformações de preprocessamento.py usando as mesmas
classes de transformadores de src/ e as mesmas configurações do YAML.

Entrada bruta (9 features originais fornecidas pelo usuário):
    median_income, housing_median_age, total_rooms, total_bedrooms,
    population, households, latitude, longitude, ocean_proximity

Saída: DataFrame de uma única linha (ou múltiplas) pronto para predição —
contém apenas as colunas features_to_keep definidas em config/preprocessing.yaml,
excluindo o target (median_house_value).

Ordem da cadeia (espelha preprocessamento.py):
    1. GroupMedianImputer        — imputa NaN em total_bedrooms por grupo ocean_proximity
    2. BinaryFlagTransformer     — flags age_at_cap e is_capped_target
    3. RatioFeatureTransformer   — razões: rooms/household, bedrooms/room, population/household
    4. LogTransformer            — log1p nas colunas com alta assimetria
    5. GeoDistanceTransformer    — distâncias euclidianas para 5 cidades da Califórnia
    6. PolynomialFeatureTransformer — termos quadráticos e interação
    7. OceanProximityEncoder     — encoding ordinal + one-hot
    8. FeatureSelector           — mantém apenas features_to_keep (sem target)

O GroupMedianImputer é ajustado uma vez no parquet de treino para aprender as
medianas reais por grupo — idêntico ao pipeline de treinamento.
Para os demais transformadores stateless, fit() é um no-op.

NOTA: StandardScalerTransformer e FeatureReducer NÃO são aplicados aqui.
Eles residem dentro do sklearn Pipeline encapsulado pelo MLflow
(imputer → scaler → reducer → estimador) e são aplicados automaticamente
pelo modelo ao chamar model.predict().
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# ── Bootstrap de paths: torna src/ importável independentemente do CWD ────────
_HERE         = Path(__file__).resolve().parent   # production_app/utils/
_APP_DIR      = _HERE.parent                      # production_app/
_PROJECT_ROOT = _APP_DIR.parent                   # demo_projeto/
_CONFIG_DIR   = _PROJECT_ROOT / "config"
_DATA_DIR     = _PROJECT_ROOT / "data"

for _p in [str(_PROJECT_ROOT), str(_CONFIG_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.preprocessing import (
    GroupMedianImputer,
    BinaryFlagTransformer,
    RatioFeatureTransformer,
    LogTransformer,
    GeoDistanceTransformer,
    PolynomialFeatureTransformer,
    OceanProximityEncoder,
    FeatureSelector,
)


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento de config (cache em nível de módulo — lido apenas uma vez)
# ─────────────────────────────────────────────────────────────────────────────

def _carregar_config_preprocessing() -> dict[str, Any]:
    """Carrega preprocessing.yaml sem depender do PipelineContext completo."""
    caminho = _CONFIG_DIR / "preprocessing.yaml"
    with open(caminho, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


_CFG_PREP: dict[str, Any] = _carregar_config_preprocessing()

# Coluna target e lista de features a manter (excluindo o target)
_TARGET_COL: str = _CFG_PREP.get("feature_selection", {}).get("target", "median_house_value")
_FEATURES_TO_KEEP: list[str] = [
    c for c in _CFG_PREP.get("feature_selection", {}).get("features_to_keep", [])
    if c != _TARGET_COL
]

# ── Caminhos dos parquets ─────────────────────────────────────────────────────
_PARQUET_PROCESSADO = _DATA_DIR / "processed" / "house_price.parquet"
_PARQUET_FEATURES   = _DATA_DIR / "features"  / "house_price_features.parquet"


# ─────────────────────────────────────────────────────────────────────────────
# Imputador stateful — ajustado uma vez nos dados de treino
# ─────────────────────────────────────────────────────────────────────────────

def _construir_imputador_ajustado() -> GroupMedianImputer:
    """
    Ajusta o GroupMedianImputer no parquet processado completo.

    Usa a mesma configuração do pipeline de treinamento para que as medianas
    aprendidas sejam idênticas às usadas durante o treino do modelo.

    Lança
    -----
    FileNotFoundError
        Se o parquet processado não existir (ingestão ainda não foi executada).
    """
    if not _PARQUET_PROCESSADO.exists():
        raise FileNotFoundError(
            f"Parquet processado não encontrado: {_PARQUET_PROCESSADO}\n"
            "Execute notebooks/ingestao.py antes de iniciar a aplicação."
        )

    cfg_imp = _CFG_PREP.get("imputation", [{}])[0]
    imputador = GroupMedianImputer(
        group_col=cfg_imp.get("group_by", "ocean_proximity"),
        target_col=cfg_imp.get("column", "total_bedrooms"),
    )
    df_treino = pd.read_parquet(_PARQUET_PROCESSADO)
    imputador.fit(df_treino)
    return imputador


# Inicializa uma vez; o Streamlit reutiliza o objeto no nível de módulo nos reruns.
_IMPUTADOR_AJUSTADO: GroupMedianImputer = _construir_imputador_ajustado()


# ─────────────────────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────────────────────

def preprocessar_entradas(raw: dict[str, Any]) -> pd.DataFrame:
    """
    Converte um dicionário de entradas brutas do usuário em features prontas para o modelo.

    Aplica a cadeia completa de feature engineering na mesma ordem do pipeline
    de treinamento, garantindo consistência entre treino e inferência.

    Parâmetros
    ----------
    raw : dict
        Chaves obrigatórias:
            median_income, housing_median_age, total_rooms, total_bedrooms,
            population, households, latitude, longitude, ocean_proximity

    Retorna
    -------
    pd.DataFrame
        DataFrame de uma linha com as colunas features_to_keep (sem target).
        Os nomes das colunas são sanitizados para compatibilidade com XGBoost
        (< → lt_, [ → (, ] → )).

    Lança
    -----
    KeyError
        Se alguma chave obrigatória estiver ausente em `raw`.
    """
    df = pd.DataFrame([raw])

    # 1. Imputar total_bedrooms (stateful — usa medianas aprendidas no treino)
    df = _IMPUTADOR_AJUSTADO.transform(df)

    # 2. Flags binárias
    df = BinaryFlagTransformer(
        flags=_CFG_PREP.get("binary_flags", [])
    ).fit_transform(df)

    # 3. Features de razão
    df = RatioFeatureTransformer(
        ratios=_CFG_PREP.get("ratio_features", [])
    ).fit_transform(df)

    # 4. Transformações logarítmicas
    df = LogTransformer(
        columns=_CFG_PREP.get("log_transform", {}).get("columns", [])
    ).fit_transform(df)

    # 5. Distâncias geográficas
    df = GeoDistanceTransformer(
        geo_config=_CFG_PREP.get("geo_distances", {})
    ).fit_transform(df)

    # 6. Features polinomiais e interações
    df = PolynomialFeatureTransformer(
        poly_config=_CFG_PREP.get("polynomial_features", [])
    ).fit_transform(df)

    # 7. Encoding de ocean_proximity
    df = OceanProximityEncoder(
        enc_config=_CFG_PREP.get("categorical_encoding", {})
    ).fit_transform(df)

    # 8. Seleção de features (sem target)
    df = FeatureSelector(
        features_to_keep=_FEATURES_TO_KEEP
    ).fit_transform(df)

    # 9. Reindexar para garantir todas as colunas dummy (uma linha ativa apenas
    #    a categoria correspondente; as demais ficam ausentes sem reindex).
    df = df.reindex(columns=_FEATURES_TO_KEEP, fill_value=0)

    # 10. Sanitizar nomes de colunas para XGBoost (< → lt_, [ → (, ] → ))
    mapa_rename = {
        c: c.replace("<", "lt_").replace("[", "(").replace("]", ")")
        for c in df.columns
        if any(ch in c for ch in ("<", "[", "]"))
    }
    if mapa_rename:
        df = df.rename(columns=mapa_rename)

    return df


def obter_parquet_features() -> pd.DataFrame:
    """
    Retorna o parquet de features completo (usado pela página de monitoramento).

    Lança
    -----
    FileNotFoundError
        Se o parquet de features não existir (preprocessamento ainda não executado).
    """
    if not _PARQUET_FEATURES.exists():
        raise FileNotFoundError(
            f"Parquet de features não encontrado: {_PARQUET_FEATURES}\n"
            "Execute notebooks/preprocessamento.py antes de iniciar a aplicação."
        )
    return pd.read_parquet(_PARQUET_FEATURES)


def obter_colunas_features() -> list[str]:
    """Retorna os nomes das features com sanitização XGBoost (como enviados ao modelo)."""
    return [
        c.replace("<", "lt_").replace("[", "(").replace("]", ")")
        for c in _FEATURES_TO_KEEP
    ]


def obter_colunas_features_brutas() -> list[str]:
    """Retorna os nomes originais das features (como aparecem no parquet)."""
    return list(_FEATURES_TO_KEEP)
