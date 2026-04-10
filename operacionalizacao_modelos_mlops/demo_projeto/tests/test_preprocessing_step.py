"""
tests/test_preprocessing_step.py — Teste de integração do PreprocessingStep.

Valida o comportamento ponta-a-ponta da etapa de pré-processamento:
  - Cria um arquivo Parquet de entrada sintético em diretório temporário.
  - Executa PreprocessingStep.run() com configuração mínima via YAML temporário.
  - Verifica que o Parquet de saída foi criado com as colunas corretas.
  - Confirma que is_capped_target NÃO está nas features finais (anti-leakage).
  - Confirma que nenhum transformador stateful (GroupMedianImputer,
    StandardScalerTransformer) é aplicado nesta etapa.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
import yaml

from src.preprocessing.pipeline_builder import PreprocessingPipelineBuilder
from src.preprocessing.step import PreprocessingStep


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def logger_nulo() -> logging.Logger:
    """Logger silencioso para não poluir a saída dos testes."""
    logger = logging.getLogger("test_step_nulo")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


@pytest.fixture
def df_california() -> pd.DataFrame:
    """
    DataFrame California Housing sintético para testes de integração.
    Inclui NaN em total_bedrooms para garantir propagação correta.
    """
    return pd.DataFrame({
        "median_house_value":   [150_000.0, 250_000.0, 500_001.0, 350_000.0, 420_000.0],
        "median_income":        [3.5, 5.0, 7.2, 4.1, 6.3],
        "housing_median_age":   [20, 35, 52, 15, 28],
        "total_rooms":          [800.0, 1_200.0, 600.0, 900.0, 1_100.0],
        "total_bedrooms":       [200.0, None, 150.0, 225.0, 275.0],
        "population":           [500.0, 900.0, 400.0, 600.0, 750.0],
        "households":           [180.0, 280.0, 140.0, 200.0, 240.0],
        "latitude":             [37.88, 34.05, 32.72, 38.58, 37.33],
        "longitude":            [-122.23, -118.24, -117.16, -121.49, -121.89],
        "ocean_proximity":      ["NEAR BAY", "<1H OCEAN", "INLAND", "INLAND", "NEAR BAY"],
    })


@pytest.fixture
def config_integracao() -> dict[str, Any]:
    """
    Configuração mínima de pré-processamento para o teste de integração.
    Replicate a estrutura esperada por PreprocessingPipelineBuilder.
    """
    return {
        # Estrutura esperada pelo PreprocessingStep._carregar_config()
        "paths": {
            "processed_data_dir": "data/processed",
            "output_filename": "house_price.parquet",
        },
        "logging": {
            "level": "WARNING",
            "format": "%(message)s",
            "datefmt": "%H:%M:%S",
            "log_to_file": False,
            "log_file": "test.log",
        },
        "preprocessing": {
            "output_dir": "data/features",
            "output_filename": "house_price_features.parquet",
            "compression": "snappy",
        },
        "binary_flags": [
            {"column": "housing_median_age", "value": 52, "new_column": "age_at_cap", "inference_safe": True},
            {"column": "median_house_value",  "value": 500_001, "new_column": "is_capped_target", "inference_safe": False},
        ],
        "ratio_features": [
            {"name": "rooms_per_household",    "numerator": "total_rooms",    "denominator": "households"},
            {"name": "bedrooms_per_room",      "numerator": "total_bedrooms", "denominator": "total_rooms"},
            {"name": "population_per_household", "numerator": "population",   "denominator": "households"},
        ],
        "log_transform": {
            "columns": ["total_rooms", "population", "households"],
        },
        "geo_distances": {
            "lat_col": "latitude",
            "lon_col": "longitude",
            "nearest_city_column": "nearest_city_distance",
            "cities": [
                {"name": "san_francisco", "lat": 37.7749, "lon": -122.4194},
                {"name": "los_angeles",   "lat": 34.0522, "lon": -118.2437},
            ],
        },
        "polynomial_features": [
            {"name": "median_income_squared",            "columns": ["median_income"]},
            {"name": "income_x_age",                     "columns": ["median_income", "housing_median_age"]},
        ],
        "categorical_encoding": {
            "column": "ocean_proximity",
            "ordinal_column": "ocean_proximity_encoded",
            "ordinal_map": {"ISLAND": 0, "NEAR BAY": 1, "NEAR OCEAN": 2, "<1H OCEAN": 3, "INLAND": 4},
            "one_hot_prefix": "op",
            "drop_first": False,
        },
        "feature_selection": {
            "target": "median_house_value",
            "features_to_keep": [
                "median_income",
                "housing_median_age",
                "age_at_cap",
                "ocean_proximity_encoded",
                "op_INLAND",
                "op_NEAR BAY",
                "op_<1H OCEAN",
                "nearest_city_distance",
                "dist_san_francisco",
                "dist_los_angeles",
                "rooms_per_household",
                "bedrooms_per_room",
                "population_per_household",
                "log_total_rooms",
                "log_population",
                "log_households",
                "median_income_squared",
                "income_x_age",
                "latitude",
                "longitude",
                "median_house_value",
            ],
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _criar_context_mock(
    tmp_path: Path,
    df_entrada: pd.DataFrame,
    config: dict[str, Any],
    logger_nulo: logging.Logger,
) -> MagicMock:
    """
    Cria um mock de PipelineContext apontando para arquivos temporários em tmp_path.

    O Parquet de entrada é escrito em tmp_path/data/processed/house_price.parquet.
    """
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    parquet_entrada = processed_dir / "house_price.parquet"
    df_entrada.to_parquet(str(parquet_entrada), index=False)

    ctx = MagicMock()
    ctx.logger = logger_nulo
    ctx.root_dir = tmp_path
    ctx.config_dir = tmp_path / "config"
    ctx.config_dir.mkdir(parents=True, exist_ok=True)
    ctx.output_path = parquet_entrada
    ctx.pipeline_cfg = config

    # Salva preprocessing.yaml temporário para _carregar_config() do step
    prep_keys = [
        "preprocessing", "binary_flags", "ratio_features", "log_transform",
        "geo_distances", "polynomial_features", "categorical_encoding", "feature_selection",
    ]
    prep_cfg_dict = {k: config[k] for k in prep_keys if k in config}
    prep_yaml = ctx.config_dir / "preprocessing.yaml"
    with open(prep_yaml, "w", encoding="utf-8") as f:
        yaml.dump(prep_cfg_dict, f, allow_unicode=True)

    return ctx


# ─────────────────────────────────────────────────────────────────────────────
# TestPreprocessingPipelineBuilder
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessingPipelineBuilder:
    """Testes unitários do PreprocessingPipelineBuilder."""

    def test_build_retorna_pipeline_sklearn(self, config_integracao, logger_nulo):
        """build() deve retornar um sklearn.Pipeline."""
        from sklearn.pipeline import Pipeline
        builder = PreprocessingPipelineBuilder(config=config_integracao, logger=logger_nulo)
        pipeline = builder.build()
        assert isinstance(pipeline, Pipeline)

    def test_pipeline_tem_sete_etapas(self, config_integracao, logger_nulo):
        """Pipeline deve ter exatamente 7 etapas na ordem correta."""
        builder = PreprocessingPipelineBuilder(config=config_integracao, logger=logger_nulo)
        pipeline = builder.build()
        nomes = [nome for nome, _ in pipeline.steps]
        assert len(nomes) == 7
        assert nomes[0] == "flags_binarias"
        assert nomes[-1] == "selecao"

    def test_pipeline_fit_transform_retorna_dataframe(self, df_california, config_integracao, logger_nulo):
        """fit_transform() deve retornar um DataFrame com as features configuradas."""
        builder = PreprocessingPipelineBuilder(config=config_integracao, logger=logger_nulo)
        pipeline = builder.build()
        df_out = pipeline.fit_transform(df_california)
        assert isinstance(df_out, pd.DataFrame)
        assert len(df_out) == len(df_california)


# ─────────────────────────────────────────────────────────────────────────────
# TestPreprocessingStep — integração ponta-a-ponta
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessingStep:
    """Testes de integração do PreprocessingStep."""

    def test_run_cria_parquet_de_saida(self, tmp_path, df_california, config_integracao, logger_nulo):
        """run() deve criar o arquivo Parquet de saída."""
        ctx = _criar_context_mock(tmp_path, df_california, config_integracao, logger_nulo)
        step = PreprocessingStep(ctx)
        step.run()

        assert step.caminho_saida.exists(), f"Parquet de saída não criado: {step.caminho_saida}"

    def test_saida_e_dataframe_valido(self, tmp_path, df_california, config_integracao, logger_nulo):
        """O Parquet de saída deve ser legível e conter dados."""
        ctx = _criar_context_mock(tmp_path, df_california, config_integracao, logger_nulo)
        step = PreprocessingStep(ctx)
        step.run()

        df_out = pd.read_parquet(str(step.caminho_saida))
        assert len(df_out) == len(df_california)
        assert len(df_out.columns) > 0

    def test_is_capped_target_nao_esta_nas_features_finais(self, tmp_path, df_california, config_integracao, logger_nulo):
        """
        Garante proteção contra data leakage:
        is_capped_target depende do target (median_house_value) que não existe
        em inferência — não deve estar nas features de saída.
        """
        ctx = _criar_context_mock(tmp_path, df_california, config_integracao, logger_nulo)
        step = PreprocessingStep(ctx)
        step.run()

        df_out = pd.read_parquet(str(step.caminho_saida))
        assert "is_capped_target" not in df_out.columns, (
            "LEAKAGE DETECTADO: is_capped_target encontrado nas features de saída. "
            "Remova-a de features_to_keep no config/preprocessing.yaml."
        )

    def test_features_esperadas_presentes(self, tmp_path, df_california, config_integracao, logger_nulo):
        """Features configuradas em features_to_keep devem estar no Parquet de saída."""
        ctx = _criar_context_mock(tmp_path, df_california, config_integracao, logger_nulo)
        step = PreprocessingStep(ctx)
        step.run()

        df_out = pd.read_parquet(str(step.caminho_saida))
        features_esperadas = [
            "median_income",
            "housing_median_age",
            "age_at_cap",
            "rooms_per_household",
            "nearest_city_distance",
            "median_income_squared",
            "ocean_proximity_encoded",
        ]
        for feat in features_esperadas:
            assert feat in df_out.columns, f"Feature esperada ausente: {feat}"

    def test_nenhum_transformador_stateful_aplicado(self, tmp_path, df_california, config_integracao, logger_nulo):
        """
        Valida que NaN em bedrooms_per_room (propagado de total_bedrooms nulo)
        NÃO é imputado pela etapa de pré-processamento — a imputação é delegada
        ao pipeline de modelagem para evitar data leakage.
        """
        ctx = _criar_context_mock(tmp_path, df_california, config_integracao, logger_nulo)
        step = PreprocessingStep(ctx)
        step.run()

        df_out = pd.read_parquet(str(step.caminho_saida))
        if "bedrooms_per_room" in df_out.columns:
            # Linha 1 tinha total_bedrooms=None → bedrooms_per_room deve ser NaN
            assert pd.isna(df_out.loc[1, "bedrooms_per_room"]), (
                "bedrooms_per_room foi imputado antes do split — isso indica "
                "que um transformador stateful foi aplicado indevidamente nesta etapa."
            )

    def test_entrada_inexistente_levanta_file_not_found(self, tmp_path, df_california, config_integracao, logger_nulo):
        """FileNotFoundError deve ser levantado se o Parquet de entrada não existir."""
        ctx = _criar_context_mock(tmp_path, df_california, config_integracao, logger_nulo)
        # Remove o arquivo de entrada criado pelo helper
        ctx.output_path.unlink()
        step = PreprocessingStep(ctx)
        with pytest.raises(FileNotFoundError):
            step.run()
