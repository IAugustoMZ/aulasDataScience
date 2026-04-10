"""
tests/test_preprocessing_transformers.py — Testes unitários dos transformadores de pré-processamento.

Cobre:
  - BinaryFlagTransformer       : criação de flags corretas, coluna ausente gera warning
  - RatioFeatureTransformer     : razões calculadas, divisão por zero → NaN, coluna ausente
  - LogTransformer              : colunas log_ criadas, valores negativos clipados, ausentes
  - GeoDistanceTransformer      : distâncias calculadas, nearest_city_distance correto, ausentes
  - PolynomialFeatureTransformer: quadrado (1 coluna), produto (2 colunas), ausentes
  - OceanProximityEncoder       : encoding ordinal, dummies geradas, coluna ausente
  - FeatureSelector             : seleção correta, ausentes geram warning (não exceção)
  - GroupMedianImputer          : fit aprende medianas, NaN preenchidos, transform sem fit levanta RuntimeError
  - StandardScalerTransformer   : z-score correto, std=0 ignorado, transform sem fit levanta RuntimeError
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.transformers import (
    BinaryFlagTransformer,
    FeatureSelector,
    GeoDistanceTransformer,
    GroupMedianImputer,
    LogTransformer,
    OceanProximityEncoder,
    PolynomialFeatureTransformer,
    RatioFeatureTransformer,
    StandardScalerTransformer,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures compartilhadas
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def logger_nulo() -> logging.Logger:
    """Logger silencioso para não poluir a saída dos testes."""
    logger = logging.getLogger("test_transformers_nulo")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


@pytest.fixture
def df_base() -> pd.DataFrame:
    """
    DataFrame mínimo no estilo California Housing para usar nos testes.
    Inclui NaN em total_bedrooms para testar imputação.
    """
    return pd.DataFrame({
        "median_house_value":   [150_000.0, 250_000.0, 500_001.0, 350_000.0],
        "median_income":        [3.5, 5.0, 7.2, 4.1],
        "housing_median_age":   [20, 35, 52, 15],
        "total_rooms":          [800.0, 1_200.0, 600.0, 900.0],
        "total_bedrooms":       [200.0, None, 150.0, 225.0],
        "population":           [500.0, 900.0, 400.0, 600.0],
        "households":           [180.0, 280.0, 140.0, 200.0],
        "latitude":             [37.88, 34.05, 32.72, 38.58],
        "longitude":            [-122.23, -118.24, -117.16, -121.49],
        "ocean_proximity":      ["NEAR BAY", "<1H OCEAN", "INLAND", "INLAND"],
    })


# ─────────────────────────────────────────────────────────────────────────────
# BinaryFlagTransformer
# ─────────────────────────────────────────────────────────────────────────────

class TestBinaryFlagTransformer:
    """Testes para BinaryFlagTransformer."""

    @pytest.fixture
    def flags_cfg(self) -> list[dict]:
        return [
            {"column": "housing_median_age", "value": 52, "new_column": "age_at_cap", "inference_safe": True},
            {"column": "median_house_value", "value": 500_001, "new_column": "is_capped_target", "inference_safe": False},
        ]

    def test_cria_flags_corretamente(self, df_base, flags_cfg, logger_nulo):
        """Deve criar colunas binárias com os valores corretos."""
        t = BinaryFlagTransformer(flags=flags_cfg, logger=logger_nulo)
        df_out = t.fit_transform(df_base)

        # age_at_cap: apenas linha com housing_median_age == 52
        assert "age_at_cap" in df_out.columns
        assert int(df_out["age_at_cap"].sum()) == 1
        assert df_out.loc[df_out["housing_median_age"] == 52, "age_at_cap"].iloc[0] == 1

        # is_capped_target: apenas linha com median_house_value == 500001
        assert "is_capped_target" in df_out.columns
        assert int(df_out["is_capped_target"].sum()) == 1

    def test_nao_modifica_original(self, df_base, flags_cfg, logger_nulo):
        """Não deve modificar o DataFrame original (retorna cópia)."""
        t = BinaryFlagTransformer(flags=flags_cfg, logger=logger_nulo)
        t.fit_transform(df_base)
        assert "age_at_cap" not in df_base.columns

    def test_coluna_ausente_emite_warning_nao_excecao(self, df_base, logger_nulo):
        """Coluna ausente deve gerar warning e pular a flag, sem levantar exceção."""
        cfg = [{"column": "coluna_inexistente", "value": 1, "new_column": "flag_x", "inference_safe": True}]
        t = BinaryFlagTransformer(flags=cfg, logger=logger_nulo)
        df_out = t.fit_transform(df_base)
        assert "flag_x" not in df_out.columns

    def test_valores_sao_zero_ou_um(self, df_base, flags_cfg, logger_nulo):
        """Todos os valores das flags devem ser 0 ou 1 (dtype int)."""
        t = BinaryFlagTransformer(flags=flags_cfg, logger=logger_nulo)
        df_out = t.fit_transform(df_base)
        for spec in flags_cfg:
            col = spec["new_column"]
            assert set(df_out[col].unique()).issubset({0, 1}), f"{col} contém valores além de 0 e 1"


# ─────────────────────────────────────────────────────────────────────────────
# RatioFeatureTransformer
# ─────────────────────────────────────────────────────────────────────────────

class TestRatioFeatureTransformer:
    """Testes para RatioFeatureTransformer."""

    @pytest.fixture
    def ratios_cfg(self) -> list[dict]:
        return [
            {"name": "rooms_per_household", "numerator": "total_rooms", "denominator": "households"},
            {"name": "bedrooms_per_room",   "numerator": "total_bedrooms", "denominator": "total_rooms"},
        ]

    def test_cria_razoes_corretas(self, df_base, ratios_cfg, logger_nulo):
        """Deve calcular razões numericamente corretas."""
        t = RatioFeatureTransformer(ratios=ratios_cfg, logger=logger_nulo)
        df_out = t.fit_transform(df_base)

        assert "rooms_per_household" in df_out.columns
        # linha 0: 800 / 180 ≈ 4.444
        assert abs(df_out.loc[0, "rooms_per_household"] - 800 / 180) < 1e-6

    def test_total_bedrooms_nan_propaga_para_razao(self, df_base, ratios_cfg, logger_nulo):
        """NaN no numerador deve resultar em NaN na razão (propagação correta)."""
        t = RatioFeatureTransformer(ratios=ratios_cfg, logger=logger_nulo)
        df_out = t.fit_transform(df_base)
        # linha 1 tem total_bedrooms=None → bedrooms_per_room deve ser NaN
        assert pd.isna(df_out.loc[1, "bedrooms_per_room"])

    def test_divisao_por_zero_gera_nan(self, logger_nulo):
        """Denominador zero deve resultar em NaN, não Inf."""
        df = pd.DataFrame({"num": [10.0], "den": [0.0]})
        cfg = [{"name": "razao", "numerator": "num", "denominator": "den"}]
        t = RatioFeatureTransformer(ratios=cfg, logger=logger_nulo)
        df_out = t.fit_transform(df)
        assert pd.isna(df_out.loc[0, "razao"])

    def test_coluna_ausente_ignorada(self, df_base, logger_nulo):
        """Par de colunas ausente deve ser ignorado sem exceção."""
        cfg = [{"name": "razao_x", "numerator": "inexistente", "denominator": "total_rooms"}]
        t = RatioFeatureTransformer(ratios=cfg, logger=logger_nulo)
        df_out = t.fit_transform(df_base)
        assert "razao_x" not in df_out.columns


# ─────────────────────────────────────────────────────────────────────────────
# LogTransformer
# ─────────────────────────────────────────────────────────────────────────────

class TestLogTransformer:
    """Testes para LogTransformer."""

    def test_cria_colunas_log(self, df_base, logger_nulo):
        """Deve criar colunas log_<nome> para cada coluna configurada."""
        t = LogTransformer(columns=["total_rooms", "population"], logger=logger_nulo)
        df_out = t.fit_transform(df_base)
        assert "log_total_rooms" in df_out.columns
        assert "log_population" in df_out.columns

    def test_valores_log_corretos(self, df_base, logger_nulo):
        """Valores log devem ser numericamente iguais a log1p(original)."""
        t = LogTransformer(columns=["total_rooms"], logger=logger_nulo)
        df_out = t.fit_transform(df_base)
        esperado = np.log1p(df_base["total_rooms"].clip(lower=0))
        pd.testing.assert_series_equal(df_out["log_total_rooms"], esperado, check_names=False)

    def test_coluna_original_preservada(self, df_base, logger_nulo):
        """A coluna original não deve ser removida."""
        t = LogTransformer(columns=["total_rooms"], logger=logger_nulo)
        df_out = t.fit_transform(df_base)
        assert "total_rooms" in df_out.columns

    def test_valor_negativo_clipado(self, logger_nulo):
        """Valores negativos devem ser clipados em 0 antes do log1p (resultado: log1p(0) = 0)."""
        df = pd.DataFrame({"col": [-5.0, 0.0, 10.0]})
        t = LogTransformer(columns=["col"], logger=logger_nulo)
        df_out = t.fit_transform(df)
        assert df_out.loc[0, "log_col"] == np.log1p(0)

    def test_coluna_ausente_ignorada(self, df_base, logger_nulo):
        """Coluna ausente não deve lançar exceção."""
        t = LogTransformer(columns=["coluna_inexistente"], logger=logger_nulo)
        df_out = t.fit_transform(df_base)
        assert "log_coluna_inexistente" not in df_out.columns


# ─────────────────────────────────────────────────────────────────────────────
# GeoDistanceTransformer
# ─────────────────────────────────────────────────────────────────────────────

class TestGeoDistanceTransformer:
    """Testes para GeoDistanceTransformer."""

    @pytest.fixture
    def geo_cfg(self) -> dict:
        return {
            "lat_col": "latitude",
            "lon_col": "longitude",
            "nearest_city_column": "nearest_city_distance",
            "cities": [
                {"name": "san_francisco", "lat": 37.7749, "lon": -122.4194},
                {"name": "los_angeles",   "lat": 34.0522, "lon": -118.2437},
            ],
        }

    def test_cria_colunas_de_distancia(self, df_base, geo_cfg, logger_nulo):
        """Deve criar dist_<cidade> para cada cidade e nearest_city_distance."""
        t = GeoDistanceTransformer(geo_config=geo_cfg, logger=logger_nulo)
        df_out = t.fit_transform(df_base)
        assert "dist_san_francisco" in df_out.columns
        assert "dist_los_angeles" in df_out.columns
        assert "nearest_city_distance" in df_out.columns

    def test_nearest_city_distance_e_minimo(self, df_base, geo_cfg, logger_nulo):
        """nearest_city_distance deve ser igual ao mínimo das distâncias individuais."""
        t = GeoDistanceTransformer(geo_config=geo_cfg, logger=logger_nulo)
        df_out = t.fit_transform(df_base)
        minimo = df_out[["dist_san_francisco", "dist_los_angeles"]].min(axis=1)
        pd.testing.assert_series_equal(df_out["nearest_city_distance"], minimo, check_names=False)

    def test_distancias_nao_negativas(self, df_base, geo_cfg, logger_nulo):
        """Distâncias euclidianas não podem ser negativas."""
        t = GeoDistanceTransformer(geo_config=geo_cfg, logger=logger_nulo)
        df_out = t.fit_transform(df_base)
        assert (df_out["dist_san_francisco"] >= 0).all()

    def test_sem_lat_lon_retorna_df_inalterado(self, geo_cfg, logger_nulo):
        """DataFrame sem colunas lat/lon deve ser retornado sem modificação."""
        df = pd.DataFrame({"col_a": [1, 2]})
        t = GeoDistanceTransformer(geo_config=geo_cfg, logger=logger_nulo)
        df_out = t.fit_transform(df)
        assert "nearest_city_distance" not in df_out.columns


# ─────────────────────────────────────────────────────────────────────────────
# PolynomialFeatureTransformer
# ─────────────────────────────────────────────────────────────────────────────

class TestPolynomialFeatureTransformer:
    """Testes para PolynomialFeatureTransformer."""

    def test_quadrado_uma_coluna(self, df_base, logger_nulo):
        """1 coluna → deve criar o quadrado."""
        cfg = [{"name": "income_sq", "columns": ["median_income"]}]
        t = PolynomialFeatureTransformer(poly_config=cfg, logger=logger_nulo)
        df_out = t.fit_transform(df_base)
        assert "income_sq" in df_out.columns
        esperado = df_base["median_income"] ** 2
        pd.testing.assert_series_equal(df_out["income_sq"], esperado, check_names=False)

    def test_produto_duas_colunas(self, df_base, logger_nulo):
        """2 colunas → deve criar o produto (interação)."""
        cfg = [{"name": "income_x_age", "columns": ["median_income", "housing_median_age"]}]
        t = PolynomialFeatureTransformer(poly_config=cfg, logger=logger_nulo)
        df_out = t.fit_transform(df_base)
        assert "income_x_age" in df_out.columns
        esperado = df_base["median_income"] * df_base["housing_median_age"]
        pd.testing.assert_series_equal(df_out["income_x_age"], esperado, check_names=False)

    def test_coluna_ausente_ignorada(self, df_base, logger_nulo):
        """Coluna ausente não deve lançar exceção."""
        cfg = [{"name": "feature_x", "columns": ["inexistente"]}]
        t = PolynomialFeatureTransformer(poly_config=cfg, logger=logger_nulo)
        df_out = t.fit_transform(df_base)
        assert "feature_x" not in df_out.columns

    def test_tres_colunas_ignorada(self, df_base, logger_nulo):
        """Spec com 3 colunas deve ser ignorada (apenas 1 ou 2 suportadas)."""
        cfg = [{"name": "tripla", "columns": ["median_income", "housing_median_age", "total_rooms"]}]
        t = PolynomialFeatureTransformer(poly_config=cfg, logger=logger_nulo)
        df_out = t.fit_transform(df_base)
        assert "tripla" not in df_out.columns


# ─────────────────────────────────────────────────────────────────────────────
# OceanProximityEncoder
# ─────────────────────────────────────────────────────────────────────────────

class TestOceanProximityEncoder:
    """Testes para OceanProximityEncoder."""

    @pytest.fixture
    def enc_cfg(self) -> dict:
        return {
            "column": "ocean_proximity",
            "ordinal_column": "ocean_proximity_encoded",
            "ordinal_map": {"ISLAND": 0, "NEAR BAY": 1, "NEAR OCEAN": 2, "<1H OCEAN": 3, "INLAND": 4},
            "one_hot_prefix": "op",
            "drop_first": False,
        }

    def test_encoding_ordinal_correto(self, df_base, enc_cfg, logger_nulo):
        """Valores ordinais devem corresponder ao mapa configurado."""
        t = OceanProximityEncoder(enc_config=enc_cfg, logger=logger_nulo)
        df_out = t.fit_transform(df_base)
        assert "ocean_proximity_encoded" in df_out.columns
        assert df_out.loc[0, "ocean_proximity_encoded"] == 1  # NEAR BAY
        assert df_out.loc[2, "ocean_proximity_encoded"] == 4  # INLAND

    def test_dummies_criadas(self, df_base, enc_cfg, logger_nulo):
        """Deve criar colunas one-hot com prefixo 'op_'."""
        t = OceanProximityEncoder(enc_config=enc_cfg, logger=logger_nulo)
        df_out = t.fit_transform(df_base)
        dummies = [c for c in df_out.columns if c.startswith("op_")]
        assert len(dummies) > 0

    def test_coluna_ausente_retorna_df_inalterado(self, enc_cfg, logger_nulo):
        """DataFrame sem a coluna categórica deve ser retornado sem modificação."""
        df = pd.DataFrame({"col_a": [1, 2]})
        t = OceanProximityEncoder(enc_config=enc_cfg, logger=logger_nulo)
        df_out = t.fit_transform(df)
        assert "ocean_proximity_encoded" not in df_out.columns

    def test_nao_modifica_original(self, df_base, enc_cfg, logger_nulo):
        """Não deve modificar o DataFrame original."""
        t = OceanProximityEncoder(enc_config=enc_cfg, logger=logger_nulo)
        t.fit_transform(df_base)
        assert "ocean_proximity_encoded" not in df_base.columns


# ─────────────────────────────────────────────────────────────────────────────
# FeatureSelector
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureSelector:
    """Testes para FeatureSelector."""

    def test_seleciona_colunas_corretas(self, df_base, logger_nulo):
        """Deve retornar apenas as colunas solicitadas."""
        features = ["median_income", "housing_median_age", "latitude"]
        t = FeatureSelector(features_to_keep=features, logger=logger_nulo)
        df_out = t.fit_transform(df_base)
        assert list(df_out.columns) == features

    def test_shape_correto(self, df_base, logger_nulo):
        """Shape de saída deve ter apenas as colunas selecionadas."""
        features = ["median_income", "latitude", "longitude"]
        t = FeatureSelector(features_to_keep=features, logger=logger_nulo)
        df_out = t.fit_transform(df_base)
        assert df_out.shape == (len(df_base), 3)

    def test_coluna_ausente_nao_levanta_excecao(self, df_base, logger_nulo):
        """Coluna ausente deve ser ignorada (warning), nunca levantar exceção."""
        features = ["median_income", "coluna_inexistente"]
        t = FeatureSelector(features_to_keep=features, logger=logger_nulo)
        df_out = t.fit_transform(df_base)
        assert "coluna_inexistente" not in df_out.columns
        assert "median_income" in df_out.columns

    def test_is_capped_target_nao_deve_passar_se_ausente(self, df_base, logger_nulo):
        """
        Verifica princípio de data leakage: is_capped_target NÃO está em
        features_to_keep (o YAML já remove), então se alguém tentar selecioná-la
        a partir de um df sem ela, o seletor ignora silenciosamente.
        """
        features = ["median_income", "is_capped_target"]
        t = FeatureSelector(features_to_keep=features, logger=logger_nulo)
        df_out = t.fit_transform(df_base)  # df_base não tem is_capped_target
        assert "is_capped_target" not in df_out.columns


# ─────────────────────────────────────────────────────────────────────────────
# GroupMedianImputer (stateful)
# ─────────────────────────────────────────────────────────────────────────────

class TestGroupMedianImputer:
    """Testes para GroupMedianImputer — transformador stateful."""

    def test_fit_aprende_medianas(self, df_base, logger_nulo):
        """Após fit, deve ter atributo medians_ com medianas por grupo."""
        t = GroupMedianImputer(group_col="ocean_proximity", target_col="total_bedrooms", logger=logger_nulo)
        t.fit(df_base)
        assert hasattr(t, "medians_")
        assert hasattr(t, "global_median_")
        assert "NEAR BAY" in t.medians_ or "INLAND" in t.medians_

    def test_transform_preenche_nan(self, df_base, logger_nulo):
        """NaN em total_bedrooms deve ser preenchido após fit+transform."""
        t = GroupMedianImputer(group_col="ocean_proximity", target_col="total_bedrooms", logger=logger_nulo)
        df_out = t.fit_transform(df_base)
        assert df_out["total_bedrooms"].isna().sum() == 0

    def test_nao_altera_valores_existentes(self, df_base, logger_nulo):
        """Valores não-NaN não devem ser alterados."""
        t = GroupMedianImputer(group_col="ocean_proximity", target_col="total_bedrooms", logger=logger_nulo)
        df_out = t.fit_transform(df_base)
        # linha 0: total_bedrooms=200 (não era NaN)
        assert df_out.loc[0, "total_bedrooms"] == 200.0

    def test_transform_sem_fit_levanta_runtime_error(self, df_base, logger_nulo):
        """transform() sem fit() deve levantar RuntimeError."""
        t = GroupMedianImputer(group_col="ocean_proximity", target_col="total_bedrooms", logger=logger_nulo)
        with pytest.raises(RuntimeError, match="não foi ajustado"):
            t.transform(df_base)

    def test_fit_coluna_ausente_levanta_key_error(self, df_base, logger_nulo):
        """fit() com coluna inexistente deve levantar KeyError."""
        t = GroupMedianImputer(group_col="ocean_proximity", target_col="inexistente", logger=logger_nulo)
        with pytest.raises(KeyError):
            t.fit(df_base)

    def test_grupo_nao_visto_usa_mediana_global(self, logger_nulo):
        """Grupo não visto no fit deve receber a mediana global no transform."""
        df_treino = pd.DataFrame({
            "grupo": ["A", "A", "B"],
            "valor": [10.0, 20.0, 30.0],
        })
        df_teste = pd.DataFrame({
            "grupo": ["C"],   # grupo não visto no fit
            "valor": [None],
        })
        t = GroupMedianImputer(group_col="grupo", target_col="valor", logger=logger_nulo)
        t.fit(df_treino)
        df_out = t.transform(df_teste)
        assert not pd.isna(df_out.loc[0, "valor"])
        assert df_out.loc[0, "valor"] == t.global_median_


# ─────────────────────────────────────────────────────────────────────────────
# StandardScalerTransformer (stateful)
# ─────────────────────────────────────────────────────────────────────────────

class TestStandardScalerTransformer:
    """Testes para StandardScalerTransformer — transformador stateful."""

    def test_fit_aprende_media_e_std(self, df_base, logger_nulo):
        """Após fit, mean_ e std_ devem estar presentes para as colunas configuradas."""
        cols = ["median_income", "housing_median_age"]
        t = StandardScalerTransformer(columns=cols, logger=logger_nulo)
        t.fit(df_base)
        assert hasattr(t, "mean_")
        assert hasattr(t, "std_")
        for c in cols:
            assert c in t.mean_

    def test_z_score_correto(self, logger_nulo):
        """Valores escalonados devem ter média ≈ 0 e std ≈ 1."""
        df = pd.DataFrame({"feat": [1.0, 2.0, 3.0, 4.0, 5.0]})
        t = StandardScalerTransformer(columns=["feat"], logger=logger_nulo)
        df_out = t.fit_transform(df)
        assert abs(df_out["feat"].mean()) < 1e-10
        assert abs(df_out["feat"].std(ddof=1) - 1.0) < 1e-6

    def test_std_zero_ignorada(self, logger_nulo):
        """Coluna constante (std=0) deve ser ignorada sem exceção."""
        df = pd.DataFrame({"constante": [5.0, 5.0, 5.0], "normal": [1.0, 2.0, 3.0]})
        t = StandardScalerTransformer(columns=["constante", "normal"], logger=logger_nulo)
        df_out = t.fit_transform(df)
        # constante não deve ser escalonada (não está em mean_)
        assert "constante" not in t.mean_
        # coluna constante permanece inalterada
        assert (df_out["constante"] == 5.0).all()

    def test_transform_sem_fit_levanta_runtime_error(self, df_base, logger_nulo):
        """transform() sem fit() deve levantar RuntimeError."""
        t = StandardScalerTransformer(columns=["median_income"], logger=logger_nulo)
        with pytest.raises(RuntimeError, match="não foi ajustado"):
            t.transform(df_base)

    def test_coluna_ausente_no_fit_ignorada(self, df_base, logger_nulo):
        """Coluna ausente do DataFrame no fit deve ser ignorada silenciosamente."""
        cols = ["median_income", "coluna_inexistente"]
        t = StandardScalerTransformer(columns=cols, logger=logger_nulo)
        t.fit(df_base)
        assert "coluna_inexistente" not in t.mean_
        assert "median_income" in t.mean_
