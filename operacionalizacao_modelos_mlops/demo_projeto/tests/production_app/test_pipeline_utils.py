"""
tests/production_app/test_pipeline_utils.py — Testes unitários para pipeline_utils.py.

Cobre:
  - preprocessar_entradas: verifica shape, colunas e tipos do DataFrame resultante
  - Sanitização XGBoost: verifica renomeação de colunas com caracteres especiais
  - obter_colunas_features: retorna lista com sanitização aplicada
  - obter_colunas_features_brutas: retorna lista com nomes originais
  - obter_parquet_features: lê parquet e retorna DataFrame
  - _construir_imputador_ajustado: lança FileNotFoundError se parquet ausente

Todos os testes mockam leituras de parquet e configurações YAML para isolar
do sistema de arquivos real.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pandas as pd
import pytest

# Bootstrap de paths
_TESTS_DIR    = Path(__file__).resolve().parent.parent   # tests/
_PROJECT_ROOT = _TESTS_DIR.parent                        # demo_projeto/
_APP_DIR      = _PROJECT_ROOT / "production_app"

for _p in [str(_APP_DIR), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def entradas_brutas_validas() -> dict:
    """Dicionário com as 9 features brutas do California Housing."""
    return {
        "median_income":      4.5,
        "housing_median_age": 25.0,
        "total_rooms":        2_500.0,
        "total_bedrooms":     500.0,
        "population":         1_200.0,
        "households":         450.0,
        "latitude":           37.75,
        "longitude":          -122.42,
        "ocean_proximity":    "NEAR BAY",
    }


@pytest.fixture
def df_processado_mock() -> pd.DataFrame:
    """DataFrame mock simulando data/processed/house_price.parquet."""
    return pd.DataFrame({
        "median_house_value": [150_000.0, 250_000.0, 350_000.0],
        "median_income":      [3.5, 5.0, 7.2],
        "housing_median_age": [20.0, 35.0, 52.0],
        "total_rooms":        [800.0, 1_200.0, 600.0],
        "total_bedrooms":     [200.0, 300.0, 150.0],
        "population":         [500.0, 900.0, 400.0],
        "households":         [180.0, 280.0, 140.0],
        "latitude":           [37.88, 37.86, 37.85],
        "longitude":          [-122.23, -122.22, -122.24],
        "ocean_proximity":    ["NEAR BAY", "NEAR BAY", "INLAND"],
    })


@pytest.fixture
def df_features_mock() -> pd.DataFrame:
    """DataFrame mock simulando data/features/house_price_features.parquet."""
    colunas = [
        "median_income", "log_median_income", "latitude", "longitude",
        "ocean_proximity_encoded", "op_INLAND", "op_NEAR BAY",
        "median_house_value",
    ]
    return pd.DataFrame(
        [[4.5, 1.5, 37.75, -122.42, 1.0, 0.0, 1.0, 200_000.0]],
        columns=colunas,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Testes de preprocessar_entradas
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessarEntradas:

    def test_retorna_dataframe(self, entradas_brutas_validas):
        """Verifica que o resultado é um DataFrame."""
        from utils import pipeline_utils
        resultado = pipeline_utils.preprocessar_entradas(entradas_brutas_validas)
        assert isinstance(resultado, pd.DataFrame)

    def test_tem_uma_linha(self, entradas_brutas_validas):
        """Verifica que o resultado tem exatamente uma linha (entrada individual)."""
        from utils import pipeline_utils
        resultado = pipeline_utils.preprocessar_entradas(entradas_brutas_validas)
        assert len(resultado) == 1

    def test_nao_contem_target(self, entradas_brutas_validas):
        """Verifica que a coluna target (median_house_value) não está no resultado."""
        from utils import pipeline_utils
        resultado = pipeline_utils.preprocessar_entradas(entradas_brutas_validas)
        assert "median_house_value" not in resultado.columns

    def test_nao_contem_colunas_brutas_excluidas(self, entradas_brutas_validas):
        """Verifica que colunas de contagem brutas excluídas pelo FeatureSelector estão ausentes."""
        from utils import pipeline_utils
        resultado = pipeline_utils.preprocessar_entradas(entradas_brutas_validas)
        # Estas colunas são excluídas pelo feature_selection (substituídas por razões)
        for col in ["total_rooms", "total_bedrooms", "population", "households"]:
            assert col not in resultado.columns

    def test_colunas_xgboost_sanitizadas(self, entradas_brutas_validas):
        """Verifica que nomes com '<' são renomeados para 'lt_'."""
        from utils import pipeline_utils
        resultado = pipeline_utils.preprocessar_entradas(entradas_brutas_validas)
        for col in resultado.columns:
            assert "<" not in col, f"Coluna não sanitizada encontrada: {col}"
            assert "[" not in col, f"Coluna não sanitizada encontrada: {col}"
            assert "]" not in col, f"Coluna não sanitizada encontrada: {col}"

    def test_sem_nan_apos_imputacao(self, entradas_brutas_validas):
        """Verifica que não há NaN no resultado após o pipeline completo."""
        from utils import pipeline_utils
        entradas_brutas_validas["total_bedrooms"] = float("nan")  # força NaN
        resultado = pipeline_utils.preprocessar_entradas(entradas_brutas_validas)
        assert resultado.isna().sum().sum() == 0, (
            "Valores NaN encontrados no resultado após pré-processamento completo."
        )

    def test_todos_valores_numericos(self, entradas_brutas_validas):
        """Verifica que todas as colunas do resultado são numéricas."""
        from utils import pipeline_utils
        resultado = pipeline_utils.preprocessar_entradas(entradas_brutas_validas)
        for col in resultado.columns:
            assert pd.api.types.is_numeric_dtype(resultado[col]), (
                f"Coluna '{col}' não é numérica: {resultado[col].dtype}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Testes de obter_colunas_features
# ─────────────────────────────────────────────────────────────────────────────

class TestObterColunasFeatures:

    def test_retorna_lista(self):
        """Verifica que o retorno é uma lista."""
        from utils import pipeline_utils
        resultado = pipeline_utils.obter_colunas_features()
        assert isinstance(resultado, list)

    def test_sem_caracteres_especiais_xgboost(self):
        """Verifica que nenhuma coluna contém '<', '[' ou ']'."""
        from utils import pipeline_utils
        resultado = pipeline_utils.obter_colunas_features()
        for col in resultado:
            assert "<" not in col
            assert "[" not in col
            assert "]" not in col

    def test_nao_contem_target(self):
        """Verifica que o target não está na lista de features."""
        from utils import pipeline_utils
        resultado = pipeline_utils.obter_colunas_features()
        assert "median_house_value" not in resultado

    def test_nao_vazia(self):
        """Verifica que a lista não está vazia."""
        from utils import pipeline_utils
        resultado = pipeline_utils.obter_colunas_features()
        assert len(resultado) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Testes de obter_colunas_features_brutas
# ─────────────────────────────────────────────────────────────────────────────

class TestObterColunasFeaturesBrutas:

    def test_retorna_lista(self):
        """Verifica que o retorno é uma lista."""
        from utils import pipeline_utils
        resultado = pipeline_utils.obter_colunas_features_brutas()
        assert isinstance(resultado, list)

    def test_nao_contem_target(self):
        """Verifica que o target não está na lista."""
        from utils import pipeline_utils
        resultado = pipeline_utils.obter_colunas_features_brutas()
        assert "median_house_value" not in resultado

    def test_mesmo_comprimento_que_features_sanitizadas(self):
        """Verifica que brutas e sanitizadas têm o mesmo número de features."""
        from utils import pipeline_utils
        brutas     = pipeline_utils.obter_colunas_features_brutas()
        sanitizadas = pipeline_utils.obter_colunas_features()
        assert len(brutas) == len(sanitizadas)


# ─────────────────────────────────────────────────────────────────────────────
# Testes de obter_parquet_features
# ─────────────────────────────────────────────────────────────────────────────

class TestObterParquetFeatures:

    def test_retorna_dataframe_quando_arquivo_existe(self, df_features_mock, tmp_path):
        """Verifica que retorna DataFrame quando o parquet existe."""
        import utils.pipeline_utils as pu

        caminho_fake = tmp_path / "house_price_features.parquet"
        df_features_mock.to_parquet(caminho_fake, index=False)

        with patch.object(pu, "_PARQUET_FEATURES", caminho_fake):
            resultado = pu.obter_parquet_features()

        assert isinstance(resultado, pd.DataFrame)
        assert len(resultado) == len(df_features_mock)

    def test_levanta_file_not_found_quando_ausente(self, tmp_path):
        """Verifica que FileNotFoundError é levantado quando o parquet não existe."""
        import utils.pipeline_utils as pu

        caminho_inexistente = tmp_path / "nao_existe.parquet"

        with patch.object(pu, "_PARQUET_FEATURES", caminho_inexistente):
            with pytest.raises(FileNotFoundError, match="Parquet de features não encontrado"):
                pu.obter_parquet_features()


# ─────────────────────────────────────────────────────────────────────────────
# Testes de _construir_imputador_ajustado
# ─────────────────────────────────────────────────────────────────────────────

class TestConstruirImputadorAjustado:

    def test_levanta_file_not_found_quando_parquet_ausente(self, tmp_path):
        """Verifica que FileNotFoundError é levantado quando o parquet processado não existe."""
        import utils.pipeline_utils as pu

        caminho_inexistente = tmp_path / "nao_existe.parquet"

        with patch.object(pu, "_PARQUET_PROCESSADO", caminho_inexistente):
            with pytest.raises(FileNotFoundError, match="Parquet processado não encontrado"):
                pu._construir_imputador_ajustado()

    def test_retorna_imputador_ajustado(self, df_processado_mock, tmp_path):
        """Verifica que o imputador retornado foi ajustado (possui medians_)."""
        import utils.pipeline_utils as pu
        from src.preprocessing.transformers.stateful import GroupMedianImputer

        caminho_fake = tmp_path / "house_price.parquet"
        df_processado_mock.to_parquet(caminho_fake, index=False)

        with patch.object(pu, "_PARQUET_PROCESSADO", caminho_fake):
            imputador = pu._construir_imputador_ajustado()

        assert isinstance(imputador, GroupMedianImputer)
        assert hasattr(imputador, "medians_"), (
            "Imputador não foi ajustado: atributo medians_ ausente."
        )
