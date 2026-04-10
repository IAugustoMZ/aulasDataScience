"""
tests/production_app/test_model_utils.py — Testes unitários para model_utils.py.

Cobre:
  - carregar_modelo: verifica que set_tracking_uri e load_model são chamados corretamente
  - prever_individual: retorna float para predição de uma linha
  - prever_lote: retorna lista de floats para predição em lote
  - obter_params_ic: extrai cv_rmse_std, holdout_rmse, run_id e versao_modelo
  - calcular_intervalo_confianca: valida a fórmula IC = y_hat ± z × (std / √n)

Todos os testes usam mocks para isolar completamente do MLflow e do banco SQLite.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Bootstrap de paths para importar production_app.utils
_TESTS_DIR    = Path(__file__).resolve().parent.parent   # tests/
_PROJECT_ROOT = _TESTS_DIR.parent                        # demo_projeto/
_APP_DIR      = _PROJECT_ROOT / "production_app"

for _p in [str(_APP_DIR), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.model_utils import (
    _NOME_MODELO,
    _N_FOLDS_CV,
    _Z_95,
    carregar_modelo,
    prever_individual,
    prever_lote,
    obter_params_ic,
    calcular_intervalo_confianca,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def features_uma_linha() -> pd.DataFrame:
    """DataFrame de uma linha representando features engenheiradas."""
    return pd.DataFrame({
        "median_income":       [4.5],
        "housing_median_age":  [25.0],
        "latitude":            [37.75],
        "longitude":           [-122.42],
    })


@pytest.fixture
def features_multiplas_linhas() -> pd.DataFrame:
    """DataFrame com 5 linhas representando um lote de features."""
    return pd.DataFrame({
        "median_income":      [3.0, 4.5, 6.0, 2.0, 8.0],
        "housing_median_age": [20.0, 25.0, 30.0, 15.0, 40.0],
        "latitude":           [37.0, 37.5, 38.0, 36.5, 38.5],
        "longitude":          [-122.0, -122.4, -121.8, -123.0, -121.5],
    })


@pytest.fixture
def modelo_mock() -> MagicMock:
    """Mock de mlflow.pyfunc.PyFuncModel com predict() configurado."""
    mock = MagicMock()
    mock.predict.return_value = [250_000.0]
    return mock


@pytest.fixture
def modelo_mock_lote() -> MagicMock:
    """Mock de modelo com predict() retornando múltiplas predições."""
    mock = MagicMock()
    mock.predict.return_value = [200_000.0, 250_000.0, 300_000.0, 180_000.0, 350_000.0]
    return mock


@pytest.fixture
def cliente_mock_mlflow() -> MagicMock:
    """Mock do MlflowClient com versões registradas e métricas de run."""
    versao_mock = MagicMock()
    versao_mock.run_id = "abc123def456"
    versao_mock.version = "3"
    versao_mock.current_stage = "None"

    run_mock = MagicMock()
    run_mock.data.metrics = {
        "cv_rmse_std":  45_000.0,
        "holdout_rmse": 48_000.0,
    }

    cliente = MagicMock()
    cliente.search_model_versions.return_value = [versao_mock]
    cliente.get_run.return_value = run_mock
    return cliente


# ─────────────────────────────────────────────────────────────────────────────
# Testes de carregar_modelo
# ─────────────────────────────────────────────────────────────────────────────

class TestCarregarModelo:

    def test_define_tracking_uri_corretamente(self):
        """Verifica que mlflow.set_tracking_uri é chamado com o URI fornecido."""
        uri = "sqlite:///mlruns.db"
        modelo_retornado = MagicMock()

        with patch("utils.model_utils.mlflow.set_tracking_uri") as mock_set, \
             patch("utils.model_utils.mlflow.pyfunc.load_model", return_value=modelo_retornado):
            resultado = carregar_modelo(uri)
            mock_set.assert_called_once_with(uri)

    def test_carrega_modelo_pelo_nome_correto(self):
        """Verifica que load_model é chamado com o registry name configurado."""
        uri = "sqlite:///mlruns.db"
        modelo_retornado = MagicMock()

        with patch("utils.model_utils.mlflow.set_tracking_uri"), \
             patch("utils.model_utils.mlflow.pyfunc.load_model", return_value=modelo_retornado) as mock_load:
            resultado = carregar_modelo(uri)
            mock_load.assert_called_once_with(f"models:/{_NOME_MODELO}/latest")
            assert resultado is modelo_retornado

    def test_propaga_excecao_modelo_nao_encontrado(self):
        """Verifica que exceção do MLflow é propagada quando modelo não existe."""
        import mlflow.exceptions

        with patch("utils.model_utils.mlflow.set_tracking_uri"), \
             patch("utils.model_utils.mlflow.pyfunc.load_model",
                   side_effect=mlflow.exceptions.MlflowException("Modelo não encontrado")):
            with pytest.raises(mlflow.exceptions.MlflowException):
                carregar_modelo("sqlite:///nao_existe.db")


# ─────────────────────────────────────────────────────────────────────────────
# Testes de prever_individual
# ─────────────────────────────────────────────────────────────────────────────

class TestPreverIndividual:

    def test_retorna_float(self, features_uma_linha, modelo_mock):
        """Verifica que a predição retorna um float."""
        resultado = prever_individual(features_uma_linha, modelo_mock)
        assert isinstance(resultado, float)

    def test_valor_correto(self, features_uma_linha, modelo_mock):
        """Verifica que o valor retornado é o primeiro elemento da predição do modelo."""
        resultado = prever_individual(features_uma_linha, modelo_mock)
        assert resultado == 250_000.0

    def test_chama_predict_com_dataframe(self, features_uma_linha, modelo_mock):
        """Verifica que modelo.predict() é chamado com o DataFrame de features."""
        prever_individual(features_uma_linha, modelo_mock)
        modelo_mock.predict.assert_called_once_with(features_uma_linha)

    def test_converte_para_float_quando_numpy(self, features_uma_linha):
        """Verifica conversão correta quando predict retorna array numpy."""
        import numpy as np
        modelo = MagicMock()
        modelo.predict.return_value = np.array([320_000.5])
        resultado = prever_individual(features_uma_linha, modelo)
        assert isinstance(resultado, float)
        assert resultado == pytest.approx(320_000.5)


# ─────────────────────────────────────────────────────────────────────────────
# Testes de prever_lote
# ─────────────────────────────────────────────────────────────────────────────

class TestPreverLote:

    def test_retorna_lista_de_floats(self, features_multiplas_linhas, modelo_mock_lote):
        """Verifica que predição em lote retorna lista de floats."""
        resultado = prever_lote(features_multiplas_linhas, modelo_mock_lote)
        assert isinstance(resultado, list)
        assert all(isinstance(v, float) for v in resultado)

    def test_quantidade_correta_de_predicoes(self, features_multiplas_linhas, modelo_mock_lote):
        """Verifica que o número de predições corresponde ao número de linhas."""
        resultado = prever_lote(features_multiplas_linhas, modelo_mock_lote)
        assert len(resultado) == len(features_multiplas_linhas)

    def test_valores_corretos(self, features_multiplas_linhas, modelo_mock_lote):
        """Verifica que os valores retornados correspondem às predições do modelo."""
        resultado = prever_lote(features_multiplas_linhas, modelo_mock_lote)
        esperado  = [200_000.0, 250_000.0, 300_000.0, 180_000.0, 350_000.0]
        assert resultado == pytest.approx(esperado)

    def test_chama_predict_com_dataframe(self, features_multiplas_linhas, modelo_mock_lote):
        """Verifica que modelo.predict() é chamado com o DataFrame completo."""
        prever_lote(features_multiplas_linhas, modelo_mock_lote)
        modelo_mock_lote.predict.assert_called_once_with(features_multiplas_linhas)


# ─────────────────────────────────────────────────────────────────────────────
# Testes de obter_params_ic
# ─────────────────────────────────────────────────────────────────────────────

class TestObterParamsIc:

    def test_retorna_chaves_esperadas(self, cliente_mock_mlflow):
        """Verifica que o dicionário retornado contém todas as chaves esperadas."""
        uri = "sqlite:///mlruns.db"
        with patch("utils.model_utils.mlflow.MlflowClient", return_value=cliente_mock_mlflow):
            resultado = obter_params_ic(uri)

        assert "cv_rmse_std"   in resultado
        assert "holdout_rmse"  in resultado
        assert "run_id"        in resultado
        assert "versao_modelo" in resultado

    def test_valores_corretos(self, cliente_mock_mlflow):
        """Verifica que os valores de métricas são extraídos corretamente do run."""
        uri = "sqlite:///mlruns.db"
        with patch("utils.model_utils.mlflow.MlflowClient", return_value=cliente_mock_mlflow):
            resultado = obter_params_ic(uri)

        assert resultado["cv_rmse_std"]   == pytest.approx(45_000.0)
        assert resultado["holdout_rmse"]  == pytest.approx(48_000.0)
        assert resultado["run_id"]        == "abc123def456"
        assert resultado["versao_modelo"] == "3"

    def test_levanta_erro_sem_versoes(self):
        """Verifica que ValueError é levantado quando não há versões registradas."""
        uri = "sqlite:///mlruns.db"
        cliente_vazio = MagicMock()
        cliente_vazio.search_model_versions.return_value = []

        with patch("utils.model_utils.mlflow.MlflowClient", return_value=cliente_vazio):
            with pytest.raises(ValueError, match="Nenhuma versão registrada"):
                obter_params_ic(uri)

    def test_seleciona_versao_mais_recente(self):
        """Verifica que a versão com maior número é selecionada."""
        uri = "sqlite:///mlruns.db"

        versao_antiga = MagicMock(run_id="run_antiga", version="1")
        versao_nova   = MagicMock(run_id="run_nova",   version="3")
        versao_media  = MagicMock(run_id="run_media",  version="2")

        run_mock = MagicMock()
        run_mock.data.metrics = {"cv_rmse_std": 0.0, "holdout_rmse": 0.0}

        cliente = MagicMock()
        cliente.search_model_versions.return_value = [versao_antiga, versao_nova, versao_media]
        cliente.get_run.return_value = run_mock

        with patch("utils.model_utils.mlflow.MlflowClient", return_value=cliente):
            resultado = obter_params_ic(uri)

        assert resultado["run_id"] == "run_nova"

    def test_fallback_cv_rmse_std_zero_quando_ausente(self):
        """Verifica que cv_rmse_std retorna 0.0 quando a métrica não está no run."""
        uri = "sqlite:///mlruns.db"

        versao = MagicMock(run_id="run_sem_metrica", version="1", current_stage="None")
        run_mock = MagicMock()
        run_mock.data.metrics = {}  # métricas ausentes

        cliente = MagicMock()
        cliente.search_model_versions.return_value = [versao]
        cliente.get_run.return_value = run_mock

        with patch("utils.model_utils.mlflow.MlflowClient", return_value=cliente):
            resultado = obter_params_ic(uri)

        assert resultado["cv_rmse_std"]  == 0.0
        assert resultado["holdout_rmse"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Testes de calcular_intervalo_confianca
# ─────────────────────────────────────────────────────────────────────────────

class TestCalcularIntervaloConfianca:

    def test_simetria_do_intervalo(self):
        """Verifica que o IC é simétrico em torno da predição."""
        y_hat, cv_rmse_std = 200_000.0, 30_000.0
        inferior, superior = calcular_intervalo_confianca(y_hat, cv_rmse_std)
        assert y_hat - inferior == pytest.approx(superior - y_hat)

    def test_formula_correta(self):
        """Verifica a fórmula: IC = y_hat ± z × (cv_rmse_std / √n_folds)."""
        y_hat, cv_rmse_std, n_folds, z = 200_000.0, 30_000.0, _N_FOLDS_CV, _Z_95
        se      = cv_rmse_std / math.sqrt(n_folds)
        margem  = z * se
        inferior_esperado = y_hat - margem
        superior_esperado = y_hat + margem

        inferior, superior = calcular_intervalo_confianca(y_hat, cv_rmse_std, n_folds, z)
        assert inferior == pytest.approx(inferior_esperado)
        assert superior == pytest.approx(superior_esperado)

    def test_std_zero_resulta_em_predicao_pontual(self):
        """Verifica que cv_rmse_std=0 resulta em intervalo de largura zero."""
        y_hat = 300_000.0
        inferior, superior = calcular_intervalo_confianca(y_hat, cv_rmse_std=0.0)
        assert inferior == pytest.approx(y_hat)
        assert superior == pytest.approx(y_hat)

    def test_n_folds_custom(self):
        """Verifica que n_folds customizado altera corretamente o erro padrão."""
        y_hat, cv_rmse_std = 200_000.0, 30_000.0
        # Mais folds → erro padrão menor → IC mais estreito
        inf_3,  sup_3  = calcular_intervalo_confianca(y_hat, cv_rmse_std, n_folds=3)
        inf_10, sup_10 = calcular_intervalo_confianca(y_hat, cv_rmse_std, n_folds=10)
        assert (sup_3  - inf_3)  > (sup_10 - inf_10)

    def test_retorna_tupla_de_dois_floats(self):
        """Verifica que o retorno é uma tupla com dois floats."""
        resultado = calcular_intervalo_confianca(200_000.0, 30_000.0)
        assert isinstance(resultado, tuple)
        assert len(resultado) == 2
        assert all(isinstance(v, float) for v in resultado)

    def test_inferior_menor_que_superior(self):
        """Verifica que o limite inferior é sempre menor que o superior."""
        inferior, superior = calcular_intervalo_confianca(200_000.0, 30_000.0)
        assert inferior < superior
