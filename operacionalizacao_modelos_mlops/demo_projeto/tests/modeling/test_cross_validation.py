"""
test_cross_validation.py — Testes unitários para src/modeling/cross_validation.py.

Cobre:
  • CVRunner.de_config  — construção via YAML
  • CVRunner.executar   — retorna métricas por fold, tamanho correto, sem data leakage
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.modeling.cross_validation import CVRunner


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cv_cfg_3folds() -> dict:
    """Configuração de CV com 3 folds."""
    return {'strategy': 'kfold', 'n_splits': 3, 'shuffle': True}


@pytest.fixture
def dados_sinteticos() -> tuple[pd.DataFrame, pd.Series]:
    """Dataset sintético pequeno sem NaN."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((60, 4)), columns=['a', 'b', 'c', 'd'])
    y = pd.Series(rng.random(60) * 100, name='target')
    return X, y


@pytest.fixture
def pipeline_ridge() -> Pipeline:
    """Pipeline minimalista com Ridge para testes."""
    return Pipeline([('model', Ridge(alpha=1.0))])


# ── Testes de CVRunner.de_config ──────────────────────────────────────────────

class TestCVRunnerDeConfig:
    """Testes para o construtor alternativo CVRunner.de_config."""

    def test_instancia_criada(self, cv_cfg_3folds: dict) -> None:
        """de_config deve retornar instância de CVRunner."""
        runner = CVRunner.de_config(cv_cfg_3folds, seed=0)
        assert isinstance(runner, CVRunner)

    def test_n_splits_respeitado(self, cv_cfg_3folds: dict) -> None:
        """CVRunner deve ter n folds igual ao configurado."""
        runner = CVRunner.de_config(cv_cfg_3folds, seed=0)
        assert runner.cv.n_splits == 3


# ── Testes de CVRunner.executar ───────────────────────────────────────────────

class TestCVRunnerExecutar:
    """Testes para o método CVRunner.executar."""

    def test_numero_correto_de_folds(
        self,
        cv_cfg_3folds: dict,
        pipeline_ridge: Pipeline,
        dados_sinteticos: tuple,
    ) -> None:
        """executar deve retornar lista com len == n_splits."""
        X, y = dados_sinteticos
        runner = CVRunner.de_config(cv_cfg_3folds, seed=0)
        fold_mets = runner.executar(pipeline_ridge, X, y)
        assert len(fold_mets) == 3

    def test_chaves_por_fold(
        self,
        cv_cfg_3folds: dict,
        pipeline_ridge: Pipeline,
        dados_sinteticos: tuple,
    ) -> None:
        """Cada fold deve conter ao menos rmse, mae, r2, mape."""
        X, y = dados_sinteticos
        runner = CVRunner.de_config(cv_cfg_3folds, seed=0)
        fold_mets = runner.executar(pipeline_ridge, X, y)
        chaves_obrigatorias = {'rmse', 'mae', 'r2', 'mape'}
        for fold in fold_mets:
            assert chaves_obrigatorias.issubset(set(fold.keys()))

    def test_rmse_positivo(
        self,
        cv_cfg_3folds: dict,
        pipeline_ridge: Pipeline,
        dados_sinteticos: tuple,
    ) -> None:
        """RMSE deve ser positivo em todos os folds."""
        X, y = dados_sinteticos
        runner = CVRunner.de_config(cv_cfg_3folds, seed=0)
        fold_mets = runner.executar(pipeline_ridge, X, y)
        for fold in fold_mets:
            assert fold['rmse'] > 0

    def test_modelo_original_nao_mutado(
        self,
        cv_cfg_3folds: dict,
        dados_sinteticos: tuple,
    ) -> None:
        """O modelo original não deve ser treinado (clone() em cada fold)."""
        X, y = dados_sinteticos
        modelo = Pipeline([('model', Ridge(alpha=1.0))])
        runner = CVRunner.de_config(cv_cfg_3folds, seed=0)
        runner.executar(modelo, X, y)
        # clone(): o estimador interno não deve ter atributo coef_ (não foi fitado)
        assert not hasattr(modelo.named_steps['model'], 'coef_')
