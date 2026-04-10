"""
test_model_factory.py — Testes unitários para src/modeling/model_factory.py.

Cobre:
  • construir_modelo   — instanciação dinâmica via importlib
  • construir_pipeline — monta Pipeline sklearn com os steps corretos
"""
import pytest
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

from src.modeling.model_factory import construir_modelo, construir_pipeline


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg_ridge() -> dict:
    """Configuração mínima de um modelo Ridge."""
    return {
        'module': 'sklearn.linear_model',
        'class': 'Ridge',
        'default_params': {'alpha': 2.0},
    }


@pytest.fixture
def pipe_cfg_minimo() -> dict:
    """Configuração mínima de pipeline (sem imputação nem escalonamento)."""
    return {
        'imputation': [],
        'scaling': {'columns': []},
    }


@pytest.fixture
def feat_red_cfg_none() -> dict:
    """Configuração de redução de features com método 'none'."""
    return {'method': 'none'}


# ── Testes de construir_modelo ────────────────────────────────────────────────

class TestConstruirModelo:
    """Testes para construir_modelo()."""

    def test_cria_ridge_sem_params_extras(self, cfg_ridge: dict) -> None:
        """construir_modelo deve retornar instância de Ridge."""
        modelo = construir_modelo(cfg_ridge, params_extras=None)
        assert isinstance(modelo, Ridge)

    def test_aplica_params_extras(self, cfg_ridge: dict) -> None:
        """Parâmetros extras devem sobrescrever os default_params."""
        modelo = construir_modelo(cfg_ridge, params_extras={'alpha': 99.0})
        assert modelo.alpha == pytest.approx(99.0)

    def test_params_default_sem_extras(self, cfg_ridge: dict) -> None:
        """default_params do YAML devem ser aplicados quando extras é None."""
        modelo = construir_modelo(cfg_ridge, params_extras=None)
        assert modelo.alpha == pytest.approx(2.0)

    def test_levanta_erro_classe_inexistente(self) -> None:
        """Deve levantar ImportError para módulo/classe inválido."""
        cfg_invalido = {
            'module': 'modulo.inexistente',
            'class': 'ClasseFantasma',
            'default_params': {},
        }
        with pytest.raises((ImportError, ModuleNotFoundError, AttributeError)):
            construir_modelo(cfg_invalido, params_extras=None)


# ── Testes de construir_pipeline ─────────────────────────────────────────────

class TestConstruirPipeline:
    """Testes para construir_pipeline()."""

    def test_retorna_sklearn_pipeline(
        self, cfg_ridge: dict, pipe_cfg_minimo: dict, feat_red_cfg_none: dict
    ) -> None:
        """Deve retornar instância de sklearn.pipeline.Pipeline."""
        pipe = construir_pipeline(
            model_cfg      =cfg_ridge,
            params_modelo  =None,
            params_reducer ={'method': 'none'},
            pipe_cfg       =pipe_cfg_minimo,
        )
        assert isinstance(pipe, Pipeline)

    def test_ultimo_step_e_estimador(
        self, cfg_ridge: dict, pipe_cfg_minimo: dict
    ) -> None:
        """O último step do Pipeline deve ser o estimador Ridge."""
        pipe = construir_pipeline(
            model_cfg      =cfg_ridge,
            params_modelo  =None,
            params_reducer ={'method': 'none'},
            pipe_cfg       =pipe_cfg_minimo,
        )
        nome_ultimo, estimador_ultimo = pipe.steps[-1]
        assert isinstance(estimador_ultimo, Ridge)

    def test_pipeline_fitavel_com_dados_sinteticos(
        self, cfg_ridge: dict, pipe_cfg_minimo: dict
    ) -> None:
        """Pipeline deve ser fitável com dados sintéticos simples (sem NaN)."""
        rng = np.random.default_rng(0)
        X = rng.random((50, 5))
        y = rng.random(50)

        import pandas as pd
        colnames = [f'f{i}' for i in range(5)]
        Xdf = pd.DataFrame(X, columns=colnames)

        pipe_cfg = {
            'imputation': [],
            'scaling': {'columns': []},
        }
        pipe = construir_pipeline(
            model_cfg      =cfg_ridge,
            params_modelo  =None,
            params_reducer ={'method': 'none'},
            pipe_cfg       =pipe_cfg,
        )
        pipe.fit(Xdf, y)
        preds = pipe.predict(Xdf)
        assert preds.shape == (50,)
