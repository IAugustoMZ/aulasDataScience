"""
test_optimizer.py — Testes unitários para src/modeling/optimizer.py.

Cobre:
  • OptimizerFactory.criar — retorna o otimizador correto para cada estratégia
  • Cada otimizador.otimizar — executa 1 trial sem erros e retorna chaves corretas
  
Todos os testes usam n_trials=1 e dataset sintético pequeno para velocidade.
"""
import numpy as np
import pandas as pd
import pytest
import logging

from src.modeling.cross_validation import CVRunner
from src.modeling.optimizer import (
    OptimizerFactory,
    OptunaOptimizer,
    GridSearchOptimizer,
    RandomizedSearchOptimizer,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def null_logger() -> logging.Logger:
    """Logger que descarta todas as mensagens."""
    logger = logging.getLogger('test_optimizer')
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


@pytest.fixture
def dados_sinteticos() -> tuple[pd.DataFrame, pd.Series]:
    """Dataset sintético pequeno (40 amostras, 4 features) sem NaN."""
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.random((40, 4)), columns=['a', 'b', 'c', 'd'])
    y = pd.Series(rng.random(40) * 100, name='target')
    return X, y


@pytest.fixture
def cv_runner_2folds(null_logger: logging.Logger) -> CVRunner:
    """CVRunner com 2 folds — mínimo possível para velocidade."""
    return CVRunner.de_config({'strategy': 'kfold', 'n_splits': 2, 'shuffle': True}, seed=0)


@pytest.fixture
def pipe_cfg_minimo() -> dict:
    """Config de pipeline sem imputação nem escalonamento."""
    return {'imputation': [], 'scaling': {'columns': []}}


@pytest.fixture
def feat_red_cfg_none() -> dict:
    """Redução de features desabilitada."""
    return {'method': 'none'}


@pytest.fixture
def cfg_ridge_search_space() -> dict:
    """Configuração Ridge com search_space simples para testes."""
    return {
        'module': 'sklearn.linear_model',
        'class': 'Ridge',
        'default_params': {'alpha': 1.0},
        'search_space': {
            'alpha': {'type': 'log_float', 'low': 0.01, 'high': 10.0},
        },
        'optuna_trials': 1,
    }


# ── Testes de OptimizerFactory ────────────────────────────────────────────────

class TestOptimizerFactory:
    """Testes para OptimizerFactory.criar."""

    def _fazer_cfg(self, estrategia: str, cv_runner: CVRunner, pipe_cfg: dict) -> dict:
        return {
            'optimizer': {'strategy': estrategia, 'optuna': {'default_trials': 1, 'sampler': 'tpe'}},
            'feature_reduction': {'method': 'none'},
        }

    def test_cria_optuna_optimizer(
        self, cv_runner_2folds: CVRunner, pipe_cfg_minimo: dict, null_logger: logging.Logger
    ) -> None:
        cfg = self._fazer_cfg('optuna', cv_runner_2folds, pipe_cfg_minimo)
        opt = OptimizerFactory.criar(cfg, cv_runner_2folds, pipe_cfg_minimo, 0, null_logger)
        assert isinstance(opt, OptunaOptimizer)

    def test_cria_grid_search_optimizer(
        self, cv_runner_2folds: CVRunner, pipe_cfg_minimo: dict, null_logger: logging.Logger
    ) -> None:
        cfg = self._fazer_cfg('grid_search', cv_runner_2folds, pipe_cfg_minimo)
        opt = OptimizerFactory.criar(cfg, cv_runner_2folds, pipe_cfg_minimo, 0, null_logger)
        assert isinstance(opt, GridSearchOptimizer)

    def test_cria_randomized_search_optimizer(
        self, cv_runner_2folds: CVRunner, pipe_cfg_minimo: dict, null_logger: logging.Logger
    ) -> None:
        cfg = self._fazer_cfg('randomized_search', cv_runner_2folds, pipe_cfg_minimo)
        cfg['optimizer']['randomized_search'] = {'n_iter': 1}
        opt = OptimizerFactory.criar(cfg, cv_runner_2folds, pipe_cfg_minimo, 0, null_logger)
        assert isinstance(opt, RandomizedSearchOptimizer)

    def test_estrategia_invalida_levanta_erro(
        self, cv_runner_2folds: CVRunner, pipe_cfg_minimo: dict, null_logger: logging.Logger
    ) -> None:
        """Estratégia desconhecida deve levantar ValueError."""
        cfg = {'optimizer': {'strategy': 'algoritmo_magico'}, 'feature_reduction': {'method': 'none'}}
        with pytest.raises(ValueError):
            OptimizerFactory.criar(cfg, cv_runner_2folds, pipe_cfg_minimo, 0, null_logger)


# ── Testes de execução (1 trial cada) ────────────────────────────────────────

class TestOptunaOtimizerExecucao:
    """Testa OptunaOptimizer.otimizar com 1 trial."""

    def test_retorna_chaves_esperadas(
        self,
        cv_runner_2folds: CVRunner,
        pipe_cfg_minimo: dict,
        feat_red_cfg_none: dict,
        cfg_ridge_search_space: dict,
        dados_sinteticos: tuple,
        null_logger: logging.Logger,
    ) -> None:
        X, y = dados_sinteticos
        opt = OptunaOptimizer(
            cfg_optuna      ={'default_trials': 1, 'sampler': 'tpe'},
            cv_runner       =cv_runner_2folds,
            pipe_cfg        =pipe_cfg_minimo,
            seed            =0,
            n_trials_global =1,
            logger          =null_logger,
        )
        res = opt.otimizar('ridge', cfg_ridge_search_space, X, y, pipe_cfg_minimo, feat_red_cfg_none)
        assert 'estimator_params' in res
        assert 'reducer_params'   in res
        assert 'study'            in res


class TestGridSearchOtimizerExecucao:
    """Testa GridSearchOptimizer.otimizar com grid mínimo."""

    def test_retorna_chaves_esperadas(
        self,
        cv_runner_2folds: CVRunner,
        pipe_cfg_minimo: dict,
        feat_red_cfg_none: dict,
        dados_sinteticos: tuple,
        null_logger: logging.Logger,
    ) -> None:
        X, y = dados_sinteticos
        cfg = {
            'module': 'sklearn.linear_model',
            'class': 'Ridge',
            'default_params': {'alpha': 1.0},
            'search_space': {
                'alpha': {'type': 'categorical', 'choices': [0.1, 1.0]},
            },
        }
        opt = GridSearchOptimizer(
            cfg_grid =  {},
            cv_runner=cv_runner_2folds,
            seed     =0,
            logger   =null_logger,
        )
        res = opt.otimizar('ridge', cfg, X, y, pipe_cfg_minimo, feat_red_cfg_none)
        assert 'estimator_params' in res
        assert 'reducer_params'   in res


class TestRandomizedSearchOtimizerExecucao:
    """Testa RandomizedSearchOptimizer.otimizar com n_iter=1."""

    def test_retorna_chaves_esperadas(
        self,
        cv_runner_2folds: CVRunner,
        pipe_cfg_minimo: dict,
        feat_red_cfg_none: dict,
        cfg_ridge_search_space: dict,
        dados_sinteticos: tuple,
        null_logger: logging.Logger,
    ) -> None:
        X, y = dados_sinteticos
        opt = RandomizedSearchOptimizer(
            cfg_random={'n_iter': 1},
            cv_runner =cv_runner_2folds,
            seed      =0,
            logger    =null_logger,
        )
        res = opt.otimizar('ridge', cfg_ridge_search_space, X, y, pipe_cfg_minimo, feat_red_cfg_none)
        assert 'estimator_params' in res
        assert 'reducer_params'   in res
