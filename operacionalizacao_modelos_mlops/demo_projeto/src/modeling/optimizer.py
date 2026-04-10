"""
modeling/optimizer.py — Estratégias de otimização de hiperparâmetros.

Princípio Open/Closed: para adicionar uma nova estratégia, basta criar uma
subclasse de BaseOptimizer — nenhuma mudança no código existente é necessária.

Estratégias disponíveis:
  optuna            → Busca bayesiana (TPE) com runs aninhados no MLflow
  grid_search       → Busca exaustiva em grade (GridSearchCV do sklearn)
  randomized_search → Busca aleatória amostrada (RandomizedSearchCV do sklearn)

Seleção via config/modeling.yaml:
    optimizer:
      strategy: "optuna"   # ou "grid_search" ou "randomized_search"

Fábrica:
    otimizador = OptimizerFactory.criar(cfg, cv_runner, pipe_cfg, mlflow_tracker, logger)
    resultado  = otimizador.otimizar(model_name, model_cfg, X, y, pipe_cfg, feat_red_cfg)
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import mlflow
import optuna
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.modeling.base import BaseOptimizer
from src.modeling.metrics import agregar_metricas_folds
from src.modeling.model_factory import construir_pipeline, construir_modelo
from src.modeling.cross_validation import CVRunner


# ─────────────────────────────────────────────────────────────────────────────
# Helpers compartilhados
# ─────────────────────────────────────────────────────────────────────────────

def _sugerir_parametro(trial: optuna.Trial, nome: str, spec: dict) -> Any:
    """
    Constrói uma sugestão Optuna a partir de um spec do search_space do YAML.

    Tipos suportados:
        log_float   → suggest_float(..., log=True)
        float       → suggest_float(...)
        int         → suggest_int(...)
        categorical → suggest_categorical(...)
    """
    tipo = spec['type']
    if tipo == 'log_float':
        return trial.suggest_float(nome, float(spec['low']), float(spec['high']), log=True)
    elif tipo == 'float':
        return trial.suggest_float(nome, float(spec['low']), float(spec['high']))
    elif tipo == 'int':
        return trial.suggest_int(nome, int(spec['low']), int(spec['high']))
    elif tipo == 'categorical':
        return trial.suggest_categorical(nome, spec['choices'])
    else:
        raise ValueError(f'Tipo de search_space desconhecido: {tipo!r}')


def _params_reducer_padrao(feat_red_cfg: dict) -> dict:
    """
    Constrói o dict de parâmetros para FeatureReducer a partir do config YAML.

    Parâmetros
    ----------
    feat_red_cfg : dict da seção feature_reduction em modeling.yaml

    Retorna
    -------
    dict com 'method' + parâmetros específicos do método ativo
    """
    metodo = feat_red_cfg.get('method', 'none')
    cfg_metodo = feat_red_cfg.get(metodo, {})
    params: dict = {'method': metodo}

    if metodo == 'rfe':
        params['n_features_to_select'] = cfg_metodo.get('n_features_to_select', 15)
        params['rfe_estimator']        = cfg_metodo.get('rfe_estimator', 'ridge')
    elif metodo == 'pca':
        params['n_components'] = cfg_metodo.get('n_components', 15)
    elif metodo == 'kpca':
        params['n_components'] = cfg_metodo.get('n_components', 15)
        params['kernel']       = cfg_metodo.get('kernel', 'rbf')
        params['gamma']        = cfg_metodo.get('gamma', None)
        params['degree']       = cfg_metodo.get('degree', 3)
        params['coef0']        = cfg_metodo.get('coef0', 1.0)

    return params


def _separar_params_reducer(
    best_params_all: dict,
    feat_red_cfg: dict,
) -> tuple[dict, dict]:
    """
    Separa params do estimador dos params do reducer a partir dos best_params do Optuna.

    Retorna
    -------
    (params_estimador, params_reducer)
    """
    params_estimador = {
        k: v for k, v in best_params_all.items()
        if not k.startswith('reducer_')
    }
    metodo_trial = best_params_all.get('reducer_method', feat_red_cfg.get('method', 'none'))
    params_reducer: dict = {'method': metodo_trial}

    for k, v in best_params_all.items():
        if k.startswith('reducer_') and k != 'reducer_method':
            params_reducer[k[len('reducer_'):]] = v

    # Preenche parâmetros fixos do método que não foram otimizados
    defaults_metodo = {
        k: v for k, v in feat_red_cfg.get(metodo_trial, {}).items()
        if k != 'search_space'
    }
    for k, v in defaults_metodo.items():
        if k not in params_reducer:
            params_reducer[k] = v

    return params_estimador, params_reducer


# ─────────────────────────────────────────────────────────────────────────────
# Implementações concretas
# ─────────────────────────────────────────────────────────────────────────────

class OptunaOptimizer(BaseOptimizer):
    """
    Otimizador bayesiano usando Optuna (sampler TPE por padrão).

    Cada trial é registrado como run aninhado no MLflow para rastreabilidade.
    Co-otimiza: método de redução + parâmetros do método + hiperparâmetros do estimador.

    Parâmetros
    ----------
    cfg_optuna   : dict da seção optimizer.optuna em modeling.yaml
    cv_runner    : CVRunner configurado
    pipe_cfg     : dict da seção pipeline em modeling.yaml
    seed         : semente aleatória global
    n_trials_global: número de trials padrão (sobrescrito por optuna_trials no model_cfg)
    logger       : Logger opcional
    """

    def __init__(
        self,
        cfg_optuna: dict,
        cv_runner: CVRunner,
        pipe_cfg: dict,
        seed: int,
        n_trials_global: int = 10,
        logger: logging.Logger | None = None,
    ) -> None:
        self.cfg_optuna = cfg_optuna
        self.cv_runner = cv_runner
        self.pipe_cfg = pipe_cfg
        self.seed = seed
        self.n_trials_global = n_trials_global
        self.logger = logger

    def otimizar(
        self,
        model_name: str,
        model_cfg: dict,
        X_tune: Any,
        y_tune: Any,
        pipe_cfg: dict,
        feat_red_cfg: dict,
    ) -> dict:
        """
        Executa otimização Optuna e retorna os melhores parâmetros.

        Retorna
        -------
        dict com 'estimator_params', 'reducer_params' e 'study' (objeto Optuna)
        """
        n_trials = model_cfg.get('optuna_trials', self.n_trials_global)
        search_space = model_cfg.get('search_space') or {}
        _red_global_ss = feat_red_cfg.get('search_space', {})
        _metodo_padrao = feat_red_cfg.get('method', 'none')

        def _objetivo(trial: optuna.Trial) -> float:
            # Sugere parâmetros do estimador
            params = {
                nome: _sugerir_parametro(trial, nome, spec)
                for nome, spec in search_space.items()
            }

            # Sugere método de redução
            if 'method' in _red_global_ss:
                metodo_trial = _sugerir_parametro(trial, 'reducer_method', _red_global_ss['method'])
            else:
                metodo_trial = _metodo_padrao

            # Sugere parâmetros do método de redução escolhido
            params_reducer_trial: dict = {'method': metodo_trial}
            ss_metodo = feat_red_cfg.get(metodo_trial, {}).get('search_space', {})
            for r_nome, r_spec in ss_metodo.items():
                params_reducer_trial[r_nome] = _sugerir_parametro(
                    trial, f'reducer_{r_nome}', r_spec
                )
            # Preenche defaults do método para params fora do search_space
            defaults_metodo = {
                k: v for k, v in feat_red_cfg.get(metodo_trial, {}).items()
                if k != 'search_space'
            }
            for k, v in defaults_metodo.items():
                if k not in params_reducer_trial:
                    params_reducer_trial[k] = v

            pipeline_trial = construir_pipeline(
                model_cfg=model_cfg,
                params_modelo=params,
                params_reducer=params_reducer_trial,
                pipe_cfg=pipe_cfg,
            )

            fold_mets = self.cv_runner.executar(pipeline_trial, X_tune, y_tune)
            agg = agregar_metricas_folds(fold_mets)

            # Registra trial como run aninhado no MLflow
            todos_params = {k: (str(v) if v is None else v) for k, v in params.items()}
            todos_params.update({
                f'reducer_{k}': (str(v) if v is None else v)
                for k, v in params_reducer_trial.items()
            })
            with mlflow.start_run(
                run_name=f'trial_{trial.number}',
                nested=True,
                tags={'stage': 'trial', 'model': model_name, 'trial': str(trial.number)},
            ):
                mlflow.log_params(todos_params)
                mlflow.log_metrics({
                    'cv_rmse_mean': agg['cv_rmse_mean'],
                    'cv_mae_mean' : agg['cv_mae_mean'],
                    'cv_r2_mean'  : agg['cv_r2_mean'],
                    'cv_mape_mean': agg['cv_mape_mean'],
                })

            return agg['cv_rmse_mean']

        study = optuna.create_study(
            direction='minimize',
            study_name=model_name,
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )
        study.optimize(_objetivo, n_trials=n_trials, show_progress_bar=False, catch=(Exception,))

        # Valida se algum trial convergiu
        try:
            _ = study.best_value
        except ValueError:
            if self.logger:
                self.logger.warning('[SKIP OPTUNA] %s — nenhum trial bem-sucedido', model_name)
            return {}

        params_estimador, params_reducer = _separar_params_reducer(
            study.best_params, feat_red_cfg
        )
        return {
            'estimator_params': params_estimador,
            'reducer_params'  : params_reducer,
            'study'           : study,
        }


class GridSearchOptimizer(BaseOptimizer):
    """
    Otimizador por busca exaustiva em grade (GridSearchCV do sklearn).

    O search_space do YAML é interpretado como grade de valores:
      - categorical → usa 'choices' diretamente como lista
      - int/float   → usa [low, high] como dois extremos (mínimo e máximo)
      - log_float   → usa [low, high] em escala logarítmica (3 pontos)

    Nota: GridSearch não co-otimiza o reducer junto do estimador.
    Usa o método de redução padrão do config.

    Parâmetros
    ----------
    cfg_grid  : dict da seção optimizer.grid_search em modeling.yaml
    cv_runner : CVRunner configurado (usa o KFold internamente via n_splits)
    seed      : semente aleatória global
    logger    : Logger opcional
    """

    def __init__(
        self,
        cfg_grid: dict,
        cv_runner: CVRunner,
        seed: int,
        logger: logging.Logger | None = None,
    ) -> None:
        self.cfg_grid = cfg_grid
        self.cv_runner = cv_runner
        self.seed = seed
        self.logger = logger

    def _converter_search_space_para_grade(self, search_space: dict) -> dict:
        """Converte o search_space do YAML para o formato de grade do GridSearchCV."""
        grade = {}
        for nome, spec in search_space.items():
            tipo = spec['type']
            if tipo == 'categorical':
                grade[nome] = spec['choices']
            elif tipo in ('int', 'float'):
                grade[nome] = [spec['low'], spec['high']]
            elif tipo == 'log_float':
                # Três pontos em escala log para grade
                import math
                low_log  = math.log10(float(spec['low']))
                high_log = math.log10(float(spec['high']))
                mid_log  = (low_log + high_log) / 2
                grade[nome] = [10**low_log, 10**mid_log, 10**high_log]
            else:
                grade[nome] = [spec.get('low', spec.get('choices', [None])[0])]
        return grade

    def otimizar(
        self,
        model_name: str,
        model_cfg: dict,
        X_tune: Any,
        y_tune: Any,
        pipe_cfg: dict,
        feat_red_cfg: dict,
    ) -> dict:
        """
        Executa GridSearchCV e retorna os melhores parâmetros.

        Retorna
        -------
        dict com 'estimator_params' e 'reducer_params'
        """
        search_space = model_cfg.get('search_space') or {}
        if not search_space:
            if self.logger:
                self.logger.info('[SKIP GRID] %s — sem search_space', model_name)
            return {}

        grade = self._converter_search_space_para_grade(search_space)
        estimador = construir_modelo(model_cfg)

        # Prefixo 'estimator__' para GridSearchCV dentro de Pipeline não é necessário
        # aqui pois otimizamos apenas o estimador, sem pipeline completo
        grid_cv = GridSearchCV(
            estimator=estimador,
            param_grid=grade,
            cv=self.cv_runner.cv,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            refit=True,
        )
        grid_cv.fit(X_tune, y_tune)

        if self.logger:
            self.logger.info(
                '  [GRID] %s — melhor RMSE: %.4f',
                model_name, -grid_cv.best_score_,
            )

        return {
            'estimator_params': grid_cv.best_params_,
            'reducer_params'  : _params_reducer_padrao(feat_red_cfg),
        }


class RandomizedSearchOptimizer(BaseOptimizer):
    """
    Otimizador por busca aleatória amostrada (RandomizedSearchCV do sklearn).

    O search_space do YAML é convertido para distribuições scipy:
      - categorical → lista de valores (escolha uniforme)
      - int         → randint do scipy
      - float/log_float → uniform do scipy

    Não co-otimiza o reducer junto do estimador (usa o método padrão do config).

    Parâmetros
    ----------
    cfg_random : dict da seção optimizer.randomized_search em modeling.yaml
    cv_runner  : CVRunner configurado
    seed       : semente aleatória global
    logger     : Logger opcional
    """

    def __init__(
        self,
        cfg_random: dict,
        cv_runner: CVRunner,
        seed: int,
        logger: logging.Logger | None = None,
    ) -> None:
        self.cfg_random = cfg_random
        self.cv_runner = cv_runner
        self.seed = seed
        self.logger = logger

    def _converter_search_space_para_distribuicoes(self, search_space: dict) -> dict:
        """Converte o search_space do YAML para distribuições scipy."""
        from scipy.stats import randint, uniform, loguniform
        dist = {}
        for nome, spec in search_space.items():
            tipo = spec['type']
            if tipo == 'categorical':
                dist[nome] = spec['choices']
            elif tipo == 'int':
                dist[nome] = randint(int(spec['low']), int(spec['high']) + 1)
            elif tipo == 'float':
                low  = float(spec['low'])
                high = float(spec['high'])
                dist[nome] = uniform(low, high - low)
            elif tipo == 'log_float':
                dist[nome] = loguniform(float(spec['low']), float(spec['high']))
            else:
                dist[nome] = [spec.get('low', 0)]
        return dist

    def otimizar(
        self,
        model_name: str,
        model_cfg: dict,
        X_tune: Any,
        y_tune: Any,
        pipe_cfg: dict,
        feat_red_cfg: dict,
    ) -> dict:
        """
        Executa RandomizedSearchCV e retorna os melhores parâmetros.

        Retorna
        -------
        dict com 'estimator_params' e 'reducer_params'
        """
        search_space = model_cfg.get('search_space') or {}
        if not search_space:
            if self.logger:
                self.logger.info('[SKIP RANDOM] %s — sem search_space', model_name)
            return {}

        n_iter = self.cfg_random.get('n_iter', 50)
        dist   = self._converter_search_space_para_distribuicoes(search_space)
        estimador = construir_modelo(model_cfg)

        random_cv = RandomizedSearchCV(
            estimator=estimador,
            param_distributions=dist,
            n_iter=n_iter,
            cv=self.cv_runner.cv,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            random_state=self.seed,
            refit=True,
        )
        random_cv.fit(X_tune, y_tune)

        if self.logger:
            self.logger.info(
                '  [RANDOM] %s — melhor RMSE: %.4f',
                model_name, -random_cv.best_score_,
            )

        return {
            'estimator_params': random_cv.best_params_,
            'reducer_params'  : _params_reducer_padrao(feat_red_cfg),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Fábrica de Otimizadores
# ─────────────────────────────────────────────────────────────────────────────

class OptimizerFactory:
    """
    Fábrica que cria o otimizador correto a partir da configuração YAML.

    Seleção via modeling.yaml → optimizer.strategy:
      "optuna"            → OptunaOptimizer
      "grid_search"       → GridSearchOptimizer
      "randomized_search" → RandomizedSearchOptimizer
    """

    @staticmethod
    def criar(
        cfg_modelagem: dict,
        cv_runner: CVRunner,
        pipe_cfg: dict,
        seed: int,
        logger: logging.Logger | None = None,
    ) -> BaseOptimizer:
        """
        Instancia o otimizador conforme a estratégia configurada.

        Parâmetros
        ----------
        cfg_modelagem : dict da seção raiz de modeling.yaml
        cv_runner     : CVRunner configurado
        pipe_cfg      : dict da seção pipeline em modeling.yaml
        seed          : semente aleatória global
        logger        : Logger opcional

        Retorna
        -------
        Instância de BaseOptimizer

        Raises
        ------
        ValueError se a estratégia não for reconhecida
        """
        cfg_opt  = cfg_modelagem.get('optimizer', {})
        strategy = cfg_opt.get('strategy', 'optuna')

        if strategy == 'optuna':
            cfg_optuna       = cfg_opt.get('optuna', {})
            n_trials_global  = cfg_optuna.get('default_trials', 10)
            return OptunaOptimizer(
                cfg_optuna=cfg_optuna,
                cv_runner=cv_runner,
                pipe_cfg=pipe_cfg,
                seed=seed,
                n_trials_global=n_trials_global,
                logger=logger,
            )

        elif strategy == 'grid_search':
            cfg_grid = cfg_opt.get('grid_search', {})
            return GridSearchOptimizer(
                cfg_grid=cfg_grid,
                cv_runner=cv_runner,
                seed=seed,
                logger=logger,
            )

        elif strategy == 'randomized_search':
            cfg_random = cfg_opt.get('randomized_search', {})
            return RandomizedSearchOptimizer(
                cfg_random=cfg_random,
                cv_runner=cv_runner,
                seed=seed,
                logger=logger,
            )

        else:
            raise ValueError(
                f"Estratégia de otimização desconhecida: '{strategy}'. "
                "Valores válidos: 'optuna', 'grid_search', 'randomized_search'."
            )
