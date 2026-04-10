"""
modeling/model_factory.py — Fábrica de modelos e pipelines sklearn.

Responsabilidade única: instanciar modelos e pipelines a partir de configuração YAML.
Toda a lógica de importação dinâmica (importlib) e construção de Pipeline fica aqui.

Princípio Open/Closed: para adicionar um modelo basta editar modeling.yaml —
nenhuma mudança no código é necessária.
"""
from __future__ import annotations

import importlib
from typing import Any

from sklearn.pipeline import Pipeline as SklearnPipeline

from src.preprocessing.transformers.stateful import GroupMedianImputer, StandardScalerTransformer
from src.modeling.reducer import FeatureReducer


def construir_modelo(model_cfg: dict, params_extras: dict | None = None) -> Any:
    """
    Instancia um modelo via importlib a partir da configuração (module + class).

    Mescla default_params com params_extras (params_extras sobrescreve os defaults).
    Permite instanciar qualquer modelo sklearn-compatível sem hardcode.

    Parâmetros
    ----------
    model_cfg    : dict com chaves 'module', 'class' e opcionalmente 'default_params'
    params_extras: parâmetros adicionais (ex: gerados pelo Optuna) que sobrescrevem defaults

    Retorna
    -------
    Instância do modelo sklearn-compatível
    """
    modulo = importlib.import_module(model_cfg['module'])
    cls    = getattr(modulo, model_cfg['class'])
    params = dict(model_cfg.get('default_params') or {})
    if params_extras:
        params.update(params_extras)
    return cls(**params)


def construir_pipeline(
    model_cfg: dict,
    params_modelo: dict | None,
    params_reducer: dict | None,
    pipe_cfg: dict,
) -> SklearnPipeline:
    """
    Constrói um sklearn Pipeline leak-free para um modelo.

    Estrutura do pipeline:
        GroupMedianImputer(s)       ← um por entrada em pipe_cfg['imputation']
        StandardScalerTransformer   ← colunas de pipe_cfg['scaling']['columns']
        FeatureReducer              ← method e params de params_reducer
        estimador                   ← instanciado via construir_modelo

    Por que usar Pipeline?
    - fit() em cada fold de CV chama fit() em TODOS os steps, usando apenas
      os índices de treino daquele fold. Nenhum dado de validação/holdout
      vaza para o imputador ou scaler.
    - clone() (usado no CVRunner) preserva os hiperparâmetros sem o estado
      aprendido, garantindo isolação entre folds.

    Parâmetros
    ----------
    model_cfg     : dict da seção models em modeling.yaml
    params_modelo : parâmetros extras do otimizador (sobrescrevem default_params)
    params_reducer: parâmetros para FeatureReducer (method + kwargs do método ativo)
    pipe_cfg      : dict da seção pipeline em modeling.yaml

    Retorna
    -------
    SklearnPipeline não-ajustado, pronto para fit()
    """
    steps = []

    # ── Imputação (stateful — aprende medianas apenas no treino) ──────────────
    for spec_imp in pipe_cfg.get('imputation', []):
        nome_step = f"imputer_{spec_imp['column'].replace('/', '_')}"
        steps.append((
            nome_step,
            GroupMedianImputer(
                group_col=spec_imp['group_by'],
                target_col=spec_imp['column'],
            ),
        ))

    # ── Escalonamento (stateful — aprende μ/σ apenas no treino) ──────────────
    colunas_escala = pipe_cfg.get('scaling', {}).get('columns', [])
    if colunas_escala:
        steps.append(('scaler', StandardScalerTransformer(columns=colunas_escala)))

    # ── Redução de features (opcional, tunável pelo otimizador) ──────────────
    kw_reducer = params_reducer or {}
    steps.append(('reducer', FeatureReducer(**kw_reducer)))

    # ── Estimador final ───────────────────────────────────────────────────────
    steps.append(('estimator', construir_modelo(model_cfg, params_modelo)))

    return SklearnPipeline(steps)
