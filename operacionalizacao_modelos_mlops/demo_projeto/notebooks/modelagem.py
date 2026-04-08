# %%
# ─────────────────────────────────────────────────────────────────────────────
# Aula MLOps — Experimentação e Modelagem com MLFlow + Optuna
# ─────────────────────────────────────────────────────────────────────────────
#
# Este script é a QUARTA etapa do pipeline de dados.
# Entrada : data/features/house_price_features.parquet  ← preprocessamento.py
# Saída   : mlruns/          (servidor MLFlow — experimentos, runs, artefatos)
#           outputs/modeling/ (plots PNG salvos localmente antes de logar)
#
# Conceito central: EXPERIMENTAÇÃO RASTREÁVEL
#   • Política  → config/modeling.yaml  (modelos, search spaces, CV, artefatos)
#   • Mecanismo → este script           (laços de treino, Optuna, MLFlow logging)
#
# O que é rastreado no MLFlow:
#   1. Baseline CV    → cada modelo com parâmetros padrão; métricas por fold
#   2. Optuna Trials  → cada combinação de hiperparâmetros (runs aninhados)
#   3. Ensembles      → Stacking e Voting construídos sobre o top-3 individual
#   4. Melhor Modelo  → artefatos completos (6 plots + análise de resíduo)
#   5. Holdout        → métricas finais em dados nunca vistos durante a busca
#
# Modelos testados:
#   Lineares    : LinearRegression, Ridge, Lasso
#   Árvore      : DecisionTreeRegressor
#   Vizinhança  : KNeighborsRegressor
#   Kernel      : SVR  (com subsampling nos trials — O(n²))
#   Ensemble    : RandomForestRegressor, GradientBoostingRegressor
#   Boosting    : XGBRegressor, LGBMRegressor
#   Avançados   : StackingRegressor (top-3), VotingRegressor (top-3)
# ─────────────────────────────────────────────────────────────────────────────

# %%
# Configuração do Ambiente
import sys
import json
import time
import yaml
import warnings
import importlib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
matplotlib.use('Agg')           # backend não-interativo: salva em arquivo sem abrir janela
from pathlib import Path
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

# Importações — rastreamento e otimização
import optuna
import mlflow
import mlflow.sklearn

# Importações — scikit-learn
from sklearn.base import clone
from sklearn.model_selection import KFold, train_test_split, learning_curve
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor, VotingRegressor

# Definições de caminhos — mesmo padrão dos outros walkthroughs
ROOT_DIR   = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / 'config'
PATHS_LIST = [str(ROOT_DIR), str(CONFIG_DIR)]

for _p in PATHS_LIST:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Importações do projeto
from src.utils.logger import get_logger
from src.utils.config_loader import load_yaml
from src.preprocessing import GroupMedianImputer, StandardScalerTransformer
from src.feature_reducer import FeatureReducer
from sklearn.pipeline import Pipeline as SklearnPipeline

# Suprime warnings verbosos de libs externas (XGBoost, LightGBM, sklearn deprecations)
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Funções Auxiliares
#
# Definidas aqui (topo do script) para que todas as seções as encontrem.
# Seguem o princípio de responsabilidade única: cada função faz uma coisa.
# ─────────────────────────────────────────────────────────────────────────────

# %%
def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula as quatro métricas de regressão usadas neste pipeline."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100)
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

# %%
def _run_cv(model, X: pd.DataFrame, y: pd.Series, cv: KFold) -> list[dict]:
    """
    Executa Cross-Validation e retorna métricas por fold.

    Clona o modelo em cada fold para evitar contaminação de estado entre folds.
    Retorna lista de dicts: [{fold, rmse, mae, r2, mape}, ...].
    """
    fold_metrics = []
    for fold_i, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        m = clone(model)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = m.predict(X.iloc[val_idx])
        metrics = _compute_metrics(y.iloc[val_idx].values, y_pred)
        metrics['fold'] = fold_i + 1
        fold_metrics.append(metrics)
    return fold_metrics

# %%
def _aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    """Agrega métricas de todos os folds em média ± desvio padrão."""
    df = pd.DataFrame(fold_metrics)
    result = {}
    for col in ['rmse', 'mae', 'r2', 'mape']:
        result[f'cv_{col}_mean'] = float(df[col].mean())
        result[f'cv_{col}_std']  = float(df[col].std())
    return result

# %%
def _suggest_param(trial: optuna.Trial, name: str, spec: dict):
    """
    Constrói uma sugestão Optuna a partir de um spec do search_space do YAML.

    Tipos suportados:
        log_float  → suggest_float(..., log=True)
        float      → suggest_float(...)
        int        → suggest_int(...)
        categorical→ suggest_categorical(...)
    """
    ptype = spec['type']
    # Garante tipos numéricos — PyYAML pode parsear notação científica (1.0e-4)
    # como string em algumas versões; float()/int() normalizam sem custo.
    if ptype == 'log_float':
        return trial.suggest_float(name, float(spec['low']), float(spec['high']), log=True)
    elif ptype == 'float':
        return trial.suggest_float(name, float(spec['low']), float(spec['high']))
    elif ptype == 'int':
        return trial.suggest_int(name, int(spec['low']), int(spec['high']))
    elif ptype == 'categorical':
        return trial.suggest_categorical(name, spec['choices'])
    else:
        raise ValueError(f'Tipo de search_space desconhecido: {ptype!r}')

# %%
def _build_model(model_cfg: dict, extra_params: dict | None = None):
    """
    Instancia um modelo usando importlib a partir do config (module + class).

    Mescla default_params com extra_params (extra_params sobrescreve o default).
    Permite instanciar qualquer modelo sklearn-compatível sem hardcode.
    """
    module    = importlib.import_module(model_cfg['module'])
    cls       = getattr(module, model_cfg['class'])
    params    = dict(model_cfg.get('default_params') or {})
    if extra_params:
        params.update(extra_params)
    return cls(**params)

# %%
def _build_pipeline(
    model_cfg: dict,
    model_params: dict | None,
    reducer_params: dict | None,
    pipe_cfg: dict,
) -> SklearnPipeline:
    """
    Constrói um sklearn Pipeline leak-free para um modelo:

        GroupMedianImputer(s)       ← um por entrada em pipe_cfg['imputation']
        StandardScalerTransformer   ← colunas de pipe_cfg['scaling']['columns']
        FeatureReducer              ← method e params de reducer_params
        estimator                   ← instanciado via _build_model

    Por que usar Pipeline?
    - fit() em cada fold de CV chama fit() em TODOS os steps, usando apenas
      os índices de treino daquele fold.  Nenhum dado de validação/holdout
      vaza para o imputador ou scaler.
    - clone() (usado em _run_cv) preserva os hiperparâmetros sem o estado
      aprendido, garantindo isolação entre folds.

    Parâmetros
    ----------
    model_cfg      : dict do models section em modeling.yaml
    model_params   : parâmetros extras do Optuna (sobrescrevem default_params)
    reducer_params : parâmetros para FeatureReducer (method + kwargs do método ativo)
    pipe_cfg       : dict de pipeline section em modeling.yaml
    """
    steps = []

    # ── Imputação (stateful — aprende medianas só no treino) ──────────────────
    for imp_spec in pipe_cfg.get('imputation', []):
        step_name = f"imputer_{imp_spec['column'].replace('/', '_')}"
        steps.append((
            step_name,
            GroupMedianImputer(
                group_col=imp_spec['group_by'],
                target_col=imp_spec['column'],
            ),
        ))

    # ── Escalonamento (stateful — aprende μ/σ só no treino) ──────────────────
    scale_cols = pipe_cfg.get('scaling', {}).get('columns', [])
    if scale_cols:
        steps.append(('scaler', StandardScalerTransformer(columns=scale_cols)))

    # ── Redução de features (opcional, tunable pelo Optuna) ──────────────────
    reducer_kw = reducer_params or {}
    steps.append(('reducer', FeatureReducer(**reducer_kw)))

    # ── Estimador final ───────────────────────────────────────────────────────
    steps.append(('estimator', _build_model(model_cfg, model_params)))

    return SklearnPipeline(steps)

# %%
def _get_feature_importance(
    model, feature_names: list[str],
    X_val: pd.DataFrame, y_val: pd.Series,
) -> pd.Series:
    """
    Extrai importância de features do modelo treinado.

    Suporta sklearn Pipeline: extrai o estimador final via named_steps['estimator']
    e usa os nomes de features pós-redução de named_steps['reducer'] quando
    disponível.

    Prioridade:
        1. feature_importances_ (árvores, ensembles baseados em árvores)
        2. coef_ (modelos lineares — usa valor absoluto)
        3. permutation_importance (fallback model-agnóstico: SVR, KNN, ensembles mistos)
    """
    # Desempacota Pipeline para obter estimador + nomes de features pós-redução
    if isinstance(model, SklearnPipeline):
        estimator = model.named_steps['estimator']
        reducer   = model.named_steps.get('reducer')
        if reducer is not None and reducer.selected_features is not None:
            # RFE mantém nomes originais; PCA/kPCA usa 'pc_0', 'pc_1', ...
            imp_feature_names = reducer.selected_features
        else:
            imp_feature_names = feature_names
    else:
        estimator = model
        imp_feature_names = feature_names

    if hasattr(estimator, 'feature_importances_'):
        return pd.Series(estimator.feature_importances_, index=imp_feature_names)
    elif hasattr(estimator, 'coef_'):
        coef = np.abs(estimator.coef_)
        if coef.ndim > 1:
            coef = coef.flatten()
        return pd.Series(coef, index=imp_feature_names)
    else:
        # Permutation importance sobre o PIPELINE completo (inclui transformações)
        # — usa X_val original para que o pipeline transforme consistentemente
        sample_size = min(2000, len(X_val))
        idx = np.random.default_rng(42).choice(len(X_val), sample_size, replace=False)
        r = permutation_importance(
            model, X_val.iloc[idx], y_val.iloc[idx],
            n_repeats=5, random_state=42, n_jobs=-1,
        )
        return pd.Series(r.importances_mean, index=feature_names)
    
# %%
# ─────────────────────────────────────────────────────────────────────────────
# Carregamento das Configurações
#
# Mesma estratégia dos outros walkthroughs:
#   1. load_config() carrega pipeline.yaml + data.yaml
#   2. modeling.yaml é carregado com yaml.safe_load (responsabilidade única)
#   3. Os dois dicts são mesclados com config.update()
# ─────────────────────────────────────────────────────────────────────────────

# %%
# 1. Configuração geral do pipeline
config = load_yaml(CONFIG_DIR / 'pipeline.yaml')
modeling_cfg = load_yaml(CONFIG_DIR / 'modeling.yaml')
config.update(modeling_cfg)  # mescla modeling.yaml no config geral do pipeline

# Cria o logger
log_cfg = config.get('logging')
logger  = get_logger('modelagem', log_cfg)

logger.info('=== Experimentação e Modelagem — MLFlow + Optuna ===')
logger.info('Config carregada: pipeline.yaml + modeling.yaml')

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Configuração do MLFlow
#
# tracking_uri local gera uma pasta mlruns/ no diretório de trabalho.
# Para inspecionar os resultados, execute no terminal:
#   mlflow ui --backend-store-uri mlruns
# e acesse http://localhost:5000
# ─────────────────────────────────────────────────────────────────────────────

# %%
modeling_cfg    = config.get('modeling', {})
tracking_uri    = modeling_cfg.get('tracking_uri', 'mlruns')
experiment_name = modeling_cfg.get('experiment_name', 'california-housing-experiments')
SEED            = modeling_cfg.get('random_seed', 42)
pipe_cfg        = config.get('pipeline', {})
feat_red_cfg    = config.get('feature_reduction', {})
optuna_cfg      = config.get('optuna', {})
_global_n_trials = optuna_cfg.get('default_trials', 50)

# Path.as_uri() converte o caminho absoluto para file:///E:/... no Windows,
# evitando que o MLFlow interprete a letra do drive (E:) como URI scheme.
mlflow.set_tracking_uri((ROOT_DIR / tracking_uri).as_uri())
mlflow.set_experiment(experiment_name)

logger.info('MLFlow tracking URI  : %s', (ROOT_DIR / tracking_uri).as_uri())
logger.info('MLFlow experiment    : %s', experiment_name)
logger.info('Random seed          : %d', SEED)

# %%
# Inspeciona o que foi carregado do YAML
cv_cfg      = config.get('cv', {})
holdout_cfg = config.get('holdout', {})
models_cfg  = config.get('models', {})
ensembles_cfg = config.get('ensembles', {})
artifacts_cfg = config.get('artifacts', {})

logger.info('CV              : %s (%d folds)', cv_cfg.get('strategy'), cv_cfg.get('n_splits'))
logger.info('Holdout         : %.0f%%', holdout_cfg.get('test_size', 0.2) * 100)
logger.info('Modelos         : %d configurados', len(models_cfg))
enabled_models = [k for k, v in models_cfg.items() if v.get('enabled', True)]
logger.info('  Habilitados   : %s', enabled_models)
logger.info('Imputadores     : %d step(s)', len(pipe_cfg.get('imputation', [])))
logger.info('Scaling cols    : %d', len(pipe_cfg.get('scaling', {}).get('columns', [])))
logger.info('Feature reducer : method=%s', feat_red_cfg.get('method', 'none'))


# %%
# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 1 — Carregar Features
#
# O Parquet gerado por preprocessamento_walkthrough.py contém:
#   • Features selecionadas pelo EDA (sem totais brutos)
#   • Features de engenharia (razões, log1p, distâncias, polinomiais)
#   • Encoding do ocean_proximity
#   • Escala ORIGINAL (sem z-score) — escalonamento é feito no Pipeline
#
# ✓ MLOps — Data Leakage Corrigido:
#   GroupMedianImputer e StandardScalerTransformer agora estão dentro do
#   sklearn Pipeline (_build_pipeline). O fit() de cada step é chamado
#   APENAS nos índices de treino de cada fold de CV, garantindo que nenhuma
#   estatística do conjunto de validação/holdout contamine o treinamento.
# ─────────────────────────────────────────────────────────────────────────────

# %%
features_dir  = ROOT_DIR / config.get('paths', {}).get('features_data_dir', 'data/features')
features_file = features_dir / config.get('paths', {}).get('features_filename', 'house_price_features.parquet')

logger.info('─' * 60)
logger.info('SEÇÃO 1: Carregar Features')
logger.info('Lendo: %s', features_file)

if not features_file.exists():
    raise FileNotFoundError(
        f"Arquivo de features não encontrado: {features_file}\n"
        "Execute preprocessamento.py antes deste script."
    )

# %%
# Inspeciona schema sem carregar dados (leitura de metadados é barata)
schema = pq.read_schema(str(features_file))
logger.info('Schema (%d colunas):', len(schema))
for field in schema:
    logger.info('  %-35s %s', field.name, field.type)

# %%
# Carrega o DataFrame completo
df = pq.read_table(str(features_file)).to_pandas()
logger.info('Shape: %s', df.shape)

# %%
# Separa features (X) e target (y)
# target vem do config de seleção de features do preprocessing.yaml
sel_cfg    = config.get('feature_selection', {})
target_col = sel_cfg.get('target', 'median_house_value')

feature_cols = [c for c in df.columns if c != target_col]
X = df[feature_cols].copy()
y = df[target_col]

# XGBoost rejeita nomes de colunas com '[', ']' ou '<' (ex: op_<1H OCEAN).
# Sanitizamos aqui, uma vez, para que todos os modelos downstream usem nomes limpos.
_rename_map = {
    c: c.replace('<', 'lt_').replace('[', '(').replace(']', ')')
    for c in X.columns
    if any(ch in c for ch in ('<', '[', ']'))
}
if _rename_map:
    X = X.rename(columns=_rename_map)
    logger.info('Colunas renomeadas para compatibilidade com XGBoost: %s', _rename_map)

logger.info('Features : %d colunas', len(feature_cols))
logger.info('Target   : %s  (min=%.0f, max=%.0f, média=%.0f)',
            target_col, y.min(), y.max(), y.mean())

# %%
# Distribuição do target — referência para interpretar os erros do modelo
logger.info(y.describe())

# %%
# Checagem de nulos — não deve haver nenhum após o preprocessing
n_nulls = X.isna().sum().sum()
if n_nulls > 0:
    logger.warning('ATENÇÃO: %d valores nulos encontrados nas features!', n_nulls)
else:
    logger.info('Sem valores nulos nas features ✓')

# %%
# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 2 — Divisão Treino / Holdout
#
# O holdout é separado ANTES de qualquer treino, tuning ou seleção de modelo.
# É o "cofre selado" — nunca será visto até a avaliação final (Seção 9).
#
# Estratificação por quantis do target:
#   • Garante que treino e holdout têm distribuições similares do target
#   • Evita que toda a cauda superior (imóveis caros) caia em um único set
# ─────────────────────────────────────────────────────────────────────────────

# %%
n_bins    = holdout_cfg.get('stratify_bins', 10)
test_size = holdout_cfg.get('test_size', 0.2)

# Cria bins de quantis do target para estratificação
y_bins = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')

logger.info('─' * 60)
logger.info('SEÇÃO 2: Divisão Treino / Holdout')
logger.info('Test size: %.0f%%  |  Bins de estratificação: %d', test_size * 100, n_bins)

# %%
X_train, X_holdout, y_train, y_holdout = train_test_split(
    X, y,
    test_size=test_size,
    random_state=SEED,
    stratify=y_bins,
)

logger.info('Treino  : %d amostras (%.1f%%)', len(X_train), 100 * len(X_train) / len(X))
logger.info('Holdout : %d amostras (%.1f%%)', len(X_holdout), 100 * len(X_holdout) / len(X))
logger.info('Target no treino  — média: %.0f | std: %.0f', y_train.mean(), y_train.std())
logger.info('Target no holdout — média: %.0f | std: %.0f', y_holdout.mean(), y_holdout.std())

# %%
# Verifica que o holdout reflete a distribuição original (sem grandes desvios)
logger.info('Distribuição por quantil (treino vs holdout):')
train_dist   = pd.qcut(y_train,   q=5, labels=False, duplicates='drop').value_counts(normalize=True).sort_index()
holdout_dist = pd.qcut(y_holdout, q=5, labels=False, duplicates='drop').value_counts(normalize=True).sort_index()
dist_df = pd.DataFrame({'treino': train_dist, 'holdout': holdout_dist})
logger.info('\n%s', dist_df.round(3).to_string())

# %%
# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 3 — Registro de Modelos e Configuração do Pipeline
#
# Os modelos são instanciados dinamicamente via importlib:
#   module: "sklearn.linear_model"  +  class: "Ridge"  → from sklearn.linear_model import Ridge
#
# Cada modelo é envolvido num sklearn Pipeline (_build_pipeline) que inclui:
#   1. GroupMedianImputer(s)      — imputa bedrooms_per_room por grupo
#   2. StandardScalerTransformer  — z-score nas colunas contínuas
#   3. FeatureReducer             — none | rfe | pca | kpca (config: feature_reduction)
#   4. estimador                  — o modelo em si
# ─────────────────────────────────────────────────────────────────────────────

# %%
logger.info('─' * 60)
logger.info('SEÇÃO 3: Registro de Modelos e Configuração do Pipeline')

rows = []
for name, cfg in models_cfg.items():
    enabled = cfg.get('enabled', True)
    rows.append({
        'modelo'       : name,
        'habilitado'   : '✓' if enabled else '✗',
        'classe'       : f"{cfg['module']}.{cfg['class']}",
        'optuna_trials': cfg.get('optuna_trials', _global_n_trials),
        'params_default': len(cfg.get('default_params') or {}),
        'search_space' : len(cfg.get('search_space') or {}),
    })

models_table = pd.DataFrame(rows)
logger.info('\n%s', models_table.to_string(index=False))

# %%
# Lê configuração da redução de features e prepara os parâmetros padrão
_red_method     = feat_red_cfg.get('method', 'none')
_red_method_cfg = feat_red_cfg.get(_red_method, {})

def _default_reducer_params() -> dict:
    """
    Constrói o dict de parâmetros para FeatureReducer a partir do método ativo
    e seus valores padrão em modeling.yaml (feature_reduction.<method>).
    """
    params = {'method': _red_method}
    if _red_method == 'rfe':
        params['n_features_to_select'] = _red_method_cfg.get('n_features_to_select', 15)
        params['rfe_estimator']        = _red_method_cfg.get('rfe_estimator', 'ridge')
    elif _red_method == 'pca':
        params['n_components'] = _red_method_cfg.get('n_components', 15)
    elif _red_method == 'kpca':
        params['n_components'] = _red_method_cfg.get('n_components', 15)
        params['kernel']       = _red_method_cfg.get('kernel', 'rbf')
        params['gamma']        = _red_method_cfg.get('gamma', None)
        params['degree']       = _red_method_cfg.get('degree', 3)
        params['coef0']        = _red_method_cfg.get('coef0', 1.0)
    return params

logger.info('Feature reducer: method=%s  params=%s', _red_method, _default_reducer_params())

# %%
# ─────────────────────────────────────────────────────────────────────────────
# SEÇÃO 4 — Baseline: Cross-Validation com Parâmetros Padrão
#
# Para cada modelo habilitado:
#   1. Instancia com os parâmetros padrão do YAML
#   2. Executa KFold CV sobre os dados de treino
#   3. Registra no MLFlow:
#       • params: parâmetros padrão usados
#       • metrics: RMSE/MAE/R²/MAPE por fold (step = índice do fold)
#       • metrics: médias e desvios padrão agregados
#
# Por que fazer baseline ANTES do Optuna?
#   • Serve como referência: o Optuna deve sempre superar o baseline
#   • Identifica modelos claramente inadequados (ex: Linear para dados não-lineares)
#   • Permite comparar o ganho real da otimização de hiperparâmetros
# ─────────────────────────────────────────────────────────────────────────────

# %%
# Cria o objeto de CV a partir do config
n_splits = cv_cfg.get('n_splits', 5)
shuffle  = cv_cfg.get('shuffle', True)

cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=SEED)
logger.info('─' * 60)
logger.info('SEÇÃO 4: Baseline CV  (%s, %d folds)', cv_cfg.get('strategy', 'kfold'), n_splits)

# %%
# Dicionário central de resultados:
# {model_name: {cv_rmse_mean, cv_rmse_std, ..., fold_metrics, model_cfg, best_params, tuned}}
all_results: dict[str, dict] = {}

# %%
# Loop de baseline — um MLFlow run por modelo
for model_name, model_cfg in models_cfg.items():
    if not model_cfg.get('enabled', True):
        logger.info('[SKIP] %s (desabilitado)', model_name)
        continue

    logger.info('  [BASELINE] %-25s ...', model_name)
    pipeline = _build_pipeline(
        model_cfg   = model_cfg,
        model_params = None,
        reducer_params = _default_reducer_params(),
        pipe_cfg    = pipe_cfg,
    )
    t0 = time.time()

    with mlflow.start_run(
        run_name=f'baseline_{model_name}',
        tags={'stage': 'baseline', 'model': model_name},
    ):
        # Registra parâmetros padrão (modelo + reducer)
        default_params = {
            str(k): (str(v) if v is None else v)
            for k, v in (model_cfg.get('default_params') or {}).items()
        }
        default_params['reducer_method'] = _red_method
        mlflow.log_params(default_params)
        mlflow.set_tag('model_class', f"{model_cfg['module']}.{model_cfg['class']}")
        mlflow.set_tag('reducer_method', _red_method)

        # Executa CV (clone() do pipeline garante isolação entre folds)
        fold_metrics = _run_cv(pipeline, X_train, y_train, cv)
        for fm in fold_metrics:
            step = fm['fold']
            mlflow.log_metric('fold_rmse', fm['rmse'], step=step)
            mlflow.log_metric('fold_mae',  fm['mae'],  step=step)
            mlflow.log_metric('fold_r2',   fm['r2'],   step=step)
            mlflow.log_metric('fold_mape', fm['mape'], step=step)

        # Agrega e loga métricas consolidadas
        agg = _aggregate_fold_metrics(fold_metrics)
        mlflow.log_metrics(agg)
        mlflow.log_metric('training_time_s', time.time() - t0)

    all_results[model_name] = {
        **agg,
        'fold_metrics'  : fold_metrics,
        'model_cfg'     : model_cfg,
        'best_params'   : dict(model_cfg.get('default_params') or {}),
        'reducer_params': _default_reducer_params(),
        'tuned'         : False,
    }
    logger.info(
        '    CV RMSE: %8.2f ± %6.2f  |  R²: %.4f  |  %.1fs',
        agg['cv_rmse_mean'], agg['cv_rmse_std'], agg['cv_r2_mean'], time.time() - t0,
    )

# %%
# Tabela comparativa do baseline
baseline_df = pd.DataFrame([
    {
        'modelo'        : k,
        'cv_rmse_mean'  : v['cv_rmse_mean'],
        'cv_rmse_std'   : v['cv_rmse_std'],
        'cv_mae_mean'   : v['cv_mae_mean'],
        'cv_r2_mean'    : v['cv_r2_mean'],
    }
    for k, v in all_results.items()
]).sort_values('cv_rmse_mean')

logger.info('\n%s', baseline_df.round(2).to_string(index=False))
baseline_df