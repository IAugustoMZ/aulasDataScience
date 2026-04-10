# %%
# ─────────────────────────────────────────────────────────────────────────────
# Aula MLOps — Etapa de Modelagem
# ─────────────────────────────────────────────────────────────────────────────
#
# Este script é a QUARTA etapa do pipeline de dados.
#
# Entrada : data/features/house_price_features.parquet  ← preprocessamento.py
# Saída   : mlruns.db  (tracking SQLite)
#           outputs/modeling/ (plots PNG e experiment_summary.json)
#
# TODA a política de experimentação (quais modelos, search_spaces, nº de
# trials, artefatos, CV) é definida em config/modeling.yaml.
# Este arquivo é intencionalmente agnóstico ao domínio.
# ─────────────────────────────────────────────────────────────────────────────

# %%
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# %%
# Inicializa contexto do pipeline (resolve root_dir, carrega configs, retorna logger)
from src.core.context import PipelineContext

context = PipelineContext.from_notebook(__file__)

# %%
# Executa a etapa completa de modelagem
from src.modeling.step import ModelingStep

step = ModelingStep(context)
step.run()
