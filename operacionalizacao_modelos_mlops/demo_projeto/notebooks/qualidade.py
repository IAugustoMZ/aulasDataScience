# %%
# ─────────────────────────────────────────────────────────────────────────────
# Aula MLOps — Qualidade de Dados com Great Expectations
# ─────────────────────────────────────────────────────────────────────────────
#
# SEGUNDA etapa do pipeline.
# Entrada : data/processed/house_price.parquet  ← gerado por ingestao.py
# Saída   : outputs/quality/quality_report_<timestamp>.json
#
# Conceito central: SEPARAÇÃO entre política e mecanismo
#   • Política  → config/quality.yaml      (O QUÊ validar e com quais thresholds)
#   • Mecanismo → src/quality/             (COMO executar as validações via GE)
#
# Para ajustar qualquer critério de qualidade, edite apenas o YAML.
# Este arquivo não precisa mudar.
# ─────────────────────────────────────────────────────────────────────────────
# %%
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# %%
from src.core.context import PipelineContext

context = PipelineContext.from_notebook(__file__)
context.run_step("quality")
