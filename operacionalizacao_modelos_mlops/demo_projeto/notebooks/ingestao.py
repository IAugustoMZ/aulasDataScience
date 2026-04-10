# %%
# Ingestion pipeline — download from Kaggle and convert to Parquet.
# All configuration lives in config/data.yaml and config/pipeline.yaml.
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# %%
from src.core.context import PipelineContext

context = PipelineContext.from_notebook(__file__)
context.run_step("ingestion")
