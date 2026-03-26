# %%
# ─────────────────────────────────────────────────────────────────────────────
# Aula MLOps — Qualidade de Dados com Great Expectations
# ─────────────────────────────────────────────────────────────────────────────
#
# Este script é a SEGUNDA etapa do pipeline de dados.
# Entrada : data/processed/house_price.parquet  ← gerado por ingestao_walkthrough.py
# Saída   : outputs/quality/quality_report.json
#
# Conceito central: SEPARAÇÃO entre política e mecanismo
#   • Política  → config/quality.yaml  (O QUÊ validar e com quais thresholds)
#   • Mecanismo → src/quality.py       (COMO executar as validações via GE)
#
# Para ajustar um critério de qualidade, edite apenas o YAML.
# O código não precisa mudar.
# ─────────────────────────────────────────────────────────────────────────────
import sys
import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq
# %%
# definições
ROOT_DIR = Path(__file__).resolve().parent.parent
SECRETS_PATH = ROOT_DIR / 'secrets.env'
CONFIG_DIR = ROOT_DIR / 'config'
PIPELINE_CONFIG = CONFIG_DIR / 'pipeline.yaml'
DATA_CONFIG = CONFIG_DIR / 'data.yaml'
QUALITY_CONFIG = CONFIG_DIR / 'quality.yaml'
PATHS_LIST = [str(ROOT_DIR), str(CONFIG_DIR)]

# adicionar o root e o config no meu path do sistema
for _p in PATHS_LIST:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.utils.logger import get_logger
from src.utils.config_loader import load_yaml
from src.quality_checks import run_quality_checks, save_quality_report

# Carregar arquivos de configuração
pipeline_config = load_yaml(PIPELINE_CONFIG)
data_config = load_yaml(DATA_CONFIG)
quality_config = load_yaml(QUALITY_CONFIG)

# Unir em um único dicionário
config = {
    **pipeline_config,
    **data_config,
    **quality_config
}

# Criar logger
log_cfg = config.get('logging')
logger = get_logger(
    name='qualidade',
    logging_config=log_cfg
)

#%%
# obter os caminhos dos dados de entrada
processed_dir = ROOT_DIR / config.get('paths').get('processed_data_dir')
parquet_path = processed_dir / config.get('paths').get('output_filename')

logger.info('Caminho do Parquet: %s', parquet_path)

if not parquet_path.exists():
    logger.error('Arquivo Parquet não encontrado: %s', parquet_path)
    raise FileNotFoundError(f'Arquivo Parquet não encontrado: {parquet_path}')

# inspeção do schema
schema = pq.read_schema(parquet_path)
logger.info('Schema (%d colunas):', len(schema))
for i, field in enumerate(schema):
    logger.info('  %d: %s (%s)', i + 1, field.name, field.type)

# carregar os dados em um DataFrame do Pandas
df = pd.read_parquet(parquet_path)
# pq.read_table(parquet_path).to_pandas()
logger.info('Dados carregados: %d linhas, %d colunas', df.shape[0], df.shape[1])

#%%
# execução da validação de qualidade - usando o YAML como SST
summary = run_quality_checks(
    df=df,
    config=config,
    logging_config=log_cfg
)
logger.info(summary)

# %%
# persistir o relatório de qualidade
output_dir = ROOT_DIR / config.get('output_dir', 'outputs/quality')

# salva o relatório em JSON
report_path = save_quality_report(
    summary=summary,
    output_dir=output_dir,
    logging_config=log_cfg
)

fail_on_err = config.get('fail_pipeline_on_error', True)

if summary['success']:
    logger.info('=== Qualidade APROVADA — dados prontos para o próximo passo ===')
else:
    logger.warning(
        '=== Qualidade com PENDÊNCIAS: %d/%d checks falharam ===',
        summary['failed'],
        summary['total'],
    )
    logger.warning(
        'fail_pipeline_on_error=%s — %s',
        fail_on_err,
        'RuntimeError será lançado' if fail_on_err else 'continuando mesmo assim (modo exploração)',
    )

# %%
# Resumo 
logger.info('─' * 60)
logger.info('Entrada : %s', parquet_path)
logger.info('Relatório: %s', report_path)
logger.info('Checks   : %d total | %d passaram | %d falharam',
            summary['total'], summary['passed'], summary['failed'])
logger.info('Status   : %s', 'APROVADO' if summary['success'] else 'REPROVADO')