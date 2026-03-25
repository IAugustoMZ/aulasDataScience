# %%
# Configuração do Ambiente
import sys
from pathlib import Path
# %%
# definições
ROOT_DIR = Path(__file__).resolve().parent.parent
SECRETS_PATH = ROOT_DIR / 'secrets.env'
CONFIG_DIR = ROOT_DIR / 'config'
PIPELINE_CONFIG = CONFIG_DIR / 'pipeline.yaml'
DATA_CONFIG = CONFIG_DIR / 'data.yaml'
PATHS_LIST = [str(ROOT_DIR), str(CONFIG_DIR)]

# adicionar o root e o config no meu path do sistema
for _p in PATHS_LIST:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.utils.logger import get_logger
from src.utils.config_loader import load_yaml
from src.ingestion import ingest_csv_to_parquet
from src.dowloader import check_kaggle_credentials, list_remote_files, download_dataset
# %%
# fazer a leitura dos arquivos de configuração
data_cfg = load_yaml(DATA_CONFIG)
pipeline_cfg = load_yaml(PIPELINE_CONFIG)

# %%
# obtendo a configuração do log
log_cfg = pipeline_cfg.get('logging')

# criar o logger
logger = get_logger(
    name='ingestao',
    logging_config=log_cfg
)

# %%
if check_kaggle_credentials(secrets_path=SECRETS_PATH):
    logger.info('Kaggle Credential set!')
else:
    logger.error('Kaggle Credentials not set')
# %%
# discovery de dados
dataset = data_cfg.get('kaggle').get('dataset')
file_pattern = data_cfg.get('kaggle').get('file_pattern', "*.csv")
expected_files = data_cfg.get('kaggle').get('expected_files')

logger.info('Dataset  : %s', dataset)
logger.info('Padrão   : %s', file_pattern)
logger.info('Arquivos : %s', expected_files or '(auto-descoberta)')

if not expected_files:
    expected_files = list_remote_files(
        dataset=dataset,
        file_pattern=file_pattern,
        logging_config=log_cfg
    )
    logger.info('Arquivos encontrados: %s', expected_files)

# definir o diretório de destino dos dados brutos
raw_dir = ROOT_DIR / pipeline_cfg.get('paths').get('raw_data_dir')
raw_dir.mkdir(parents=True, exist_ok=True)
skip_download = pipeline_cfg.get('execution').get('skip_download_if_exists')
force_download = pipeline_cfg.get('execution').get('force_redownload')

# dowload dos dados - fase Extract de um ETL / ELT
downloaded = download_dataset(
    dataset=dataset,
    expected_files=expected_files,
    destination_dir=raw_dir,
    skip_if_exists=skip_download,
    force=force_download,
    logging_config=log_cfg
)
logger.info('Arquivos prontos: %d', len(downloaded))

# verificar o conteúdo do diretório raw
for f in sorted(raw_dir.glob('*.csv')):
    logger.info('  %s (%.1f KB)', f.name, f.stat().st_size / 1024)

# %%
# definir o caminho de saída do Parquet
processed_dir = ROOT_DIR / pipeline_cfg['paths']['processed_data_dir']
output_path = processed_dir / pipeline_cfg['paths']['output_filename']

logger.info('Saída: %s', output_path)

# obtendo configurações de processamento
compression = data_cfg.get('ingest').get('compression', 'snappy')
chunk_size = data_cfg.get('ingest').get('chunk_size_rows', 50_000)
validate = data_cfg.get('ingest').get('validate_schema')
required_cols = data_cfg.get('schema').get('required_columns')
skip_ingest = pipeline_cfg.get('execution').get('skip_ingest_if_exists')
force_ingest = pipeline_cfg.get('execution').get('force_ingest')

result_path = ingest_csv_to_parquet(
    raw_dir=raw_dir,
    output_path=output_path,
    compression=compression,
    chunk_size_rows=chunk_size,
    validate_schema=validate,
    skip_if_exists=skip_ingest,
    force=force_ingest,
    logging_config=log_cfg
)