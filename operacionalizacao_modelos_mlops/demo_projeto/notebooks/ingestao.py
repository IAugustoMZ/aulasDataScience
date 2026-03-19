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
from src.dowloader import check_kaggle_credentials
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
# checando variável de ambiente
if check_kaggle_credentials(secrets_path=SECRETS_PATH):
    logger.info('Kaggle Credential set!')
else:
    logger.error('Kaggle Credentials not set')
# %%
