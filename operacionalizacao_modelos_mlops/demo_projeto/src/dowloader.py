import os
from pathlib import Path
from dotenv import load_dotenv

def check_kaggle_credentials(secrets_path: Path) -> None:
    """
    Verify Kaggle credentials are available either via environment variable.
    """
    load_dotenv(dotenv_path=str(secrets_path))
    has_env = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    return has_env

