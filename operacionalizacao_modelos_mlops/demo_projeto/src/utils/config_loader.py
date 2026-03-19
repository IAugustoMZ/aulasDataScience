import yaml
from pathlib import Path
from typing import Any

def load_yaml(path: Path) -> dict[str, Any]:
    """carrega os yamls
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}\n"
            f"Expected location: {path.resolve()}"
        )
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}