"""
Single entry point for loading YAML configs.
All modules receive config dicts from here — no module loads its own YAML.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def resolve_root() -> Path:
    """Return the project root (parent of src/)."""
    return Path(__file__).parent.parent.parent


def load_config(path: Path | str) -> dict:
    """Load a single YAML file. Path can be absolute or relative to project root."""
    p = Path(path)
    if not p.is_absolute():
        p = resolve_root() / p
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_all_configs(root: Path | None = None) -> dict[str, dict]:
    """Load all standard project configs and return as a named dict."""
    root = root or resolve_root()
    names = ["data", "labels", "eda", "agents"]
    configs: dict[str, dict] = {}
    for name in names:
        path = root / "configs" / f"{name}.yaml"
        if path.exists():
            configs[name] = load_config(path)
    return configs
