"""
Data pipeline for webinar_ml.
Responsibilities: load Parquet files, validate schema, enforce dtypes,
run basic quality checks, and re-export processed splits.

All parameters come from configs/data.yaml. Nothing is hardcoded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_root() -> Path:
    # src/data/pipeline.py  ->  ../../  =  project root
    return Path(__file__).parent.parent.parent


# ── Load ──────────────────────────────────────────────────────────────────────

def load_raw(config: dict, root: Path | None = None) -> pd.DataFrame:
    root = root or resolve_root()
    path = root / config["paths"]["raw"]
    return pd.read_parquet(path, engine=config["parquet"]["engine"])


def load_split(split: str, config: dict, root: Path | None = None) -> pd.DataFrame:
    """Load one of: 'train' | 'test' | 'unannotated'."""
    root = root or resolve_root()
    path = root / config["paths"][split]
    return pd.read_parquet(path, engine=config["parquet"]["engine"])


def load_all_splits(config: dict, root: Path | None = None) -> dict[str, pd.DataFrame]:
    return {
        split: load_split(split, config, root)
        for split in ("train", "test", "unannotated")
    }


# ── Schema validation ──────────────────────────────────────────────────────────

class SchemaValidationError(Exception):
    pass


def _expected_columns(schema_cfg: dict) -> list[str]:
    return list(schema_cfg.keys())


def _check_missing_columns(df: pd.DataFrame, expected: list[str]) -> list[str]:
    return [c for c in expected if c not in df.columns]


def _check_nulls(df: pd.DataFrame, schema_cfg: dict) -> dict[str, int]:
    violations: dict[str, int] = {}
    for col, spec in schema_cfg.items():
        if col not in df.columns:
            continue
        if not spec.get("nullable", True):
            n_null = int(df[col].isna().sum())
            if n_null > 0:
                violations[col] = n_null
    return violations


def validate_schema(df: pd.DataFrame, config: dict, name: str = "dataset") -> list[str]:
    """
    Validate df against the schema defined in config['schema'].
    Returns a list of warning/error strings. Raises SchemaValidationError
    if any non-nullable column has nulls or expected columns are missing.
    """
    schema_cfg = config["schema"]
    warnings: list[str] = []

    missing = _check_missing_columns(df, _expected_columns(schema_cfg))
    if missing:
        raise SchemaValidationError(
            f"[{name}] Missing columns: {missing}"
        )

    null_violations = _check_nulls(df, schema_cfg)
    if null_violations:
        detail = ", ".join(f"{c}={n}" for c, n in null_violations.items())
        raise SchemaValidationError(
            f"[{name}] Non-nullable columns have nulls: {detail}"
        )

    extra_cols = [c for c in df.columns if c not in schema_cfg]
    if extra_cols:
        warnings.append(f"[{name}] Extra columns not in schema: {extra_cols}")

    return warnings


# ── Dtype enforcement ──────────────────────────────────────────────────────────

_DTYPE_MAP: dict[str, Any] = {
    "string":   "object",
    "bool":     "bool",
    "date":     "object",
    "datetime": "object",
    "int":      "int64",
    "float":    "float64",
}


def enforce_dtypes(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Cast columns to expected dtypes where safe. Returns a copy."""
    df = df.copy()
    for col, spec in config["schema"].items():
        if col not in df.columns:
            continue
        dtype_key = spec.get("dtype", "string")
        if dtype_key in ("date", "datetime"):
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
        elif dtype_key == "bool":
            df[col] = df[col].astype(bool)
    return df


# ── Quality report ─────────────────────────────────────────────────────────────

def quality_report(df: pd.DataFrame, config: dict, name: str = "dataset") -> dict:
    """
    Return a structured quality report dict. Used by the EDA notebook
    to surface quality issues without coupling to display logic.
    """
    schema_cfg = config["schema"]
    noise_cfg = config["generation"]["noise"]

    n_total = len(df)
    n_annotated = int(df["anotado"].sum()) if "anotado" in df.columns else None
    n_unannotated = int((~df["anotado"]).sum()) if "anotado" in df.columns else None
    n_mislabeled = int(df["ruido"].sum()) if "ruido" in df.columns else None
    n_ambiguous = int(df["ambiguo"].sum()) if "ambiguo" in df.columns else None

    null_counts = {c: int(df[c].isna().sum()) for c in df.columns}
    duplicate_ids = int(df["id"].duplicated().sum()) if "id" in df.columns else 0

    class_dist: dict[str, float] = {}
    if "classe_risco" in df.columns:
        dist = df["classe_risco"].value_counts(normalize=True, dropna=True)
        class_dist = dist.round(4).to_dict()

    word_stats: dict[str, float] = {}
    if "relato" in df.columns:
        lengths = df["relato"].dropna().str.split().str.len()
        word_stats = {
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "mean": round(float(lengths.mean()), 1),
            "median": round(float(lengths.median()), 1),
            "std": round(float(lengths.std()), 1),
            "unique_relatos": int(df["relato"].nunique()),
            "unique_pct": round(df["relato"].nunique() / n_total, 4),
        }

    return {
        "name": name,
        "n_total": n_total,
        "n_annotated": n_annotated,
        "n_unannotated": n_unannotated,
        "n_mislabeled": n_mislabeled,
        "n_ambiguous": n_ambiguous,
        "null_counts": null_counts,
        "duplicate_ids": duplicate_ids,
        "class_distribution": class_dist,
        "word_stats": word_stats,
        "expected_noise": {
            "mislabeled_rate": noise_cfg["mislabeled_rate"],
            "ambiguous_rate": noise_cfg["ambiguous_rate"],
            "unannotated_rate": noise_cfg["unannotated_rate"],
        },
    }


# ── Imbalance helpers ─────────────────────────────────────────────────────────

def imbalance_ratio(df: pd.DataFrame, label_col: str) -> float:
    """Ratio of majority to minority class count (annotated records only)."""
    counts = df[label_col].value_counts()
    return round(counts.max() / counts.min(), 2)


def class_counts(df: pd.DataFrame, label_col: str, class_order: list[str]) -> pd.DataFrame:
    """Return count + fraction per class, sorted by class_order."""
    counts = df[label_col].value_counts()
    fracs = df[label_col].value_counts(normalize=True)
    result = pd.DataFrame({"count": counts, "fraction": fracs.round(4)})
    present = [c for c in class_order if c in result.index]
    return result.reindex(present)


# ── CLI (re-export splits from raw) ───────────────────────────────────────────

def _cli_reexport(config: dict, root: Path) -> None:
    """Re-run train/test/unannotated split from raw Parquet. Idempotent."""
    from sklearn.model_selection import train_test_split

    df = load_raw(config, root)
    validate_schema(df, config, name="raw")
    df = enforce_dtypes(df, config)

    annotated = df[df["anotado"]].copy()
    unannotated = df[~df["anotado"]].copy()

    split_cfg = config["split"]
    train_df, test_df = train_test_split(
        annotated,
        test_size=split_cfg["test_size"],
        stratify=annotated[split_cfg["stratify_by"]],
        random_state=config["generation"]["random_seed"],
    )

    parquet_opts = {
        "engine": config["parquet"]["engine"],
        "compression": config["parquet"]["compression"],
        "index": config["parquet"]["index"],
    }

    for split_name, split_df in [("train", train_df), ("test", test_df), ("unannotated", unannotated)]:
        out = root / config["paths"][split_name]
        out.parent.mkdir(parents=True, exist_ok=True)
        split_df.to_parquet(out, **parquet_opts)
        print(f"[OK] {split_name:<12}: {out}  ({len(split_df):,} records)")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Re-export train/test/unannotated splits from raw Parquet")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    root = resolve_root()
    config = load_config(root / args.config)
    _cli_reexport(config, root)
    print("Done.")


if __name__ == "__main__":
    main()
