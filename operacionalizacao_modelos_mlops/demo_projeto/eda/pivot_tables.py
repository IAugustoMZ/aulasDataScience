"""
pivot_tables.py — Pivot table and contingency analysis for the California Housing dataset.

Entry point: run(config, base_dir)

Produces:
  01_pivot_proximity_income.csv   — ocean_proximity × income_category → house value stats
  02_pivot_proximity_age.csv      — ocean_proximity × age_category → house value stats
  03_pivot_income_age.csv         — income_category × age_category → mean house value
  04_stats_by_proximity.csv       — all numeric feature stats per ocean_proximity group
  05_contingency_proximity_income.csv — count contingency table
  06_top10_blocks.csv             — top-10 census blocks by median_house_value
  07_bottom10_blocks.csv          — bottom-10 census blocks by median_house_value
  08_crosstab_value_by_income_proximity.csv — mean house value cross-tab

All parameters are read from config/eda.yaml — no hardcoded values.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

def _bootstrap_src_path(base_dir: Path) -> None:
    """Add base_dir/src to sys.path so internal utilities are importable."""
    src_path = str(base_dir / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_data(data_path: Path, logger: Any) -> pd.DataFrame:
    """Load Parquet file with descriptive error messages."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Input data not found: {data_path}\n"
            f"Expected resolved path: {data_path.resolve()}"
        )
    logger.info("Loading dataset from: %s", data_path)
    df = pd.read_parquet(data_path)
    logger.info("Dataset loaded — shape: %s", df.shape)
    return df


def _ensure_output_dirs(base_dir: Path, config: dict) -> dict[str, Path]:
    """Create output directories and return path dict."""
    dirs: dict[str, Path] = {}
    for key in ("stats_dir", "tables_dir", "figures_dir"):
        raw = config.get("paths", {}).get(key, f"outputs/{key.replace('_dir', '')}")
        path = base_dir / raw
        path.mkdir(parents=True, exist_ok=True)
        dirs[key] = path
    return dirs


def _save_csv(df: pd.DataFrame, path: Path, logger: Any) -> None:
    """Save DataFrame to CSV."""
    df.to_csv(path)
    logger.info("Saved CSV: %s  (%d rows × %d cols)", path.name, len(df), len(df.columns))


# ---------------------------------------------------------------------------
# Binning helpers
# ---------------------------------------------------------------------------

def _add_binned_columns(df: pd.DataFrame, config: dict, logger: Any) -> pd.DataFrame:
    """
    Add derived categorical columns via pd.cut using cuts/labels from config.

    Creates: income_category, age_category, house_value_category,
             population_density_category.

    The population_density_category is derived from the raw population column
    (not a computed density) because block-level area is not in the dataset.
    """
    df = df.copy()
    bins_cfg = config.get("pivot_tables", {}).get("bins", {})

    bin_map = {
        "income_category": ("median_income", "income"),
        "age_category": ("housing_median_age", "age"),
        "house_value_category": ("median_house_value", "house_value"),
        "population_density_category": ("population", "population_density"),
    }

    for new_col, (src_col, cfg_key) in bin_map.items():
        if src_col not in df.columns:
            logger.warning("Source column '%s' not found — skipping bin '%s'.", src_col, new_col)
            continue
        if cfg_key not in bins_cfg:
            logger.warning("Bin config '%s' missing from eda.yaml — skipping '%s'.", cfg_key, new_col)
            continue
        cuts = bins_cfg[cfg_key]["cuts"]
        labels = bins_cfg[cfg_key]["labels"]
        try:
            df[new_col] = pd.cut(
                df[src_col],
                bins=cuts,
                labels=labels,
                right=True,
                include_lowest=True,
            )
            logger.info("Created '%s' with %d categories.", new_col, len(labels))
        except Exception as exc:
            logger.error("Binning '%s' → '%s' failed: %s", src_col, new_col, exc)
            raise

    return df


# ---------------------------------------------------------------------------
# Pivot / aggregation functions
# ---------------------------------------------------------------------------

def pivot_proximity_income(df: pd.DataFrame, config: dict, logger: Any) -> pd.DataFrame:
    """
    Pivot: ocean_proximity × income_category → mean/median/count of median_house_value.

    Produces a multi-level column DataFrame flattened with '_' separator.
    """
    logger.info("Building pivot: ocean_proximity × income_category ...")
    target = config.get("schema", {}).get("target", "median_house_value")

    pivot = df.pivot_table(
        values=target,
        index="ocean_proximity",
        columns="income_category",
        aggfunc=["mean", "median", "count"],
    )
    pivot.columns = [f"{agg}_{cat}" for agg, cat in pivot.columns]
    pivot = pivot.round(2)
    return pivot


def pivot_proximity_age(df: pd.DataFrame, config: dict, logger: Any) -> pd.DataFrame:
    """Pivot: ocean_proximity × age_category → mean/median/count of median_house_value."""
    logger.info("Building pivot: ocean_proximity × age_category ...")
    target = config.get("schema", {}).get("target", "median_house_value")

    pivot = df.pivot_table(
        values=target,
        index="ocean_proximity",
        columns="age_category",
        aggfunc=["mean", "median", "count"],
    )
    pivot.columns = [f"{agg}_{cat}" for agg, cat in pivot.columns]
    pivot = pivot.round(2)
    return pivot


def pivot_income_age(df: pd.DataFrame, config: dict, logger: Any) -> pd.DataFrame:
    """Pivot: income_category × age_category → mean median_house_value."""
    logger.info("Building pivot: income_category × age_category ...")
    target = config.get("schema", {}).get("target", "median_house_value")

    pivot = df.pivot_table(
        values=target,
        index="income_category",
        columns="age_category",
        aggfunc="mean",
    )
    pivot = pivot.round(2)
    return pivot


def stats_by_proximity(df: pd.DataFrame, config: dict, logger: Any) -> pd.DataFrame:
    """
    Compute mean/std/median/count for all numeric features grouped by ocean_proximity.

    Returns a wide DataFrame with multi-level columns flattened.
    """
    logger.info("Computing summary stats grouped by ocean_proximity ...")
    numeric_cols = config.get("schema", {}).get("numeric_features", [])
    target = config.get("schema", {}).get("target", "median_house_value")
    cols = numeric_cols + [target]
    cols = [c for c in cols if c in df.columns]

    grouped = df.groupby("ocean_proximity")[cols].agg(["mean", "std", "median", "count"])
    grouped.columns = [f"{col}_{agg}" for col, agg in grouped.columns]
    return grouped.round(4)


def contingency_proximity_income(df: pd.DataFrame, logger: Any) -> pd.DataFrame:
    """Count contingency table: ocean_proximity (rows) × income_category (columns)."""
    logger.info("Building contingency table: ocean_proximity × income_category ...")
    ct = pd.crosstab(df["ocean_proximity"], df["income_category"])
    return ct


def top_bottom_blocks(
    df: pd.DataFrame, config: dict, logger: Any, n: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return the top-N and bottom-N census blocks by median_house_value.

    Includes all original columns for full context.
    """
    logger.info("Selecting top-%d and bottom-%d blocks by house value ...", n, n)
    target = config.get("schema", {}).get("target", "median_house_value")
    all_cols = df.columns.tolist()
    top = df.nlargest(n, target)[all_cols].reset_index(drop=True)
    bottom = df.nsmallest(n, target)[all_cols].reset_index(drop=True)
    return top, bottom


def crosstab_value_by_income_proximity(
    df: pd.DataFrame, config: dict, logger: Any
) -> pd.DataFrame:
    """
    Mean median_house_value cross-tabulation: income_category (rows) × ocean_proximity (cols).

    Provides a quick visual summary of income–proximity interaction on house prices.
    """
    logger.info("Building cross-tab: mean house value by income × proximity ...")
    target = config.get("schema", {}).get("target", "median_house_value")

    ct = df.pivot_table(
        values=target,
        index="income_category",
        columns="ocean_proximity",
        aggfunc="mean",
    ).round(2)
    return ct


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(config: dict, base_dir: Path) -> dict:
    """
    Execute the full pivot table analysis.

    Args:
        config:   Merged configuration dict loaded from eda.yaml.
        base_dir: Repository root used to resolve relative paths.

    Returns:
        Dict with keys corresponding to each produced table.
    """
    _bootstrap_src_path(base_dir)
    from utils.logger import get_logger  # noqa: PLC0415

    logger = get_logger("eda.pivot_tables", config.get("logging", {}))
    logger.info("=== Pivot Tables EDA started ===")

    dirs = _ensure_output_dirs(base_dir, config)
    tables_dir = dirs["tables_dir"]
    data_path = base_dir / config.get("paths", {}).get(
        "input_data", "data/processed/house_price.parquet"
    )

    try:
        df_raw = _load_data(data_path, logger)
    except FileNotFoundError as exc:
        logger.error("Failed to load data: %s", exc)
        raise

    # Enrich with binned columns required for pivots
    try:
        df = _add_binned_columns(df_raw, config, logger)
    except Exception as exc:
        logger.error("Binning step failed: %s", exc)
        raise

    results: dict[str, Any] = {}

    # 1. Pivot: proximity × income
    try:
        t1 = pivot_proximity_income(df, config, logger)
        _save_csv(t1, tables_dir / "01_pivot_proximity_income.csv", logger)
        results["pivot_proximity_income"] = t1
    except Exception as exc:
        logger.error("Pivot proximity×income failed: %s", exc)
        raise

    # 2. Pivot: proximity × age
    try:
        t2 = pivot_proximity_age(df, config, logger)
        _save_csv(t2, tables_dir / "02_pivot_proximity_age.csv", logger)
        results["pivot_proximity_age"] = t2
    except Exception as exc:
        logger.error("Pivot proximity×age failed: %s", exc)
        raise

    # 3. Pivot: income × age
    try:
        t3 = pivot_income_age(df, config, logger)
        _save_csv(t3, tables_dir / "03_pivot_income_age.csv", logger)
        results["pivot_income_age"] = t3
    except Exception as exc:
        logger.error("Pivot income×age failed: %s", exc)
        raise

    # 4. Stats by proximity
    try:
        t4 = stats_by_proximity(df, config, logger)
        _save_csv(t4, tables_dir / "04_stats_by_proximity.csv", logger)
        results["stats_by_proximity"] = t4
    except Exception as exc:
        logger.error("Stats by proximity failed: %s", exc)
        raise

    # 5. Contingency table
    try:
        t5 = contingency_proximity_income(df, logger)
        _save_csv(t5, tables_dir / "05_contingency_proximity_income.csv", logger)
        results["contingency_proximity_income"] = t5
    except Exception as exc:
        logger.error("Contingency table failed: %s", exc)
        raise

    # 6 & 7. Top/bottom blocks
    try:
        top10, bot10 = top_bottom_blocks(df_raw, config, logger, n=10)
        _save_csv(top10, tables_dir / "06_top10_blocks.csv", logger)
        _save_csv(bot10, tables_dir / "07_bottom10_blocks.csv", logger)
        results["top10_blocks"] = top10
        results["bottom10_blocks"] = bot10
    except Exception as exc:
        logger.error("Top/bottom blocks failed: %s", exc)
        raise

    # 8. Cross-tab: mean value by income × proximity
    try:
        t8 = crosstab_value_by_income_proximity(df, config, logger)
        _save_csv(t8, tables_dir / "08_crosstab_value_by_income_proximity.csv", logger)
        results["crosstab_value_by_income_proximity"] = t8
    except Exception as exc:
        logger.error("Cross-tab failed: %s", exc)
        raise

    logger.info("=== Pivot Tables EDA complete — 8 tables saved ===")
    return results


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml

    _SCRIPT_DIR = Path(__file__).resolve().parent
    _BASE_DIR = _SCRIPT_DIR.parent

    _bootstrap_src_path(_BASE_DIR)

    _CONFIG_PATH = _BASE_DIR / "config" / "eda.yaml"
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"eda.yaml not found at {_CONFIG_PATH}")

    with _CONFIG_PATH.open("r", encoding="utf-8") as _fh:
        _config = yaml.safe_load(_fh)

    run(_config, _BASE_DIR)
