"""
descriptive.py — Comprehensive descriptive statistics for the California Housing dataset.

Entry point: run(config, base_dir)

Computes and saves:
  01_basic_info.json          — shape, dtypes, memory, capped-value count
  02_descriptive_stats.csv    — mean/std/min/max/percentiles/skew/kurtosis
  03_missing_values.json      — count and % of NaN per column
  04_ocean_proximity_counts.csv — value counts for the categorical feature
  05_outliers_iqr.csv         — per-column outlier counts via IQR fence
  06_distribution_summary.csv — skewness/kurtosis classification per feature
  07_correlation_matrix.csv   — Pearson correlation matrix (all numeric)
  08_target_correlation.csv   — features ranked by |corr| with target

All parameters are read from config/eda.yaml — no hardcoded values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Path bootstrap — allows running the script directly from the eda/ directory
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
    """Create output directories from config paths and return them."""
    dirs: dict[str, Path] = {}
    for key in ("stats_dir", "tables_dir", "figures_dir"):
        raw = config.get("paths", {}).get(key, f"outputs/{key.replace('_dir', '')}")
        path = base_dir / raw
        path.mkdir(parents=True, exist_ok=True)
        dirs[key] = path
    return dirs


def _save_json(data: Any, path: Path, logger: Any) -> None:
    """Serialize dict to JSON with indentation, converting non-serialisable types."""
    def _convert(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")

    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=_convert)
    logger.info("Saved JSON: %s", path.name)


def _save_csv(df: pd.DataFrame, path: Path, logger: Any) -> None:
    """Save DataFrame to CSV with index."""
    df.to_csv(path)
    logger.info("Saved CSV: %s  (%d rows × %d cols)", path.name, len(df), len(df.columns))


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compute_basic_info(df: pd.DataFrame, config: dict, logger: Any) -> dict:
    """
    Compute dataset shape, dtypes, memory usage, and capped-value counts.

    Returns a dict ready for JSON serialisation.
    """
    logger.info("Computing basic dataset info...")

    cap_value = (
        config.get("schema", {})
        .get("value_bounds", {})
        .get("median_house_value", {})
        .get("cap_value", 500001.0)
    )
    capped_count = int((df["median_house_value"] == cap_value).sum())

    info = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 ** 2, 4),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "capped_median_house_value": {
            "cap_value": cap_value,
            "count": capped_count,
            "pct_of_total": round(capped_count / len(df) * 100, 4),
        },
    }
    logger.info(
        "Basic info: %d rows, %d cols, %d capped target values",
        info["rows"], info["columns"], capped_count,
    )
    return info


def compute_descriptive_stats(df: pd.DataFrame, config: dict, logger: Any) -> pd.DataFrame:
    """
    Compute extended descriptive statistics for all numeric columns.

    Includes pandas describe() percentiles from config plus skewness and kurtosis.
    """
    logger.info("Computing descriptive statistics...")

    percentiles = config.get("descriptive", {}).get(
        "percentiles", [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    stats = df[numeric_cols].describe(percentiles=percentiles)
    stats.loc["skewness"] = df[numeric_cols].skew()
    stats.loc["kurtosis"] = df[numeric_cols].kurt()  # excess kurtosis (Fisher)

    logger.info("Descriptive stats computed for %d numeric columns.", len(numeric_cols))
    return stats.T  # Transpose: features as rows, statistics as columns


def compute_missing_values(df: pd.DataFrame, config: dict, logger: Any) -> dict:
    """
    Analyse missing values per column.

    Flags columns where missing% exceeds the config threshold.
    """
    logger.info("Analysing missing values...")

    threshold = config.get("descriptive", {}).get("missing_threshold", 0.05)
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(4)

    result = {}
    for col in df.columns:
        count = int(missing_counts[col])
        pct = float(missing_pct[col])
        result[col] = {
            "count": count,
            "pct": pct,
            "exceeds_threshold": pct / 100 > threshold,
        }

    total_missing = sum(v["count"] for v in result.values())
    logger.info(
        "Missing values: %d total across %d columns (threshold=%.0f%%)",
        total_missing, sum(1 for v in result.values() if v["count"] > 0), threshold * 100,
    )
    return result


def compute_ocean_proximity_counts(df: pd.DataFrame, logger: Any) -> pd.DataFrame:
    """Return value counts and percentages for ocean_proximity."""
    logger.info("Computing ocean_proximity value counts...")

    counts = df["ocean_proximity"].value_counts()
    pcts = (counts / len(df) * 100).round(4)
    result = pd.DataFrame({"count": counts, "pct": pcts})
    result.index.name = "ocean_proximity"
    return result


def compute_outliers_iqr(df: pd.DataFrame, config: dict, logger: Any) -> pd.DataFrame:
    """
    Detect outliers using the IQR fence method.

    Lower fence = Q1 - multiplier * IQR
    Upper fence = Q3 + multiplier * IQR
    """
    logger.info("Detecting outliers via IQR method...")

    multiplier = config.get("descriptive", {}).get("outlier_iqr_multiplier", 1.5)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    records = []
    for col in numeric_cols:
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        n_outliers = int(((series < lower) | (series > upper)).sum())
        records.append({
            "feature": col,
            "q1": round(q1, 4),
            "q3": round(q3, 4),
            "iqr": round(iqr, 4),
            "lower_fence": round(lower, 4),
            "upper_fence": round(upper, 4),
            "n_outliers": n_outliers,
            "pct_outliers": round(n_outliers / len(series) * 100, 4),
        })

    result = pd.DataFrame(records).set_index("feature")
    logger.info("Outlier detection complete (multiplier=%.1f).", multiplier)
    return result


def compute_distribution_summary(df: pd.DataFrame, config: dict, logger: Any) -> pd.DataFrame:
    """
    Classify each numeric feature's distribution shape using skewness and kurtosis.

    Classifications:
      - normal         : |skew| <= skewness_threshold
      - right-skewed   : skew > threshold
      - left-skewed    : skew < -threshold
      - heavy-tailed   : excess kurtosis > kurtosis_threshold (applied on top of skew label)
    """
    logger.info("Classifying distribution shapes...")

    skew_thresh = config.get("descriptive", {}).get("skewness_threshold", 0.5)
    kurt_thresh = config.get("descriptive", {}).get("kurtosis_threshold", 3.0)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    records = []
    for col in numeric_cols:
        skew = float(df[col].skew())
        kurt = float(df[col].kurt())  # excess kurtosis

        if abs(skew) <= skew_thresh:
            shape = "normal"
        elif skew > skew_thresh:
            shape = "right-skewed"
        else:
            shape = "left-skewed"

        if kurt > kurt_thresh:
            shape = f"{shape} / heavy-tailed"

        records.append({
            "feature": col,
            "skewness": round(skew, 6),
            "excess_kurtosis": round(kurt, 6),
            "distribution_shape": shape,
            "flagged_skewed": abs(skew) > skew_thresh,
            "flagged_heavy_tailed": kurt > kurt_thresh,
        })

    result = pd.DataFrame(records).set_index("feature")
    logger.info("Distribution classification complete for %d features.", len(numeric_cols))
    return result


def compute_correlation_matrix(df: pd.DataFrame, logger: Any) -> pd.DataFrame:
    """Compute Pearson correlation matrix for all numeric columns."""
    logger.info("Computing Pearson correlation matrix...")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    corr = df[numeric_cols].corr(method="pearson").round(6)
    return corr


def compute_target_correlation(
    df: pd.DataFrame, config: dict, logger: Any
) -> pd.DataFrame:
    """
    Rank all numeric features by absolute Pearson correlation with the target.

    Returns a DataFrame sorted descending by |correlation|.
    """
    logger.info("Computing feature-target correlations...")

    target = config.get("schema", {}).get("target", "median_house_value")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target]

    records = []
    for col in feature_cols:
        corr_val = float(df[col].corr(df[target], method="pearson"))
        records.append({"feature": col, "pearson_corr": round(corr_val, 6)})

    result = (
        pd.DataFrame(records)
        .set_index("feature")
        .assign(abs_corr=lambda x: x["pearson_corr"].abs())
        .sort_values("abs_corr", ascending=False)
        .drop(columns="abs_corr")
    )
    logger.info("Target correlation ranking complete.")
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(config: dict, base_dir: Path) -> dict:
    """
    Execute the full descriptive statistics analysis.

    Args:
        config:   Merged configuration dict loaded from eda.yaml.
        base_dir: Repository root used to resolve relative paths.

    Returns:
        Dict with keys: basic_info, descriptive_stats, missing_values,
        ocean_proximity_counts, outliers_iqr, distribution_summary,
        correlation_matrix, target_correlation.
    """
    _bootstrap_src_path(base_dir)
    from utils.logger import get_logger  # noqa: PLC0415 — deferred for path bootstrap

    logger = get_logger("eda.descriptive", config.get("logging", {}))
    logger.info("=== Descriptive Statistics EDA started ===")

    # Resolve paths
    dirs = _ensure_output_dirs(base_dir, config)
    stats_dir = dirs["stats_dir"]
    data_path = base_dir / config.get("paths", {}).get(
        "input_data", "data/processed/house_price.parquet"
    )

    try:
        df = _load_data(data_path, logger)
    except FileNotFoundError as exc:
        logger.error("Failed to load data: %s", exc)
        raise

    results: dict[str, Any] = {}

    # 1. Basic info
    try:
        basic_info = compute_basic_info(df, config, logger)
        _save_json(basic_info, stats_dir / "01_basic_info.json", logger)
        results["basic_info"] = basic_info
    except Exception as exc:
        logger.error("Basic info computation failed: %s", exc)
        raise

    # 2. Descriptive statistics
    try:
        desc_stats = compute_descriptive_stats(df, config, logger)
        _save_csv(desc_stats, stats_dir / "02_descriptive_stats.csv", logger)
        results["descriptive_stats"] = desc_stats
    except Exception as exc:
        logger.error("Descriptive stats computation failed: %s", exc)
        raise

    # 3. Missing values
    try:
        missing = compute_missing_values(df, config, logger)
        _save_json(missing, stats_dir / "03_missing_values.json", logger)
        results["missing_values"] = missing
    except Exception as exc:
        logger.error("Missing values analysis failed: %s", exc)
        raise

    # 4. Ocean proximity counts
    try:
        prox_counts = compute_ocean_proximity_counts(df, logger)
        _save_csv(prox_counts, stats_dir / "04_ocean_proximity_counts.csv", logger)
        results["ocean_proximity_counts"] = prox_counts
    except Exception as exc:
        logger.error("Ocean proximity counts failed: %s", exc)
        raise

    # 5. Outlier detection
    try:
        outliers = compute_outliers_iqr(df, config, logger)
        _save_csv(outliers, stats_dir / "05_outliers_iqr.csv", logger)
        results["outliers_iqr"] = outliers
    except Exception as exc:
        logger.error("Outlier detection failed: %s", exc)
        raise

    # 6. Distribution summary
    try:
        dist_summary = compute_distribution_summary(df, config, logger)
        _save_csv(dist_summary, stats_dir / "06_distribution_summary.csv", logger)
        results["distribution_summary"] = dist_summary
    except Exception as exc:
        logger.error("Distribution summary failed: %s", exc)
        raise

    # 7. Correlation matrix
    try:
        corr_matrix = compute_correlation_matrix(df, logger)
        _save_csv(corr_matrix, stats_dir / "07_correlation_matrix.csv", logger)
        results["correlation_matrix"] = corr_matrix
    except Exception as exc:
        logger.error("Correlation matrix failed: %s", exc)
        raise

    # 8. Target correlation
    try:
        target_corr = compute_target_correlation(df, config, logger)
        _save_csv(target_corr, stats_dir / "08_target_correlation.csv", logger)
        results["target_correlation"] = target_corr
    except Exception as exc:
        logger.error("Target correlation failed: %s", exc)
        raise

    logger.info("=== Descriptive Statistics EDA complete — %d outputs saved ===", 8)
    return results


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml

    _SCRIPT_DIR = Path(__file__).resolve().parent
    _BASE_DIR = _SCRIPT_DIR.parent  # aula02/

    # Bootstrap src path before importing internal utilities
    _bootstrap_src_path(_BASE_DIR)

    _CONFIG_PATH = _BASE_DIR / "config" / "eda.yaml"
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"eda.yaml not found at {_CONFIG_PATH}")

    with _CONFIG_PATH.open("r", encoding="utf-8") as _fh:
        _config = yaml.safe_load(_fh)

    run(_config, _BASE_DIR)
