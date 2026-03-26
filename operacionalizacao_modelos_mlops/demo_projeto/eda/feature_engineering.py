"""
feature_engineering.py — Feature engineering and enrichment for the California Housing dataset.

Entry point: run(config, base_dir)

Computes and saves:
  fig_23_engineered_features_dist.png      — Histograms of all engineered features
  fig_24_log_vs_raw_distributions.png      — Side-by-side raw vs log distributions
  fig_25_distance_features_map.png         — Geographic scatter coloured by city distance
  fig_26_feature_importance_correlation.png — Horizontal bar chart |corr| with target
  stats/20_engineered_feature_correlations.csv  — All features ranked by |corr| with target
  stats/21_feature_engineering_summary.json     — Created features + correlation to target
  outputs/tables/09_enriched_dataset_sample.csv — First 1 000 rows of enriched dataset

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
    """Create output directories declared in config and return their paths."""
    paths = config.get("paths", {})
    dirs: dict[str, Path] = {
        "figures": base_dir / paths.get("figures_dir", "outputs/figures"),
        "tables":  base_dir / paths.get("tables_dir",  "outputs/tables"),
        "stats":   base_dir / paths.get("stats_dir",   "outputs/stats"),
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _savefig(fig: plt.Figure, path: Path, dpi: int = 120) -> None:
    """Apply tight_layout, save figure, and close all matplotlib state."""
    try:
        plt.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close("all")


# ---------------------------------------------------------------------------
# Feature engineering steps
# ---------------------------------------------------------------------------

def _build_ratio_features(df: pd.DataFrame, ratio_cfg: dict, logger: Any) -> pd.DataFrame:
    """Create ratio features defined in config. Safe-divides with replace(inf → NaN)."""
    created: list[str] = []
    for feat_name, spec in ratio_cfg.items():
        num = spec.get("numerator")
        den = spec.get("denominator")
        if num not in df.columns or den not in df.columns:
            logger.warning("Skipping ratio '%s': missing column(s) '%s' or '%s'", feat_name, num, den)
            continue
        df[feat_name] = (df[num] / df[den].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        created.append(feat_name)
    logger.info("Ratio features created: %s", created)
    return df


def _build_log_features(df: pd.DataFrame, log_feats: list[str], logger: Any) -> pd.DataFrame:
    """Apply np.log1p to each feature listed; source column must already exist in df."""
    created: list[str] = []
    for feat in log_feats:
        if feat not in df.columns:
            logger.warning("Skipping log transform for '%s': column not found", feat)
            continue
        log_name = f"log_{feat}"
        df[log_name] = np.log1p(df[feat].clip(lower=0))
        created.append(log_name)
    logger.info("Log features created: %s", created)
    return df


def _build_geo_distance_features(
    df: pd.DataFrame,
    ref_points: dict,
    lat_col: str,
    lon_col: str,
    logger: Any,
) -> pd.DataFrame:
    """
    Compute Euclidean distance (degrees) from each reference city.

    Adds one column per city (e.g. 'dist_san_francisco') plus:
      - nearest_city_distance  — minimum distance across all cities
      - nearest_city_name      — name of the closest city
    """
    dist_cols: dict[str, str] = {}  # city_key → column name
    for city_key, coords in ref_points.items():
        ref_lat = coords.get("lat")
        ref_lon = coords.get("lon")
        col_name = f"dist_{city_key}"
        df[col_name] = np.sqrt(
            (df[lat_col] - ref_lat) ** 2 + (df[lon_col] - ref_lon) ** 2
        )
        dist_cols[city_key] = col_name

    if not dist_cols:
        logger.warning("No geographic reference points found in config; skipping geo distance features.")
        return df

    dist_matrix = df[list(dist_cols.values())]
    df["nearest_city_distance"] = dist_matrix.min(axis=1)
    nearest_idx = dist_matrix.values.argmin(axis=1)
    city_names = list(dist_cols.keys())
    df["nearest_city_name"] = [city_names[i] for i in nearest_idx]
    logger.info("Geographic distance features created for cities: %s", list(dist_cols.keys()))
    return df


def _build_polynomial_features(df: pd.DataFrame, poly_cfg: dict, logger: Any) -> pd.DataFrame:
    """
    Create squared terms and interaction term for polynomial features.

    Config key 'features' specifies the base features; degree must be 2.
    """
    features = poly_cfg.get("features", [])
    if len(features) < 2:
        logger.warning("Polynomial features require at least 2 base features; got %s", features)
        return df

    f0, f1 = features[0], features[1]
    missing = [f for f in [f0, f1] if f not in df.columns]
    if missing:
        logger.warning("Skipping polynomial features — missing columns: %s", missing)
        return df

    df[f"{f0}_squared"] = df[f0] ** 2
    df[f"{f1}_squared"] = df[f1] ** 2
    df[f"{f0}_x_{f1}"] = df[f0] * df[f1]
    logger.info(
        "Polynomial features created: %s_squared, %s_squared, %s_x_%s", f0, f1, f0, f1
    )
    return df


def _build_categorical_encoding(df: pd.DataFrame, logger: Any) -> pd.DataFrame:
    """
    Encode ocean_proximity:
      - Label encoding (ordinal by distance from coast)
      - One-hot dummies
    """
    if "ocean_proximity" not in df.columns:
        logger.warning("Column 'ocean_proximity' not found; skipping categorical encoding.")
        return df

    # Ordinal: higher number = further from coast
    ordinal_map = {
        "ISLAND":     0,
        "NEAR BAY":   1,
        "NEAR OCEAN": 2,
        "<1H OCEAN":  3,
        "INLAND":     4,
    }
    df["ocean_proximity_encoded"] = df["ocean_proximity"].map(ordinal_map)
    unknown = df["ocean_proximity_encoded"].isna().sum()
    if unknown > 0:
        logger.warning("%d rows had unknown ocean_proximity values — encoded as NaN", unknown)

    # One-hot dummies (drop_first=False to keep all categories explicit)
    dummies = pd.get_dummies(df["ocean_proximity"], prefix="op")
    # Cast bool columns to int so they survive CSV round-trips
    dummies = dummies.astype(int)
    df = pd.concat([df, dummies], axis=1)
    logger.info(
        "Categorical encoding done — label column: ocean_proximity_encoded; "
        "dummy columns: %s", list(dummies.columns)
    )
    return df


# ---------------------------------------------------------------------------
# Analysis & output helpers
# ---------------------------------------------------------------------------

def _correlation_analysis(
    df: pd.DataFrame,
    target: str,
    stats_dir: Path,
    logger: Any,
) -> pd.DataFrame:
    """
    Compute Pearson |correlation| of every numeric feature with the target.

    Saves ranked results to stats/20_engineered_feature_correlations.csv.
    Returns the correlation Series sorted descending by |corr|.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if target not in numeric_df.columns:
        logger.error("Target column '%s' not found in numeric features.", target)
        return pd.DataFrame()

    corr_series = numeric_df.corr()[target].drop(labels=[target], errors="ignore")
    corr_df = (
        corr_series
        .rename("correlation")
        .to_frame()
        .assign(abs_correlation=lambda x: x["correlation"].abs())
        .sort_values("abs_correlation", ascending=False)
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    out_path = stats_dir / "20_engineered_feature_correlations.csv"
    corr_df.to_csv(out_path, index=False)
    logger.info("Saved correlation analysis → %s", out_path)
    return corr_df


def _plot_engineered_distributions(
    df: pd.DataFrame,
    engineered_cols: list[str],
    figures_dir: Path,
    vis_cfg: dict,
    logger: Any,
) -> None:
    """Grid of histograms for all engineered features — fig_23."""
    cols_present = [c for c in engineered_cols if c in df.columns]
    if not cols_present:
        logger.warning("No engineered columns to plot distributions for.")
        return

    n_cols = 4
    n_rows = max(1, int(np.ceil(len(cols_present) / n_cols)))
    dpi = vis_cfg.get("figure_dpi", 120)
    bins = vis_cfg.get("histogram", {}).get("bins", 50)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes_flat = np.array(axes).flatten()

    for ax, col in zip(axes_flat, cols_present):
        data = df[col].dropna()
        ax.hist(data, bins=bins, color="steelblue", alpha=0.8, edgecolor="white")
        ax.set_title(col, fontsize=8)
        ax.set_xlabel("", fontsize=7)
        ax.tick_params(labelsize=7)

    for ax in axes_flat[len(cols_present):]:
        ax.set_visible(False)

    fig.suptitle("Engineered Feature Distributions", fontsize=12, y=1.01)
    out_path = figures_dir / "fig_23_engineered_features_dist.png"
    _savefig(fig, out_path, dpi=dpi)
    logger.info("Saved → %s", out_path)


def _plot_log_vs_raw(
    df: pd.DataFrame,
    log_feats: list[str],
    figures_dir: Path,
    vis_cfg: dict,
    logger: Any,
) -> None:
    """
    Side-by-side raw vs log comparison for the 4 most skewed features — fig_24.
    """
    # Pick top-4 most skewed raw features among those that have a log counterpart
    skew_scores: list[tuple[float, str]] = []
    for feat in log_feats:
        if feat in df.columns:
            skew_scores.append((abs(df[feat].dropna().skew()), feat))

    skew_scores.sort(reverse=True)
    top4 = [feat for _, feat in skew_scores[:4]]

    if not top4:
        logger.warning("No raw features available for log-vs-raw plot.")
        return

    dpi = vis_cfg.get("figure_dpi", 120)
    bins = vis_cfg.get("histogram", {}).get("bins", 50)
    n = len(top4)

    fig, axes = plt.subplots(n, 2, figsize=(12, n * 3))
    if n == 1:
        axes = [axes]

    for row_axes, feat in zip(axes, top4):
        log_col = f"log_{feat}"
        ax_raw, ax_log = row_axes[0], row_axes[1]

        raw_data = df[feat].dropna()
        ax_raw.hist(raw_data, bins=bins, color="salmon", alpha=0.85, edgecolor="white")
        ax_raw.set_title(f"Raw: {feat}  (skew={raw_data.skew():.2f})", fontsize=9)
        ax_raw.tick_params(labelsize=8)

        if log_col in df.columns:
            log_data = df[log_col].dropna()
            ax_log.hist(log_data, bins=bins, color="seagreen", alpha=0.85, edgecolor="white")
            ax_log.set_title(f"log1p: {feat}  (skew={log_data.skew():.2f})", fontsize=9)
        else:
            ax_log.set_title(f"log1p: {feat}  (not available)", fontsize=9)
        ax_log.tick_params(labelsize=8)

    fig.suptitle("Raw vs Log-Transformed Distributions (Top 4 Most Skewed)", fontsize=11)
    out_path = figures_dir / "fig_24_log_vs_raw_distributions.png"
    _savefig(fig, out_path, dpi=dpi)
    logger.info("Saved → %s", out_path)


def _plot_distance_map(
    df: pd.DataFrame,
    ref_points: dict,
    figures_dir: Path,
    vis_cfg: dict,
    logger: Any,
) -> None:
    """
    4-subplot geographic scatter: colour = distance from each reference city — fig_25.
    (Shows up to 4 cities; remaining cities are skipped.)
    """
    lat_col = "latitude"
    lon_col = "longitude"
    if lat_col not in df.columns or lon_col not in df.columns:
        logger.warning("Latitude/longitude columns not found; skipping distance map.")
        return

    city_keys = list(ref_points.keys())[:4]
    dpi = vis_cfg.get("figure_dpi", 120)
    alpha = vis_cfg.get("geo_map", {}).get("alpha", 0.4)
    size_default = 2

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()

    for ax, city_key in zip(axes_flat, city_keys):
        col_name = f"dist_{city_key}"
        if col_name not in df.columns:
            ax.set_title(f"{city_key} (data missing)", fontsize=9)
            continue

        sc = ax.scatter(
            df[lon_col], df[lat_col],
            c=df[col_name], cmap="YlOrRd",
            s=size_default, alpha=alpha
        )
        plt.colorbar(sc, ax=ax, label="Distance (degrees)")
        city_name_display = city_key.replace("_", " ").title()
        ax.set_title(f"Distance from {city_name_display}", fontsize=9)
        ax.set_xlabel("Longitude", fontsize=8)
        ax.set_ylabel("Latitude", fontsize=8)
        ax.tick_params(labelsize=7)

    for ax in axes_flat[len(city_keys):]:
        ax.set_visible(False)

    fig.suptitle("Geographic Distance Features", fontsize=12)
    out_path = figures_dir / "fig_25_distance_features_map.png"
    _savefig(fig, out_path, dpi=dpi)
    logger.info("Saved → %s", out_path)


def _plot_feature_importance(
    corr_df: pd.DataFrame,
    figures_dir: Path,
    vis_cfg: dict,
    logger: Any,
) -> None:
    """
    Horizontal bar chart of |correlation with target| for all features — fig_26.
    Shows top 40 features maximum to keep the chart readable.
    """
    if corr_df.empty:
        logger.warning("Correlation dataframe is empty; skipping importance plot.")
        return

    top_n = min(40, len(corr_df))
    plot_df = corr_df.head(top_n)
    dpi = vis_cfg.get("figure_dpi", 120)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    colors = ["#2196F3" if v >= 0 else "#F44336"
              for v in plot_df["correlation"]]
    ax.barh(plot_df["feature"], plot_df["abs_correlation"], color=colors, alpha=0.85)
    ax.set_xlabel("|Pearson Correlation with Target|", fontsize=10)
    ax.set_title(f"Feature Importance Proxy — Top {top_n} Features by |Correlation|", fontsize=11)
    ax.tick_params(axis="y", labelsize=7)
    ax.invert_yaxis()

    out_path = figures_dir / "fig_26_feature_importance_correlation.png"
    _savefig(fig, out_path, dpi=dpi)
    logger.info("Saved → %s", out_path)


def _save_enriched_sample(df: pd.DataFrame, tables_dir: Path, logger: Any) -> None:
    """Save first 1 000 rows of the enriched dataset to tables/09_enriched_dataset_sample.csv."""
    out_path = tables_dir / "09_enriched_dataset_sample.csv"
    df.head(1000).to_csv(out_path, index=False)
    logger.info("Saved enriched dataset sample (1 000 rows) → %s", out_path)


def _save_feature_summary(
    df: pd.DataFrame,
    engineered_cols: list[str],
    target: str,
    stats_dir: Path,
    logger: Any,
) -> None:
    """
    Save JSON summary: list of created features with their correlation to target.
    → stats/21_feature_engineering_summary.json
    """
    summary: list[dict] = []
    cols_present = [c for c in engineered_cols if c in df.columns and c != target]
    target_series = df[target] if target in df.columns else None

    for col in cols_present:
        entry: dict[str, Any] = {"feature": col}
        if target_series is not None and pd.api.types.is_numeric_dtype(df[col]):
            valid = df[[col, target]].dropna()
            if len(valid) > 1:
                entry["correlation_with_target"] = float(
                    valid[col].corr(valid[target])
                )
            else:
                entry["correlation_with_target"] = None
        else:
            entry["correlation_with_target"] = None
        summary.append(entry)

    out_path = stats_dir / "21_feature_engineering_summary.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump({"n_features_created": len(summary), "features": summary}, fh, indent=2)
    logger.info("Saved feature engineering summary → %s", out_path)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(config: dict, base_dir: Path) -> pd.DataFrame:
    """
    Main entry point for the feature engineering EDA module.

    Args:
        config:   Full config dict loaded from config/eda.yaml.
        base_dir: Repository root (parent of eda/, config/, outputs/).

    Returns:
        Enriched DataFrame with all engineered features appended.
    """
    _bootstrap_src_path(base_dir)

    from utils.logger import get_logger  # noqa: PLC0415

    logging_cfg = config.get("logging", {})
    # Resolve log file relative to base_dir
    log_file_rel = logging_cfg.get("log_file", "outputs/eda.log")
    logging_cfg = {**logging_cfg, "log_file": str(base_dir / log_file_rel)}
    logger = get_logger("eda.feature_engineering", logging_cfg)

    logger.info("=== Feature Engineering EDA — START ===")

    # --- paths & output dirs ---------------------------------------------------
    paths_cfg = config.get("paths", {})
    data_path = base_dir / paths_cfg.get("input_data", "data/processed/house_price.parquet")
    dirs = _ensure_output_dirs(base_dir, config)

    # --- config sections -------------------------------------------------------
    fe_cfg    = config.get("feature_engineering", {})
    vis_cfg   = config.get("visualizations", {})
    schema    = config.get("schema", {})
    target    = schema.get("target", "median_house_value")
    lat_col   = schema.get("geo_features", {}).get("latitude", "latitude")
    lon_col   = schema.get("geo_features", {}).get("longitude", "longitude")

    # --- load data -------------------------------------------------------------
    try:
        df = _load_data(data_path, logger)
    except FileNotFoundError as exc:
        logger.error("Data loading failed: %s", exc)
        raise

    # Track which new columns we create
    original_cols = set(df.columns)

    # --- ratio features --------------------------------------------------------
    try:
        ratio_cfg = fe_cfg.get("ratio_features", {})
        df = _build_ratio_features(df, ratio_cfg, logger)
    except Exception as exc:
        logger.error("Ratio feature creation failed: %s", exc, exc_info=True)

    # --- log transformations ---------------------------------------------------
    try:
        log_feats = fe_cfg.get("log_features", [])
        df = _build_log_features(df, log_feats, logger)
    except Exception as exc:
        logger.error("Log feature creation failed: %s", exc, exc_info=True)

    # --- geographic distance features ------------------------------------------
    try:
        ref_points = fe_cfg.get("geo_reference_points", {})
        df = _build_geo_distance_features(df, ref_points, lat_col, lon_col, logger)
    except Exception as exc:
        logger.error("Geographic distance features failed: %s", exc, exc_info=True)

    # --- polynomial features ---------------------------------------------------
    try:
        poly_cfg = fe_cfg.get("polynomial", {})
        df = _build_polynomial_features(df, poly_cfg, logger)
    except Exception as exc:
        logger.error("Polynomial feature creation failed: %s", exc, exc_info=True)

    # --- categorical encoding --------------------------------------------------
    try:
        df = _build_categorical_encoding(df, logger)
    except Exception as exc:
        logger.error("Categorical encoding failed: %s", exc, exc_info=True)

    # Collect all engineered column names (new columns only)
    engineered_cols = [c for c in df.columns if c not in original_cols]
    logger.info("Total engineered features: %d", len(engineered_cols))

    # --- correlation analysis --------------------------------------------------
    try:
        corr_df = _correlation_analysis(df, target, dirs["stats"], logger)
    except Exception as exc:
        logger.error("Correlation analysis failed: %s", exc, exc_info=True)
        corr_df = pd.DataFrame()

    # --- plots -----------------------------------------------------------------
    try:
        _plot_engineered_distributions(df, engineered_cols, dirs["figures"], vis_cfg, logger)
    except Exception as exc:
        logger.error("Engineered distribution plot failed: %s", exc, exc_info=True)

    try:
        _plot_log_vs_raw(df, log_feats, dirs["figures"], vis_cfg, logger)
    except Exception as exc:
        logger.error("Log vs raw plot failed: %s", exc, exc_info=True)

    try:
        _plot_distance_map(df, ref_points, dirs["figures"], vis_cfg, logger)
    except Exception as exc:
        logger.error("Distance map plot failed: %s", exc, exc_info=True)

    try:
        _plot_feature_importance(corr_df, dirs["figures"], vis_cfg, logger)
    except Exception as exc:
        logger.error("Feature importance plot failed: %s", exc, exc_info=True)

    # --- save outputs ----------------------------------------------------------
    try:
        _save_enriched_sample(df, dirs["tables"], logger)
    except Exception as exc:
        logger.error("Saving enriched sample failed: %s", exc, exc_info=True)

    try:
        _save_feature_summary(df, engineered_cols, target, dirs["stats"], logger)
    except Exception as exc:
        logger.error("Saving feature summary failed: %s", exc, exc_info=True)

    logger.info("=== Feature Engineering EDA — END (enriched shape: %s) ===", df.shape)
    return df


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml

    # Resolve paths relative to this file's location
    _this_dir  = Path(__file__).resolve().parent
    _base_dir  = _this_dir.parent
    _cfg_path  = _base_dir / "config" / "eda.yaml"

    if not _cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {_cfg_path}")

    with _cfg_path.open("r", encoding="utf-8") as _fh:
        _config = yaml.safe_load(_fh)

    result_df = run(_config, _base_dir)
    print(f"Done. Enriched dataset shape: {result_df.shape}")
