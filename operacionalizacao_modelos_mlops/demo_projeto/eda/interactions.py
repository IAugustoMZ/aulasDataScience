"""
interactions.py — Interaction-effect EDA for the California Housing dataset.

Produces 7 figures (fig_16 through fig_22) analysing 2-way and 3-way feature
interactions against median_house_value. Also writes two CSV statistics files:
    outputs/stats/18_interaction_2way_means.csv
    outputs/stats/19_interaction_3way_means.csv

All parameters are driven by config/eda.yaml; nothing is hardcoded.

Entry point:
    run(config, base_dir) -> dict
        Returns a dict with keys 'figures' (list of paths) and 'stats' (dict of DataFrames).
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — must come before any other matplotlib import

import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Logger bootstrap (uses the existing logger utility)
# ---------------------------------------------------------------------------
_BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE / "src"))

from utils.logger import get_logger  # noqa: E402


def _bootstrap_logger(config: dict) -> logging.Logger:
    """Return a logger using the project logging config, falling back to defaults."""
    logging_cfg = config.get("logging", {
        "level": "INFO",
        "format": "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
        "log_to_file": False,
    })
    return get_logger("eda.interactions", logging_cfg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_data(config: dict, base_dir: Path) -> pd.DataFrame:
    """Load the processed parquet file, resolving the path against base_dir."""
    rel_path = config["paths"]["input_data"]
    data_path = base_dir / rel_path
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. "
            "Run the ingestion pipeline first (main.py)."
        )
    return pd.read_parquet(data_path)


def _ensure_dirs(config: dict, base_dir: Path) -> tuple[Path, Path]:
    """Create output figures and stats directories; return both Paths."""
    figures_dir = base_dir / config["paths"]["figures_dir"]
    stats_dir = base_dir / config["paths"]["stats_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, stats_dir


def _fig_params(config: dict) -> dict:
    """Extract visualisation parameters from config."""
    vis = config.get("visualizations", {})
    return {
        "dpi": vis.get("figure_dpi", 120),
        "size_default": tuple(vis.get("figure_size_default", [10, 6])),
        "size_large": tuple(vis.get("figure_size_large", [14, 10])),
        "palette": vis.get("palette", "husl"),
        "style": vis.get("style", "seaborn-v0_8-whitegrid"),
    }


def _save(fig: plt.Figure, path: Path, dpi: int, logger: logging.Logger) -> str:
    """Save figure, close it, and return path as string."""
    try:
        plt.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved figure: %s", path.name)
    except Exception as exc:
        logger.error("Failed to save figure '%s': %s", path.name, exc)
        raise
    finally:
        plt.close("all")
    return str(path)


def _engineer_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Compute derived features required by interaction plots.

    Adds:
        rooms_per_household, population_per_household,
        income_category, age_category, house_value_quartile
    """
    df = df.copy()
    fe = config.get("feature_engineering", {}).get("ratio_features", {})

    # Ratio features — use config definitions where available
    for feat_name, feat_cfg in fe.items():
        num = feat_cfg["numerator"]
        den = feat_cfg["denominator"]
        if num in df.columns and den in df.columns:
            df[feat_name] = df[num] / df[den].replace(0, np.nan)

    # Ensure the two key ratio columns exist even if not in config
    if "rooms_per_household" not in df.columns:
        df["rooms_per_household"] = df["total_rooms"] / df["households"].replace(0, np.nan)
    if "population_per_household" not in df.columns:
        df["population_per_household"] = df["population"] / df["households"].replace(0, np.nan)

    # Income category — use explicit cuts from config (pivot_tables.bins.income)
    income_bins_cfg = config.get("pivot_tables", {}).get("bins", {}).get("income", {})
    if income_bins_cfg:
        income_cuts = income_bins_cfg["cuts"]
        income_labels = income_bins_cfg["labels"]
        df["income_category"] = pd.cut(
            df["median_income"],
            bins=income_cuts,
            labels=income_labels,
            right=True,
            include_lowest=True,
        )
    else:
        df["income_category"] = pd.qcut(
            df["median_income"],
            q=config.get("interactions", {}).get("interaction_bins", 5),
            duplicates="drop",
        )

    # Age category — use explicit cuts from config (pivot_tables.bins.age)
    age_bins_cfg = config.get("pivot_tables", {}).get("bins", {}).get("age", {})
    if age_bins_cfg:
        age_cuts = age_bins_cfg["cuts"]
        age_labels = age_bins_cfg["labels"]
        df["age_category"] = pd.cut(
            df["housing_median_age"],
            bins=age_cuts,
            labels=age_labels,
            right=True,
            include_lowest=True,
        )
    else:
        df["age_category"] = pd.qcut(
            df["housing_median_age"],
            q=config.get("interactions", {}).get("interaction_bins", 5),
            duplicates="drop",
        )

    # House value quartile (Q1–Q4)
    df["house_value_quartile"] = pd.qcut(
        df["median_house_value"], q=4,
        labels=["Q1 (<25%)", "Q2 (25-50%)", "Q3 (50-75%)", "Q4 (>75%)"],
        duplicates="drop",
    )

    return df


def _qcut_labels(series: pd.Series, n_bins: int) -> pd.Series:
    """Bin a continuous series into n_bins quantile buckets with readable labels."""
    try:
        return pd.qcut(series, q=n_bins, duplicates="drop")
    except Exception:
        return pd.cut(series, bins=n_bins)


# ---------------------------------------------------------------------------
# Individual figure functions
# ---------------------------------------------------------------------------

def _fig16_interaction_income_x_proximity(
    df: pd.DataFrame, config: dict, params: dict, figures_dir: Path, logger: logging.Logger
) -> str:
    """Grouped bar chart: mean house value per income_category × ocean_proximity."""
    target = "median_house_value"
    cat_col = "ocean_proximity"
    income_cat = "income_category"

    fig, ax = plt.subplots(figsize=params["size_large"])
    try:
        grouped = (
            df.groupby([income_cat, cat_col], observed=True)[target]
            .mean()
            .reset_index()
            .rename(columns={target: "mean_house_value"})
        )
        categories = sorted(df[cat_col].dropna().unique())
        income_levels = df[income_cat].cat.categories.tolist() if hasattr(df[income_cat], "cat") \
            else sorted(df[income_cat].dropna().unique())

        x = np.arange(len(income_levels))
        width = 0.15
        palette = sns.color_palette(params["palette"], len(categories))

        for i, cat in enumerate(categories):
            vals = []
            for inc in income_levels:
                mask = (grouped[income_cat] == inc) & (grouped[cat_col] == cat)
                vals.append(grouped.loc[mask, "mean_house_value"].values[0]
                            if mask.any() else np.nan)
            offset = (i - len(categories) / 2) * width
            ax.bar(x + offset, vals, width=width, label=cat,
                   color=palette[i], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(income_levels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Mean House Value ($)", fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e3:.0f}k"))
        ax.set_title("2-Way Interaction: Income Category × Ocean Proximity\n→ Mean House Value",
                     fontsize=13)
        ax.legend(title=cat_col, fontsize=9, title_fontsize=10)
    except Exception as exc:
        logger.error("Error building fig_16: %s", exc)
        raise

    return _save(fig, figures_dir / "fig_16_interaction_income_x_proximity.png",
                 params["dpi"], logger)


def _fig17_interaction_income_x_age(
    df: pd.DataFrame, config: dict, params: dict, figures_dir: Path, logger: logging.Logger
) -> str:
    """2D heatmap: income quintile bins × age quintile bins → mean house value."""
    target = "median_house_value"
    n_bins = config.get("interactions", {}).get("interaction_bins", 5)

    fig, ax = plt.subplots(figsize=params["size_default"])
    try:
        tmp = df[["median_income", "housing_median_age", target]].dropna().copy()
        tmp["inc_bin"] = _qcut_labels(tmp["median_income"], n_bins)
        tmp["age_bin"] = _qcut_labels(tmp["housing_median_age"], n_bins)

        pivot = tmp.pivot_table(values=target, index="age_bin", columns="inc_bin",
                                aggfunc="mean")
        sns.heatmap(
            pivot / 1000, ax=ax, cmap="YlOrRd", annot=True, fmt=".0f",
            linewidths=0.4, cbar_kws={"label": "Mean House Value (×$1k)"},
        )
        ax.set_xlabel("Median Income (quintile bins)", fontsize=10)
        ax.set_ylabel("Housing Median Age (quintile bins)", fontsize=10)
        ax.set_title("2-Way Interaction: Income × Age Quintiles\n→ Mean House Value ($k)",
                     fontsize=12)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)
    except Exception as exc:
        logger.error("Error building fig_17: %s", exc)
        raise

    return _save(fig, figures_dir / "fig_17_interaction_income_x_age.png",
                 params["dpi"], logger)


def _fig18_interaction_income_x_rooms(
    df: pd.DataFrame, config: dict, params: dict, figures_dir: Path, logger: logging.Logger
) -> str:
    """2D heatmap: income quintile bins × rooms_per_household quintile bins → mean house value."""
    target = "median_house_value"
    n_bins = config.get("interactions", {}).get("interaction_bins", 5)

    fig, ax = plt.subplots(figsize=params["size_default"])
    try:
        tmp = df[["median_income", "rooms_per_household", target]].dropna().copy()
        tmp["inc_bin"] = _qcut_labels(tmp["median_income"], n_bins)
        tmp["rooms_bin"] = _qcut_labels(tmp["rooms_per_household"], n_bins)

        pivot = tmp.pivot_table(values=target, index="rooms_bin", columns="inc_bin",
                                aggfunc="mean")
        sns.heatmap(
            pivot / 1000, ax=ax, cmap="YlOrRd", annot=True, fmt=".0f",
            linewidths=0.4, cbar_kws={"label": "Mean House Value (×$1k)"},
        )
        ax.set_xlabel("Median Income (quintile bins)", fontsize=10)
        ax.set_ylabel("Rooms per Household (quintile bins)", fontsize=10)
        ax.set_title("2-Way Interaction: Income × Rooms/HH Quintiles\n→ Mean House Value ($k)",
                     fontsize=12)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)
    except Exception as exc:
        logger.error("Error building fig_18: %s", exc)
        raise

    return _save(fig, figures_dir / "fig_18_interaction_income_x_rooms.png",
                 params["dpi"], logger)


def _fig19_interaction_age_x_proximity(
    df: pd.DataFrame, config: dict, params: dict, figures_dir: Path, logger: logging.Logger
) -> str:
    """Line plot: mean house value by housing_median_age bins for each ocean_proximity group."""
    target = "median_house_value"
    cat_col = "ocean_proximity"
    n_bins = config.get("interactions", {}).get("interaction_bins", 5)

    fig, ax = plt.subplots(figsize=params["size_default"])
    try:
        tmp = df[["housing_median_age", cat_col, target]].dropna().copy()
        tmp["age_bin"] = _qcut_labels(tmp["housing_median_age"], n_bins)
        tmp["age_mid"] = tmp["age_bin"].apply(
            lambda x: (x.left + x.right) / 2 if hasattr(x, "left") else np.nan
        )

        categories = sorted(df[cat_col].dropna().unique())
        palette = sns.color_palette(params["palette"], len(categories))

        for cat, color in zip(categories, palette):
            sub = tmp[tmp[cat_col] == cat]
            line_data = (
                sub.groupby("age_mid", observed=True)[target]
                .mean()
                .reset_index()
                .sort_values("age_mid")
            )
            ax.plot(line_data["age_mid"], line_data[target], marker="o",
                    color=color, linewidth=2, markersize=5, label=cat)

        ax.set_xlabel("Housing Median Age (bin midpoint, years)", fontsize=11)
        ax.set_ylabel("Mean House Value ($)", fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e3:.0f}k"))
        ax.set_title("2-Way Interaction: Age × Ocean Proximity\n→ Mean House Value",
                     fontsize=13)
        ax.legend(title=cat_col, fontsize=9, title_fontsize=10)
    except Exception as exc:
        logger.error("Error building fig_19: %s", exc)
        raise

    return _save(fig, figures_dir / "fig_19_interaction_age_x_proximity.png",
                 params["dpi"], logger)


def _fig20_interaction_pop_density_x_income(
    df: pd.DataFrame, config: dict, params: dict, figures_dir: Path, logger: logging.Logger
) -> str:
    """Scatter: population_per_household vs median_income, coloured by house value quartile, with 2D KDE."""
    target = "median_house_value"
    seed = config.get("eda", {}).get("random_seed", 42)

    fig, ax = plt.subplots(figsize=params["size_default"])
    try:
        cols = ["population_per_household", "median_income", target, "house_value_quartile"]
        sample = df[cols].dropna().sample(n=min(5000, len(df)), random_state=seed)

        quartiles = ["Q1 (<25%)", "Q2 (25-50%)", "Q3 (50-75%)", "Q4 (>75%)"]
        palette = sns.color_palette("RdYlGn", len(quartiles))
        color_map = dict(zip(quartiles, palette))

        for qrt in quartiles:
            sub = sample[sample["house_value_quartile"] == qrt]
            if len(sub) < 2:
                continue
            ax.scatter(
                sub["population_per_household"], sub["median_income"],
                color=color_map[qrt], alpha=0.35, s=14, label=qrt,
            )

        # 2D KDE overlay using the full sample
        x = sample["population_per_household"].clip(upper=sample["population_per_household"].quantile(0.99))
        y = sample["median_income"]
        sns.kdeplot(x=x, y=y, ax=ax, levels=6, color="darkblue",
                    linewidths=1.0, alpha=0.6)

        ax.set_xlim(left=0, right=x.quantile(0.99) * 1.05)
        ax.set_xlabel("Population per Household", fontsize=11)
        ax.set_ylabel("Median Income (×$10k)", fontsize=11)
        ax.set_title(
            "2-Way Interaction: Population Density × Income\n(coloured by House Value Quartile)",
            fontsize=12,
        )
        ax.legend(title="House Value Quartile", fontsize=9, title_fontsize=10)
    except Exception as exc:
        logger.error("Error building fig_20: %s", exc)
        raise

    return _save(fig, figures_dir / "fig_20_interaction_pop_density_x_income.png",
                 params["dpi"], logger)


def _fig21_3way_income_proximity_age(
    df: pd.DataFrame, config: dict, params: dict, figures_dir: Path, logger: logging.Logger
) -> str:
    """FacetGrid: income_category (facets) × ocean_proximity (x) → box plots, hued by age_category."""
    target = "median_house_value"
    cat_col = "ocean_proximity"

    try:
        plot_df = df[["income_category", cat_col, "age_category", target]].dropna()

        income_levels = (
            df["income_category"].cat.categories.tolist()
            if hasattr(df["income_category"], "cat")
            else sorted(df["income_category"].dropna().unique())
        )
        # Use only first 3 income levels to keep the facet grid readable
        income_levels_plot = income_levels[:3]
        plot_df = plot_df[plot_df["income_category"].isin(income_levels_plot)]

        # Convert categoricals to strings for seaborn compatibility
        plot_df = plot_df.copy()
        plot_df["income_category"] = plot_df["income_category"].astype(str)
        plot_df["age_category"] = plot_df["age_category"].astype(str)

        age_levels = sorted(plot_df["age_category"].dropna().unique())
        palette = sns.color_palette(params["palette"], len(age_levels))

        g = sns.FacetGrid(
            plot_df,
            col="income_category",
            col_order=[str(il) for il in income_levels_plot],
            height=5, aspect=1.1, sharey=True,
        )
        g.map_dataframe(
            sns.boxplot,
            x=cat_col,
            y=target,
            hue="age_category",
            hue_order=age_levels,
            palette=palette,
            linewidth=0.8,
            flierprops=dict(marker=".", markersize=2, alpha=0.2),
        )
        g.add_legend(title="Age Category", fontsize=8, title_fontsize=9)

        for ax in g.axes.flat:
            ax.set_xlabel("Ocean Proximity", fontsize=9)
            ax.set_ylabel("House Value ($)", fontsize=9)
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"${v/1e3:.0f}k")
            )
            plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=7)

        g.figure.suptitle(
            "3-Way Interaction: Income × Proximity × Age → House Value",
            fontsize=12, y=1.02,
        )
        plt.tight_layout()
        out_path = figures_dir / "fig_21_3way_income_proximity_age.png"
        g.figure.savefig(out_path, dpi=params["dpi"], bbox_inches="tight")
        logger.info("Saved figure: %s", out_path.name)
    except Exception as exc:
        logger.error("Error building fig_21: %s", exc)
        raise
    finally:
        plt.close("all")

    return str(out_path)


def _fig22_3way_income_rooms_proximity(
    df: pd.DataFrame, config: dict, params: dict, figures_dir: Path, logger: logging.Logger
) -> str:
    """FacetGrid by ocean_proximity: scatter median_income vs rooms_per_household coloured by house value."""
    target = "median_house_value"
    cat_col = "ocean_proximity"
    seed = config.get("eda", {}).get("random_seed", 42)

    try:
        cols = ["median_income", "rooms_per_household", target, cat_col]
        sample = df[cols].dropna().sample(n=min(5000, len(df)), random_state=seed)

        # Clip extreme rooms values for readability
        sample = sample[sample["rooms_per_household"] < sample["rooms_per_household"].quantile(0.99)]

        categories = sorted(df[cat_col].dropna().unique())
        g = sns.FacetGrid(
            sample, col=cat_col, col_order=categories,
            col_wrap=3, height=4, aspect=1.1,
        )

        def _scatter_with_colorbar(x, y, **kwargs):
            data = kwargs.pop("data", None)
            color_vals = data[target] if data is not None else None
            sc = plt.scatter(x, y, c=color_vals, cmap="YlOrRd", alpha=0.4,
                             s=12, **{k: v for k, v in kwargs.items()
                                      if k not in ("color", "label")})

        g.map_dataframe(_scatter_with_colorbar, "median_income", "rooms_per_household")

        for ax in g.axes.flat:
            ax.set_xlabel("Median Income (×$10k)", fontsize=9)
            ax.set_ylabel("Rooms per Household", fontsize=9)

        # Add a shared colorbar
        sm = plt.cm.ScalarMappable(
            cmap="YlOrRd",
            norm=plt.Normalize(vmin=sample[target].min(), vmax=sample[target].max()),
        )
        sm.set_array([])
        cbar = g.figure.colorbar(sm, ax=g.axes.ravel().tolist(), shrink=0.6, pad=0.02)
        cbar.set_label("House Value ($)", fontsize=9)

        g.figure.suptitle(
            "3-Way Interaction: Income × Rooms/HH by Ocean Proximity → House Value",
            fontsize=11, y=1.02,
        )
        plt.tight_layout()
        out_path = figures_dir / "fig_22_3way_income_rooms_proximity.png"
        g.figure.savefig(out_path, dpi=params["dpi"], bbox_inches="tight")
        logger.info("Saved figure: %s", out_path.name)
    except Exception as exc:
        logger.error("Error building fig_22: %s", exc)
        raise
    finally:
        plt.close("all")

    return str(out_path)


# ---------------------------------------------------------------------------
# Statistics export
# ---------------------------------------------------------------------------

def _compute_2way_means(df: pd.DataFrame, config: dict, logger: logging.Logger) -> pd.DataFrame:
    """Compute mean house value for all 2-way combinations listed in config."""
    target = "median_house_value"
    two_way_pairs = config.get("interactions", {}).get("two_way", [])
    n_bins = config.get("interactions", {}).get("interaction_bins", 5)

    rows = []
    for pair in two_way_pairs:
        feat_a, feat_b = pair
        if feat_a not in df.columns or feat_b not in df.columns:
            logger.warning("Skipping 2-way pair (%s, %s) — column missing.", feat_a, feat_b)
            continue

        tmp = df[[feat_a, feat_b, target]].dropna().copy()

        # Bin continuous features if not already categorical
        for feat in [feat_a, feat_b]:
            if not pd.api.types.is_categorical_dtype(tmp[feat]) and \
               not pd.api.types.is_object_dtype(tmp[feat]):
                try:
                    tmp[feat] = _qcut_labels(tmp[feat], n_bins).astype(str)
                except Exception:
                    tmp[feat] = tmp[feat].astype(str)
            else:
                tmp[feat] = tmp[feat].astype(str)

        group = tmp.groupby([feat_a, feat_b], observed=True)[target].agg(
            ["mean", "median", "std", "count"]
        ).reset_index()
        group.columns = [feat_a, feat_b, "mean_house_value",
                         "median_house_value", "std_house_value", "count"]
        group["interaction"] = f"{feat_a}_x_{feat_b}"
        rows.append(group)

    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()


def _compute_3way_means(df: pd.DataFrame, config: dict, logger: logging.Logger) -> pd.DataFrame:
    """Compute mean house value for all 3-way combinations listed in config."""
    target = "median_house_value"
    three_way_triples = config.get("interactions", {}).get("three_way", [])
    n_bins = config.get("interactions", {}).get("interaction_bins", 5)

    rows = []
    for triple in three_way_triples:
        feat_a, feat_b, feat_c = triple
        missing = [f for f in [feat_a, feat_b, feat_c] if f not in df.columns]
        if missing:
            logger.warning("Skipping 3-way triple — missing columns: %s", missing)
            continue

        tmp = df[[feat_a, feat_b, feat_c, target]].dropna().copy()

        for feat in [feat_a, feat_b, feat_c]:
            if not pd.api.types.is_categorical_dtype(tmp[feat]) and \
               not pd.api.types.is_object_dtype(tmp[feat]):
                try:
                    tmp[feat] = _qcut_labels(tmp[feat], n_bins).astype(str)
                except Exception:
                    tmp[feat] = tmp[feat].astype(str)
            else:
                tmp[feat] = tmp[feat].astype(str)

        group = tmp.groupby([feat_a, feat_b, feat_c], observed=True)[target].agg(
            ["mean", "median", "std", "count"]
        ).reset_index()
        group.columns = [feat_a, feat_b, feat_c, "mean_house_value",
                         "median_house_value", "std_house_value", "count"]
        group["interaction"] = f"{feat_a}_x_{feat_b}_x_{feat_c}"
        rows.append(group)

    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()


def _save_stats(
    df_2way: pd.DataFrame,
    df_3way: pd.DataFrame,
    stats_dir: Path,
    logger: logging.Logger,
) -> dict[str, str]:
    """Write statistics CSV files; return dict of {label: path}."""
    saved = {}
    for label, df_stat, filename in [
        ("2way_means", df_2way, "18_interaction_2way_means.csv"),
        ("3way_means", df_3way, "19_interaction_3way_means.csv"),
    ]:
        if df_stat.empty:
            logger.warning("Stats table '%s' is empty — skipping write.", filename)
            continue
        out_path = stats_dir / filename
        try:
            df_stat.to_csv(out_path, index=False, encoding="utf-8")
            logger.info("Saved stats: %s", out_path.name)
            saved[label] = str(out_path)
        except Exception as exc:
            logger.error("Failed to save stats '%s': %s", filename, exc)
    return saved


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(config: dict, base_dir: Path) -> dict[str, Any]:
    """
    Execute all interaction EDA figures and compute interaction statistics.

    Args:
        config:   Loaded configuration dict (from config/eda.yaml).
        base_dir: Root directory of the project (paths resolved from here).

    Returns:
        Dict with keys:
            'figures' -> list[str]   — saved PNG paths
            'stats'   -> dict        — {label: path} for CSV files
            'dataframes' -> dict     — {label: DataFrame} with computed stats
    """
    logger = _bootstrap_logger(config)
    logger.info("=== EDA Interactions — START ===")

    try:
        style = config.get("visualizations", {}).get("style", "seaborn-v0_8-whitegrid")
        plt.style.use(style)
    except Exception:
        logger.warning("Could not apply style '%s'; using default.", style)

    try:
        df_raw = _load_data(config, base_dir)
        logger.info("Loaded dataset: %d rows × %d columns", *df_raw.shape)
    except FileNotFoundError as exc:
        logger.error("Dataset load failed: %s", exc)
        raise

    figures_dir, stats_dir = _ensure_dirs(config, base_dir)
    params = _fig_params(config)

    # Engineer all derived features up front
    try:
        df = _engineer_features(df_raw, config)
        logger.info("Feature engineering complete. Columns added: %s",
                    [c for c in df.columns if c not in df_raw.columns])
    except Exception as exc:
        logger.error("Feature engineering failed: %s", exc)
        raise

    # ------------------------------------------------------------------ figures
    figure_fns = [
        _fig16_interaction_income_x_proximity,
        _fig17_interaction_income_x_age,
        _fig18_interaction_income_x_rooms,
        _fig19_interaction_age_x_proximity,
        _fig20_interaction_pop_density_x_income,
        _fig21_3way_income_proximity_age,
        _fig22_3way_income_rooms_proximity,
    ]

    saved_paths: list[str] = []
    for fn in figure_fns:
        try:
            path = fn(df, config, params, figures_dir, logger)
            saved_paths.append(path)
        except Exception as exc:
            logger.error("Figure function '%s' failed: %s", fn.__name__, exc)

    # ------------------------------------------------------------------ stats
    df_2way = pd.DataFrame()
    df_3way = pd.DataFrame()
    saved_stats: dict[str, str] = {}

    try:
        df_2way = _compute_2way_means(df, config, logger)
        df_3way = _compute_3way_means(df, config, logger)
        saved_stats = _save_stats(df_2way, df_3way, stats_dir, logger)
    except Exception as exc:
        logger.error("Statistics computation failed: %s", exc)

    logger.info(
        "=== EDA Interactions — DONE (%d/%d figures, %d stats files) ===",
        len(saved_paths), len(figure_fns), len(saved_stats),
    )

    return {
        "figures": saved_paths,
        "stats": saved_stats,
        "dataframes": {
            "two_way_means": df_2way,
            "three_way_means": df_3way,
        },
    }


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml

    _base_dir = Path(__file__).resolve().parent.parent
    _config_path = _base_dir / "config" / "eda.yaml"

    if not _config_path.exists():
        print(f"ERROR: Config not found at {_config_path}")
        sys.exit(1)

    with _config_path.open("r", encoding="utf-8") as _fh:
        _config = yaml.safe_load(_fh)

    _result = run(_config, _base_dir)

    print(f"\nSaved {len(_result['figures'])} figure(s):")
    for _p in _result["figures"]:
        print(f"  {_p}")

    print(f"\nSaved {len(_result['stats'])} stats file(s):")
    for _label, _p in _result["stats"].items():
        print(f"  [{_label}] {_p}")
