"""
visualizations.py — Core EDA visualizations for the California Housing dataset.

Produces 15 figures covering target distribution, feature distributions,
correlations, geographic maps, and categorical breakdowns. All parameters
are driven by config/eda.yaml; nothing is hardcoded in this module.

Entry point:
    run(config, base_dir) -> list[str]  (list of saved PNG paths)
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
from scipy import stats
from scipy.stats import gaussian_kde
from statsmodels.nonparametric.smoothers_lowess import lowess

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
    return get_logger("eda.visualizations", logging_cfg)


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
    df = pd.read_parquet(data_path)
    return df


def _ensure_dirs(config: dict, base_dir: Path) -> Path:
    """Create output figure directory and return its Path."""
    figures_dir = base_dir / config["paths"]["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def _fig_params(config: dict) -> dict:
    """Extract visualisation parameters from config into a convenient dict."""
    vis = config.get("visualizations", {})
    return {
        "dpi": vis.get("figure_dpi", 120),
        "fmt": vis.get("figure_format", "png"),
        "size_default": tuple(vis.get("figure_size_default", [10, 6])),
        "size_large": tuple(vis.get("figure_size_large", [14, 10])),
        "size_map": tuple(vis.get("figure_size_map", [12, 9])),
        "style": vis.get("style", "seaborn-v0_8-whitegrid"),
        "palette": vis.get("palette", "husl"),
        "hist_bins": vis.get("histogram", {}).get("bins", 50),
        "corr_method": vis.get("correlation", {}).get("method", "pearson"),
        "corr_annot": vis.get("correlation", {}).get("annot", True),
        "corr_fmt": vis.get("correlation", {}).get("fmt", ".2f"),
        "geo_alpha": vis.get("geo_map", {}).get("alpha", 0.4),
        "geo_size_feat": vis.get("geo_map", {}).get("size_feature", "population"),
        "geo_color_feat": vis.get("geo_map", {}).get("color_feature", "median_house_value"),
        "geo_cmap": vis.get("geo_map", {}).get("colormap", "YlOrRd"),
        "geo_size_scale": vis.get("geo_map", {}).get("size_scale", 0.001),
    }


def _save(fig: plt.Figure, path: Path, dpi: int, logger: logging.Logger) -> str:
    """Save figure, close it, and return the path as string."""
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


# ---------------------------------------------------------------------------
# Individual figure functions
# ---------------------------------------------------------------------------

def _fig01_target_distribution(
    df: pd.DataFrame, config: dict, params: dict, figures_dir: Path, logger: logging.Logger
) -> str:
    """Histogram + KDE of median_house_value with $500,001 cap line."""
    target = config["schema"]["target"]
    cap = config["schema"]["value_bounds"][target]["cap_value"]

    fig, ax = plt.subplots(figsize=params["size_default"])
    try:
        data = df[target].dropna()
        ax.hist(data, bins=params["hist_bins"], density=True, alpha=0.6,
                color="steelblue", label="Frequency")

        kde_vals = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 500)
        ax.plot(x_range, kde_vals(x_range), color="navy", linewidth=2, label="KDE")

        ax.axvline(cap, color="crimson", linestyle="--", linewidth=1.8,
                   label=f"Cap at ${cap:,.0f}")

        skew = float(data.skew())
        ax.set_title(
            f"Target Distribution — median_house_value\n(skewness = {skew:.3f})",
            fontsize=13,
        )
        ax.set_xlabel("Median House Value ($)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
        ax.legend(fontsize=10)
    except Exception as exc:
        logger.error("Error building fig_01: %s", exc)
        raise

    return _save(fig, figures_dir / "fig_01_target_distribution.png", params["dpi"], logger)


def _fig02_log_target_distribution(
    df: pd.DataFrame, config: dict, params: dict, figures_dir: Path, logger: logging.Logger
) -> str:
    """Histogram + KDE of log(median_house_value) showing normality improvement."""
    target = config["schema"]["target"]
    data = df[target].dropna()
    log_data = np.log1p(data)

    fig, axes = plt.subplots(1, 2, figsize=params["size_default"])
    try:
        for ax, values, label, color in zip(
            axes,
            [data, log_data],
            ["Original", "Log-Transformed (log1p)"],
            ["steelblue", "seagreen"],
        ):
            ax.hist(values, bins=params["hist_bins"], density=True, alpha=0.6, color=color)
            kde_vals = gaussian_kde(values)
            x_range = np.linspace(values.min(), values.max(), 500)
            ax.plot(x_range, kde_vals(x_range), color="navy", linewidth=2)
            skew = float(values.skew())
            ax.set_title(f"{label}\nskewness = {skew:.3f}", fontsize=11)
            ax.set_xlabel(label, fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
    except Exception as exc:
        logger.error("Error building fig_02: %s", exc)
        raise

    fig.suptitle("Target Distribution: Original vs Log-Transformed", fontsize=13, y=1.02)
    return _save(fig, figures_dir / "fig_02_log_target_distribution.png", params["dpi"], logger)


def _fig03_feature_distributions(
    df: pd.DataFrame, config: dict, params: dict, figures_dir: Path, logger: logging.Logger
) -> str:
    """3×3 grid of histograms for all 9 numeric features."""
    numeric_cols = config["schema"]["numeric_features"] + [config["schema"]["target"]]
    # Limit to 9 for 3×3 layout
    cols_to_plot = numeric_cols[:9]
    n_cols = 3
    n_rows = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=params["size_large"])
    axes_flat = axes.flatten()

    try:
        palette = sns.color_palette(params["palette"], len(cols_to_plot))
        for i, col in enumerate(cols_to_plot):
            ax = axes_flat[i]
            data = df[col].dropna()
            ax.hist(data, bins=params["hist_bins"], color=palette[i], alpha=0.75, density=True)
            try:
                kde_vals = gaussian_kde(data)
                x_range = np.linspace(data.min(), data.max(), 300)
                ax.plot(x_range, kde_vals(x_range), color="black", linewidth=1.2)
            except Exception:
                pass  # KDE may fail on low-variance data; histogram is sufficient
            skew = float(data.skew())
            ax.set_title(f"{col}\nskew={skew:.2f}", fontsize=9)
            ax.tick_params(labelsize=7)

        # Hide unused axes if any
        for j in range(len(cols_to_plot), len(axes_flat)):
            axes_flat[j].set_visible(False)
    except Exception as exc:
        logger.error("Error building fig_03: %s", exc)
        raise

    fig.suptitle("Numeric Feature Distributions", fontsize=14, y=1.01)
    return _save(fig, figures_dir / "fig_03_feature_distributions.png", params["dpi"], logger)


def _fig04_feature_boxplots(
    df: pd.DataFrame, config: dict, params: dict, figures_dir: Path, logger: logging.Logger
) -> str:
    """Box plots of all numeric features normalised for comparison."""
    numeric_cols = config["schema"]["numeric_features"] + [config["schema"]["target"]]

    fig, ax = plt.subplots(figsize=params["size_large"])
    try:
        subset = df[numeric_cols].dropna()
        # Z-score normalise so all features share the same y-axis
        normalized = (subset - subset.mean()) / subset.std()
        normalized.boxplot(ax=ax, rot=45, notch=False,
                           boxprops=dict(color="steelblue"),
                           medianprops=dict(color="crimson", linewidth=2),
                           whiskerprops=dict(color="gray"),
                           capprops=dict(color="gray"),
                           flierprops=dict(marker=".", markersize=2, alpha=0.3))
        ax.set_title("Feature Box Plots (Z-score Normalised)", fontsize=13)
        ax.set_ylabel("Standard Deviations from Mean", fontsize=11)
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    except Exception as exc:
        logger.error("Error building fig_04: %s", exc)
        raise

    return _save(fig, figures_dir / "fig_04_feature_boxplots.png", params["dpi"], logger)


def _fig05_correlation_heatmap(
    df: pd.DataFrame, config: dict, params: dict, figures_dir: Path, logger: logging.Logger
) -> str:
    """Pearson correlation heatmap with annotations."""
    numeric_cols = config["schema"]["numeric_features"] + [config["schema"]["target"]]
    method = params["corr_method"]

    fig, ax = plt.subplots(figsize=params["size_large"])
    try:
        corr = df[numeric_cols].corr(method=method)
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(
            corr,
            ax=ax,
            annot=params["corr_annot"],
            fmt=params["corr_fmt"],
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            mask=False,  # show full matrix
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(f"Correlation Matrix ({method.capitalize()})", fontsize=13)
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        ax.tick_params(axis="y", rotation=0, labelsize=9)
    except Exception as exc:
        logger.error("Error building fig_05: %s", exc)
        raise

    return _save(fig, figures_dir / "fig_05_correlation_heatmap.png", params["dpi"], logger)


def _fig06_pairplot_key_features(
    df: pd.DataFrame, config: dict, params: dict, figures_dir: Path, logger: logging.Logger
) -> str:
    """Seaborn pairplot of key features, coloured by ocean_proximity."""
    target = config["schema"]["target"]
    cat_col = config["schema"]["categorical_features"][0]
    seed = config.get("eda", {}).get("random_seed", 42)

    try:
        sample = df.sample(n=min(5000, len(df)), random_state=seed).copy()
        sample["rooms_per_household"] = sample["total_rooms"] / sample["households"].replace(0, np.nan)
        plot_cols = ["median_income", "housing_median_age", "rooms_per_household", target]

        pg = sns.pairplot(
            sample[plot_cols + [cat_col]].dropna(),
            hue=cat_col,
            plot_kws={"alpha": 0.3, "s": 10},
            diag_kws={"fill": True},
            corner=False,
        )
        pg.figure.suptitle("Pairplot of Key Features (5 000-row sample)", y=1.01, fontsize=12)
        plt.tight_layout()
        out_path = figures_dir / "fig_06_pairplot_key_features.png"
        pg.figure.savefig(out_path, dpi=params["dpi"], bbox_inches="tight")
        logger.info("Saved figure: %s", out_path.name)
    except Exception as exc:
        logger.error("Error building fig_06: %s", exc)
        raise
    finally:
        plt.close("all")

    return str(out_path)


def _fig07_geo_scatter_map(
    df: pd.DataFrame, config: dict, params: dict, figures_dir: Path, logger: logging.Logger
) -> str:
    """Geographic scatter: bubble size = population, colour = median_house_value."""
    lat_col = config["schema"]["geo_features"]["latitude"]
    lon_col = config["schema"]["geo_features"]["longitude"]
    color_col = params["geo_color_feat"]
    size_col = params["geo_size_feat"]

    fig, ax = plt.subplots(figsize=params["size_map"])
    try:
        sizes = df[size_col] * params["geo_size_scale"]
        sizes = sizes.clip(lower=1, upper=300)

        sc = ax.scatter(
            df[lon_col], df[lat_col],
            c=df[color_col],
            s=sizes,
            cmap=params["geo_cmap"],
            alpha=params["geo_alpha"],
            linewidths=0,
        )
        cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label("Median House Value ($)", fontsize=10)
        cbar.formatter = mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k")
        cbar.update_ticks()

        ax.set_xlabel("Longitude", fontsize=11)
        ax.set_ylabel("Latitude", fontsize=11)
        ax.set_title(
            "Geographic Distribution — House Value\n(bubble size ∝ population)",
            fontsize=13,
        )
    except Exception as exc:
        logger.error("Error building fig_07: %s", exc)
        raise

    return _save(fig, figures_dir / "fig_07_geo_scatter_map.png", params["dpi"], logger)


def _fig08_geo_log_price_map(
    df: pd.DataFrame, config: dict, params: dict, figures_dir: Path, logger: logging.Logger
) -> str:
    """Geographic scatter coloured by log(median_house_value)."""
    lat_col = config["schema"]["geo_features"]["latitude"]
    lon_col = config["schema"]["geo_features"]["longitude"]
    color_col = params["geo_color_feat"]
    size_col = params["geo_size_feat"]

    fig, ax = plt.subplots(figsize=params["size_map"])
    try:
        sizes = df[size_col] * params["geo_size_scale"]
        sizes = sizes.clip(lower=1, upper=300)
        log_values = np.log1p(df[color_col])

        sc = ax.scatter(
            df[lon_col], df[lat_col],
            c=log_values,
            s=sizes,
            cmap=params["geo_cmap"],
            alpha=params["geo_alpha"],
            linewidths=0,
        )
        cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label("log(Median House Value + 1)", fontsize=10)

        ax.set_xlabel("Longitude", fontsize=11)
        ax.set_ylabel("Latitude", fontsize=11)
        ax.set_title(
            "Geographic Distribution — log(House Value)\n(bubble size ∝ population)",
            fontsize=13,
        )
    except Exception as exc:
        logger.error("Error building fig_08: %s", exc)
        raise

    return _save(fig, figures_dir / "fig_08_geo_log_price_map.png", params["dpi"], logger)


def _fig09_income_vs_price_scatter(
    df: pd.DataFrame, config: dict, params: dict, figures_dir: Path, logger: logging.Logger
) -> str:
    """Scatter median_income vs median_house_value, coloured by ocean_proximity with trend lines."""
    target = config["schema"]["target"]
    income_col = "median_income"
    cat_col = config["schema"]["categorical_features"][0]
    seed = config.get("eda", {}).get("random_seed", 42)

    fig, ax = plt.subplots(figsize=params["size_large"])
    try:
        sample = df.sample(n=min(5000, len(df)), random_state=seed)
        categories = sorted(sample[cat_col].dropna().unique())
        palette = sns.color_palette(params["palette"], len(categories))
        color_map = dict(zip(categories, palette))

        for cat in categories:
            mask = sample[cat_col] == cat
            sub = sample[mask].dropna(subset=[income_col, target])
            ax.scatter(
                sub[income_col], sub[target],
                color=color_map[cat], alpha=0.35, s=12, label=cat,
            )
            # Trend line via linear regression
            if len(sub) >= 10:
                slope, intercept, *_ = stats.linregress(sub[income_col], sub[target])
                x_vals = np.linspace(sub[income_col].min(), sub[income_col].max(), 100)
                ax.plot(x_vals, slope * x_vals + intercept, color=color_map[cat], linewidth=1.8)

        ax.set_xlabel("Median Income (×$10k)", fontsize=11)
        ax.set_ylabel("Median House Value ($)", fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
        ax.set_title("Median Income vs House Value by Ocean Proximity", fontsize=13)
        ax.legend(title=cat_col, fontsize=9, title_fontsize=10, loc="upper left")
    except Exception as exc:
        logger.error("Error building fig_09: %s", exc)
        raise

    return _save(fig, figures_dir / "fig_09_income_vs_price_scatter.png", params["dpi"], logger)


def _fig10_price_by_proximity(
    df: pd.DataFrame, config: dict, params: dict, figures_dir: Path, logger: logging.Logger
) -> str:
    """Box plots of median_house_value by ocean_proximity, sorted by median price."""
    target = config["schema"]["target"]
    cat_col = config["schema"]["categorical_features"][0]

    fig, ax = plt.subplots(figsize=params["size_default"])
    try:
        order = (
            df.groupby(cat_col)[target]
            .median()
            .sort_values()
            .index.tolist()
        )
        plot_data = [df.loc[df[cat_col] == cat, target].dropna().values for cat in order]
        ax.boxplot(plot_data, labels=order, notch=False, patch_artist=True,
                   boxprops=dict(facecolor="steelblue", alpha=0.6),
                   medianprops=dict(color="crimson", linewidth=2),
                   flierprops=dict(marker=".", markersize=2, alpha=0.3))
        ax.set_xlabel("Ocean Proximity", fontsize=11)
        ax.set_ylabel("Median House Value ($)", fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
        ax.set_title("House Value Distribution by Ocean Proximity\n(sorted by median)", fontsize=13)
        plt.xticks(rotation=20)
    except Exception as exc:
        logger.error("Error building fig_10: %s", exc)
        raise

    return _save(fig, figures_dir / "fig_10_price_by_proximity.png", params["dpi"], logger)


def _fig11_price_by_proximity_violin(
    df: pd.DataFrame, config: dict, params: dict, figures_dir: Path, logger: logging.Logger
) -> str:
    """Violin plots of median_house_value by ocean_proximity."""
    target = config["schema"]["target"]
    cat_col = config["schema"]["categorical_features"][0]

    fig, ax = plt.subplots(figsize=params["size_default"])
    try:
        order = (
            df.groupby(cat_col)[target]
            .median()
            .sort_values()
            .index.tolist()
        )
        palette = sns.color_palette(params["palette"], len(order))
        sns.violinplot(
            data=df, x=cat_col, y=target, order=order,
            palette=palette, ax=ax, inner="box", cut=0,
        )
        ax.set_xlabel("Ocean Proximity", fontsize=11)
        ax.set_ylabel("Median House Value ($)", fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
        ax.set_title("House Value Violin Plot by Ocean Proximity", fontsize=13)
        plt.xticks(rotation=20)
    except Exception as exc:
        logger.error("Error building fig_11: %s", exc)
        raise

    return _save(fig, figures_dir / "fig_11_price_by_proximity_violin.png", params["dpi"], logger)


def _fig12_age_vs_price(
    df: pd.DataFrame, config: dict, params: dict, figures_dir: Path, logger: logging.Logger
) -> str:
    """Scatter + LOWESS trend: housing_median_age vs median_house_value."""
    target = config["schema"]["target"]
    age_col = "housing_median_age"
    seed = config.get("eda", {}).get("random_seed", 42)

    fig, ax = plt.subplots(figsize=params["size_default"])
    try:
        sample = df[[age_col, target]].dropna().sample(
            n=min(5000, len(df)), random_state=seed
        )
        ax.scatter(sample[age_col], sample[target], alpha=0.2, s=8, color="steelblue")

        # LOWESS smooth
        lowess_result = lowess(sample[target], sample[age_col], frac=0.3)
        sorted_idx = np.argsort(lowess_result[:, 0])
        ax.plot(
            lowess_result[sorted_idx, 0], lowess_result[sorted_idx, 1],
            color="crimson", linewidth=2.5, label="LOWESS trend",
        )
        ax.set_xlabel("Housing Median Age (years)", fontsize=11)
        ax.set_ylabel("Median House Value ($)", fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
        ax.set_title("Housing Age vs House Value with LOWESS Trend", fontsize=13)
        ax.legend(fontsize=10)
    except Exception as exc:
        logger.error("Error building fig_12: %s", exc)
        raise

    return _save(fig, figures_dir / "fig_12_age_vs_price.png", params["dpi"], logger)


def _fig13_income_distribution_by_proximity(
    df: pd.DataFrame, config: dict, params: dict, figures_dir: Path, logger: logging.Logger
) -> str:
    """KDE plots of median_income for each ocean_proximity category (overlaid)."""
    income_col = "median_income"
    cat_col = config["schema"]["categorical_features"][0]

    fig, ax = plt.subplots(figsize=params["size_default"])
    try:
        categories = sorted(df[cat_col].dropna().unique())
        palette = sns.color_palette(params["palette"], len(categories))
        for cat, color in zip(categories, palette):
            data = df.loc[df[cat_col] == cat, income_col].dropna()
            kde_vals = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 300)
            ax.plot(x_range, kde_vals(x_range), color=color, linewidth=2, label=cat)
            ax.fill_between(x_range, kde_vals(x_range), alpha=0.1, color=color)

        ax.set_xlabel("Median Income (×$10k)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title("Median Income Distribution by Ocean Proximity (KDE)", fontsize=13)
        ax.legend(title=cat_col, fontsize=9, title_fontsize=10)
    except Exception as exc:
        logger.error("Error building fig_13: %s", exc)
        raise

    return _save(fig, figures_dir / "fig_13_income_distribution_by_proximity.png", params["dpi"], logger)


def _fig14_missing_values(
    df: pd.DataFrame, config: dict, params: dict, figures_dir: Path, logger: logging.Logger
) -> str:
    """Bar chart of missing value percentage per column."""
    missing_threshold = config.get("descriptive", {}).get("missing_threshold", 0.05)

    fig, ax = plt.subplots(figsize=params["size_default"])
    try:
        missing_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
        colors = [
            "crimson" if v / 100 > missing_threshold else "steelblue"
            for v in missing_pct.values
        ]
        bars = ax.bar(missing_pct.index, missing_pct.values, color=colors, alpha=0.8)

        # Annotate bars with exact values
        for bar, val in zip(bars, missing_pct.values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.2f}%",
                    ha="center", va="bottom", fontsize=8,
                )

        ax.axhline(
            missing_threshold * 100, color="crimson", linestyle="--", linewidth=1.2,
            label=f"Threshold ({missing_threshold*100:.0f}%)",
        )
        ax.set_ylabel("Missing Values (%)", fontsize=11)
        ax.set_title("Missing Value Analysis per Column", fontsize=13)
        ax.set_xlabel("Column", fontsize=11)
        plt.xticks(rotation=30, ha="right")
        ax.legend(fontsize=9)
    except Exception as exc:
        logger.error("Error building fig_14: %s", exc)
        raise

    return _save(fig, figures_dir / "fig_14_missing_values.png", params["dpi"], logger)


def _fig15_outlier_analysis(
    df: pd.DataFrame, config: dict, params: dict, figures_dir: Path, logger: logging.Logger
) -> str:
    """Z-score distribution per feature with outlier threshold lines."""
    numeric_cols = config["schema"]["numeric_features"] + [config["schema"]["target"]]
    z_threshold = 3.0  # Standard ±3σ outlier boundary

    n_cols = 3
    n_rows = -(-len(numeric_cols) // n_cols)  # ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=params["size_large"])
    axes_flat = axes.flatten()

    try:
        palette = sns.color_palette(params["palette"], len(numeric_cols))
        for i, col in enumerate(numeric_cols):
            ax = axes_flat[i]
            data = df[col].dropna()
            z_scores = (data - data.mean()) / data.std()

            ax.hist(z_scores, bins=params["hist_bins"], density=True,
                    color=palette[i], alpha=0.65)
            try:
                kde_vals = gaussian_kde(z_scores)
                x_range = np.linspace(z_scores.min(), z_scores.max(), 300)
                ax.plot(x_range, kde_vals(x_range), color="black", linewidth=1.2)
            except Exception:
                pass

            ax.axvline(-z_threshold, color="crimson", linestyle="--", linewidth=1.2,
                       label=f"±{z_threshold}σ")
            ax.axvline(z_threshold, color="crimson", linestyle="--", linewidth=1.2)

            n_outliers = int(((z_scores < -z_threshold) | (z_scores > z_threshold)).sum())
            ax.set_title(f"{col}\n({n_outliers} outliers)", fontsize=8)
            ax.tick_params(labelsize=7)

        for j in range(len(numeric_cols), len(axes_flat)):
            axes_flat[j].set_visible(False)
    except Exception as exc:
        logger.error("Error building fig_15: %s", exc)
        raise

    fig.suptitle(f"Outlier Analysis — Z-Score Distributions (threshold = ±{z_threshold}σ)",
                 fontsize=13, y=1.01)
    return _save(fig, figures_dir / "fig_15_outlier_analysis.png", params["dpi"], logger)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(config: dict, base_dir: Path) -> list[str]:
    """
    Execute all EDA visualizations.

    Args:
        config:   Loaded configuration dict (from config/eda.yaml).
        base_dir: Root directory of the project (paths resolved from here).

    Returns:
        List of absolute paths to all saved PNG figures.
    """
    logger = _bootstrap_logger(config)
    logger.info("=== EDA Visualizations — START ===")

    try:
        style = config.get("visualizations", {}).get("style", "seaborn-v0_8-whitegrid")
        plt.style.use(style)
    except Exception:
        logger.warning("Could not apply style '%s'; using default.", style)

    try:
        df = _load_data(config, base_dir)
        logger.info("Loaded dataset: %d rows × %d columns", *df.shape)
    except FileNotFoundError as exc:
        logger.error("Dataset load failed: %s", exc)
        raise

    figures_dir = _ensure_dirs(config, base_dir)
    params = _fig_params(config)
    saved_paths: list[str] = []

    figure_fns = [
        _fig01_target_distribution,
        _fig02_log_target_distribution,
        _fig03_feature_distributions,
        _fig04_feature_boxplots,
        _fig05_correlation_heatmap,
        _fig06_pairplot_key_features,
        _fig07_geo_scatter_map,
        _fig08_geo_log_price_map,
        _fig09_income_vs_price_scatter,
        _fig10_price_by_proximity,
        _fig11_price_by_proximity_violin,
        _fig12_age_vs_price,
        _fig13_income_distribution_by_proximity,
        _fig14_missing_values,
        _fig15_outlier_analysis,
    ]

    for fn in figure_fns:
        try:
            path = fn(df, config, params, figures_dir, logger)
            saved_paths.append(path)
        except Exception as exc:
            logger.error("Figure function '%s' failed: %s", fn.__name__, exc)
            # Continue generating remaining figures rather than aborting

    logger.info("=== EDA Visualizations — DONE (%d/%d figures saved) ===",
                len(saved_paths), len(figure_fns))
    return saved_paths


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

    _saved = run(_config, _base_dir)
    print(f"\nSaved {len(_saved)} figure(s):")
    for _p in _saved:
        print(f"  {_p}")
