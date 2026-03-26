"""
clustering.py — Clustering and dimensionality-reduction analysis for the California Housing dataset.

Entry point: run(config, base_dir)

Computes and saves:
  stats/22_geo_kmeans_scores.csv          — silhouette / CH / inertia per k (geo KMeans)
  stats/23_fullspace_kmeans_scores.csv    — silhouette / CH / inertia per k (full-space KMeans)
  stats/24_cluster_profiles.csv           — feature means per full-space cluster
  stats/25_dbscan_results.json            — DBSCAN run summary
  stats/26_pca_loadings.csv               — PC1 / PC2 / PC3 feature loadings
  stats/27_pca_explained_variance.csv     — per-component explained variance
  stats/28_cluster_price_stats.csv        — price stats per geo cluster
  tables/10_cluster_proximity_crosstab.csv — cluster × ocean_proximity counts

  fig_27_geo_kmeans_elbow.png             — elbow + silhouette plots
  fig_28_geo_kmeans_optimal_map.png       — geo map coloured by optimal k cluster
  fig_29_full_kmeans_map.png              — geo map coloured by full-space cluster
  fig_30_dbscan_clusters_map.png          — geo map coloured by DBSCAN cluster
  fig_31_pca_explained_variance.png       — cumulative explained variance curve
  fig_32_pca_scatter.png                  — PC1 vs PC2 coloured by house value
  fig_33_pca_by_proximity.png             — PC1 vs PC2 coloured by ocean_proximity
  fig_34_cluster_price_distribution.png   — violin plots: price per geo cluster

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
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler


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


def _add_ratio_features(df: pd.DataFrame, logger: Any) -> pd.DataFrame:
    """
    Compute ratio features required by full-space clustering.

    Only creates a feature if the source columns are present and the result
    column does not already exist.
    """
    ratio_map = {
        "rooms_per_household":      ("total_rooms",     "households"),
        "bedrooms_per_room":        ("total_bedrooms",  "total_rooms"),
        "population_per_household": ("population",      "households"),
    }
    for col, (num, den) in ratio_map.items():
        if col in df.columns:
            continue
        if num not in df.columns or den not in df.columns:
            logger.warning("Cannot compute '%s': missing column(s) '%s' or '%s'", col, num, den)
            continue
        df[col] = (df[num] / df[den].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return df


def _scale_features(X: pd.DataFrame, logger: Any) -> tuple[np.ndarray, StandardScaler]:
    """Fit a StandardScaler and return (scaled_array, fitted_scaler)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logger.info("Features scaled with StandardScaler — shape: %s", X_scaled.shape)
    return X_scaled, scaler


def _run_kmeans_grid(
    X_scaled: np.ndarray,
    n_clusters_range: list[int],
    random_state: int,
    logger: Any,
) -> pd.DataFrame:
    """
    Fit KMeans for each k in n_clusters_range.

    Returns a DataFrame with columns: k, inertia, silhouette, calinski_harabasz.
    """
    records: list[dict] = []
    for k in n_clusters_range:
        try:
            km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = km.fit_predict(X_scaled)
            sil = float(silhouette_score(X_scaled, labels, sample_size=min(5000, len(labels))))
            ch  = float(calinski_harabasz_score(X_scaled, labels))
            records.append({
                "k":                  k,
                "inertia":            float(km.inertia_),
                "silhouette":         sil,
                "calinski_harabasz":  ch,
            })
            logger.info("KMeans k=%d — silhouette=%.4f, CH=%.1f, inertia=%.1f", k, sil, ch, km.inertia_)
        except Exception as exc:
            logger.warning("KMeans k=%d failed: %s", k, exc)

    return pd.DataFrame(records)


def _select_optimal_k(scores_df: pd.DataFrame, metric: str, logger: Any) -> int:
    """Return the k with the best score for the chosen metric."""
    if scores_df.empty:
        logger.warning("Scores DataFrame is empty; defaulting to k=4.")
        return 4
    col = metric if metric in scores_df.columns else "silhouette"
    best_row = scores_df.loc[scores_df[col].idxmax()]
    k_opt = int(best_row["k"])
    logger.info("Optimal k selected by '%s': k=%d (score=%.4f)", col, k_opt, best_row[col])
    return k_opt


# ---------------------------------------------------------------------------
# 1. Geographic K-Means
# ---------------------------------------------------------------------------

def _geo_kmeans(
    df: pd.DataFrame,
    geo_cfg: dict,
    random_state: int,
    dirs: dict[str, Path],
    vis_cfg: dict,
    logger: Any,
) -> pd.Series:
    """
    Fit K-Means on latitude/longitude.

    Returns the cluster label Series (index aligned to df).
    """
    features      = geo_cfg.get("features", ["latitude", "longitude"])
    n_clusters_range = geo_cfg.get("n_clusters_range", [3, 4, 5, 6, 8, 10])
    metric        = geo_cfg.get("optimal_metric", "silhouette")
    dpi           = vis_cfg.get("figure_dpi", 120)

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for geo KMeans: {missing}")

    X = df[features].dropna()
    X_scaled, _ = _scale_features(X, logger)

    # --- grid search ---
    scores_df = _run_kmeans_grid(X_scaled, n_clusters_range, random_state, logger)
    if scores_df.empty:
        raise RuntimeError("No successful KMeans fits for geographic clustering.")

    out_scores = dirs["stats"] / "22_geo_kmeans_scores.csv"
    scores_df.to_csv(out_scores, index=False)
    logger.info("Saved geo KMeans scores → %s", out_scores)

    # --- elbow + silhouette plot (fig_27) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(scores_df["k"], scores_df["inertia"], marker="o", color="#1565C0")
    ax1.set_title("Elbow Plot — Inertia vs k")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia")
    ax1.tick_params(labelsize=8)

    ax2.plot(scores_df["k"], scores_df["silhouette"], marker="o", color="#2E7D32")
    ax2.set_title("Silhouette Score vs k")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.tick_params(labelsize=8)

    out_elbow = dirs["figures"] / "fig_27_geo_kmeans_elbow.png"
    _savefig(fig, out_elbow, dpi=dpi)
    logger.info("Saved → %s", out_elbow)

    # --- fit optimal k ---
    k_opt = _select_optimal_k(scores_df, metric, logger)
    km_final = KMeans(n_clusters=k_opt, random_state=random_state, n_init=10)
    labels_aligned = pd.Series(np.nan, index=df.index)
    labels_aligned.loc[X.index] = km_final.fit_predict(X_scaled)

    # --- optimal cluster map (fig_28) ---
    alpha = vis_cfg.get("geo_map", {}).get("alpha", 0.4)
    fig, ax = plt.subplots(figsize=vis_cfg.get("figure_size_map", [12, 9]))
    scatter = ax.scatter(
        df.loc[X.index, "longitude"],
        df.loc[X.index, "latitude"],
        c=labels_aligned.loc[X.index].astype(int),
        cmap="tab10",
        s=2,
        alpha=alpha,
    )
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_title(f"Geographic K-Means Clusters (k={k_opt})", fontsize=11)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    out_map = dirs["figures"] / "fig_28_geo_kmeans_optimal_map.png"
    _savefig(fig, out_map, dpi=dpi)
    logger.info("Saved → %s", out_map)

    return labels_aligned


# ---------------------------------------------------------------------------
# 2. Full Feature-Space K-Means
# ---------------------------------------------------------------------------

def _full_kmeans(
    df: pd.DataFrame,
    full_cfg: dict,
    random_state: int,
    dirs: dict[str, Path],
    vis_cfg: dict,
    logger: Any,
) -> pd.Series:
    """
    Fit K-Means on a multi-feature space.

    Returns the cluster label Series (index aligned to df).
    """
    features         = full_cfg.get("features", [])
    n_clusters_range = full_cfg.get("n_clusters_range", [3, 4, 5, 6, 8])
    metric           = "silhouette"
    dpi              = vis_cfg.get("figure_dpi", 120)

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for full-space KMeans: {missing}")

    X = df[features].dropna()
    X_scaled, _ = _scale_features(X, logger)

    # --- grid search ---
    scores_df = _run_kmeans_grid(X_scaled, n_clusters_range, random_state, logger)
    if scores_df.empty:
        raise RuntimeError("No successful KMeans fits for full-space clustering.")

    out_scores = dirs["stats"] / "23_fullspace_kmeans_scores.csv"
    scores_df.to_csv(out_scores, index=False)
    logger.info("Saved full-space KMeans scores → %s", out_scores)

    # --- fit optimal k ---
    k_opt = _select_optimal_k(scores_df, metric, logger)
    km_final = KMeans(n_clusters=k_opt, random_state=random_state, n_init=10)
    labels_raw = km_final.fit_predict(X_scaled)
    labels_aligned = pd.Series(np.nan, index=df.index)
    labels_aligned.loc[X.index] = labels_raw

    # --- cluster profiles ---
    profile_df = df.loc[X.index].copy()
    profile_df["cluster"] = labels_raw
    profile_csv = profile_df.groupby("cluster").mean(numeric_only=True)
    out_profiles = dirs["stats"] / "24_cluster_profiles.csv"
    profile_csv.to_csv(out_profiles)
    logger.info("Saved cluster profiles → %s", out_profiles)

    # --- geo map (fig_29) ---
    alpha = vis_cfg.get("geo_map", {}).get("alpha", 0.4)
    fig, ax = plt.subplots(figsize=vis_cfg.get("figure_size_map", [12, 9]))
    sc = ax.scatter(
        df.loc[X.index, "longitude"],
        df.loc[X.index, "latitude"],
        c=labels_raw,
        cmap="tab10",
        s=2,
        alpha=alpha,
    )
    plt.colorbar(sc, ax=ax, label="Cluster")
    ax.set_title(f"Full Feature-Space K-Means Clusters (k={k_opt})", fontsize=11)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    out_map = dirs["figures"] / "fig_29_full_kmeans_map.png"
    _savefig(fig, out_map, dpi=dpi)
    logger.info("Saved → %s", out_map)

    return labels_aligned


# ---------------------------------------------------------------------------
# 3. DBSCAN Geographic Clustering
# ---------------------------------------------------------------------------

def _dbscan_clustering(
    df: pd.DataFrame,
    dbscan_cfg: dict,
    dirs: dict[str, Path],
    vis_cfg: dict,
    logger: Any,
) -> pd.Series:
    """
    Run DBSCAN for each eps value; select the run with the best silhouette score.

    Returns the best cluster label Series (index aligned to df).
    """
    eps_range   = dbscan_cfg.get("eps_range", [0.5, 1.0, 1.5])
    min_samples = dbscan_cfg.get("min_samples", 100)
    features    = dbscan_cfg.get("features", ["latitude", "longitude"])
    dpi         = vis_cfg.get("figure_dpi", 120)
    alpha       = vis_cfg.get("geo_map", {}).get("alpha", 0.4)

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for DBSCAN: {missing}")

    X = df[features].dropna()
    X_scaled, _ = _scale_features(X, logger)

    results: list[dict] = []
    best_labels: np.ndarray | None = None
    best_sil = -2.0
    best_eps = eps_range[0]

    for eps in eps_range:
        try:
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(X_scaled)
            unique_labels = set(labels)
            n_clusters    = len(unique_labels - {-1})
            n_noise       = int((labels == -1).sum())

            sil_score: float | None = None
            if n_clusters > 1:
                mask = labels != -1
                if mask.sum() > 1:
                    sil_score = float(
                        silhouette_score(X_scaled[mask], labels[mask], sample_size=min(5000, mask.sum()))
                    )
                    if sil_score > best_sil:
                        best_sil    = sil_score
                        best_labels = labels
                        best_eps    = eps
            else:
                logger.warning(
                    "DBSCAN eps=%.1f found %d cluster(s) — skipping silhouette.", eps, n_clusters
                )

            entry: dict[str, Any] = {
                "eps":        eps,
                "min_samples": min_samples,
                "n_clusters":  n_clusters,
                "n_noise":     n_noise,
                "silhouette":  sil_score,
            }
            results.append(entry)
            logger.info(
                "DBSCAN eps=%.1f — clusters=%d, noise=%d, silhouette=%s",
                eps, n_clusters, n_noise,
                f"{sil_score:.4f}" if sil_score is not None else "N/A",
            )
        except Exception as exc:
            logger.warning("DBSCAN eps=%.1f failed: %s", eps, exc)

    out_json = dirs["stats"] / "25_dbscan_results.json"
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Saved DBSCAN results → %s", out_json)

    # --- plot best result (fig_30) ---
    labels_aligned = pd.Series(np.nan, index=df.index)
    if best_labels is not None:
        labels_aligned.loc[X.index] = best_labels
        fig, ax = plt.subplots(figsize=vis_cfg.get("figure_size_map", [12, 9]))
        sc = ax.scatter(
            df.loc[X.index, "longitude"],
            df.loc[X.index, "latitude"],
            c=best_labels,
            cmap="tab10",
            s=2,
            alpha=alpha,
        )
        plt.colorbar(sc, ax=ax, label="Cluster (-1 = noise)")
        ax.set_title(f"DBSCAN Geographic Clusters (eps={best_eps}, min_samples={min_samples})", fontsize=11)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    else:
        logger.warning("No valid DBSCAN result to plot; generating empty figure.")
        fig, ax = plt.subplots(figsize=vis_cfg.get("figure_size_map", [12, 9]))
        ax.text(0.5, 0.5, "DBSCAN: no valid cluster found\n(all noise or single cluster)",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.set_title("DBSCAN Geographic Clusters", fontsize=11)

    out_map = dirs["figures"] / "fig_30_dbscan_clusters_map.png"
    _savefig(fig, out_map, dpi=dpi)
    logger.info("Saved → %s", out_map)

    return labels_aligned


# ---------------------------------------------------------------------------
# 4. PCA Analysis
# ---------------------------------------------------------------------------

def _pca_analysis(
    df: pd.DataFrame,
    pca_cfg: dict,
    schema: dict,
    dirs: dict[str, Path],
    vis_cfg: dict,
    logger: Any,
) -> None:
    """
    Fit PCA on numeric features + engineered ratio columns.

    Saves loadings, explained variance, and three scatter/line plots.
    """
    n_components = pca_cfg.get("n_components", 3)
    dpi          = vis_cfg.get("figure_dpi", 120)

    # Base numeric features from schema
    base_numeric = schema.get("numeric_features", [])
    ratio_extras = ["rooms_per_household", "bedrooms_per_room", "population_per_household"]
    all_feats    = base_numeric + [f for f in ratio_extras if f in df.columns and f not in base_numeric]
    all_feats    = [f for f in all_feats if f in df.columns]

    X = df[all_feats].dropna()
    if X.empty or X.shape[1] < n_components:
        logger.warning("Insufficient data/features for PCA; skipping.")
        return

    X_scaled, _ = _scale_features(X, logger)
    n_components = min(n_components, X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(X_scaled)

    # --- save loadings ---
    loading_cols = [f"PC{i+1}" for i in range(n_components)]
    loadings_df = pd.DataFrame(
        pca.components_[:n_components].T,
        index=all_feats,
        columns=loading_cols,
    )
    out_loadings = dirs["stats"] / "26_pca_loadings.csv"
    loadings_df.to_csv(out_loadings)
    logger.info("Saved PCA loadings → %s", out_loadings)

    # --- save explained variance ---
    ev_df = pd.DataFrame({
        "component":           loading_cols,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance":      np.cumsum(pca.explained_variance_ratio_),
    })
    out_ev = dirs["stats"] / "27_pca_explained_variance.csv"
    ev_df.to_csv(out_ev, index=False)
    logger.info("Saved PCA explained variance → %s", out_ev)

    # --- fig_31: cumulative explained variance ---
    fig, ax = plt.subplots(figsize=vis_cfg.get("figure_size_default", [10, 6]))
    ax.plot(loading_cols, ev_df["cumulative_variance"], marker="o", color="#1565C0", label="Cumulative")
    ax.bar(loading_cols, pca.explained_variance_ratio_, color="#90CAF9", alpha=0.7, label="Per component")
    ax.axhline(pca_cfg.get("explained_variance_threshold", 0.90), color="red", linestyle="--",
               label=f"Threshold ({pca_cfg.get('explained_variance_threshold', 0.90):.0%})")
    ax.set_title("PCA — Explained Variance", fontsize=11)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.legend(fontsize=8)
    out_ev_fig = dirs["figures"] / "fig_31_pca_explained_variance.png"
    _savefig(fig, out_ev_fig, dpi=dpi)
    logger.info("Saved → %s", out_ev_fig)

    # PC1, PC2 arrays aligned to X.index
    pc1 = components[:, 0]
    pc2 = components[:, 1]

    # --- fig_32: PC1 vs PC2 coloured by house value ---
    target_col = schema.get("target", "median_house_value")
    if target_col in df.columns:
        color_vals = df.loc[X.index, target_col].values
        fig, ax = plt.subplots(figsize=vis_cfg.get("figure_size_default", [10, 6]))
        sc = ax.scatter(pc1, pc2, c=color_vals, cmap="YlOrRd", s=2, alpha=0.5)
        plt.colorbar(sc, ax=ax, label=target_col)
        ax.set_title("PCA: PC1 vs PC2 — coloured by Median House Value", fontsize=11)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        out_pca32 = dirs["figures"] / "fig_32_pca_scatter.png"
        _savefig(fig, out_pca32, dpi=dpi)
        logger.info("Saved → %s", out_pca32)

    # --- fig_33: PC1 vs PC2 coloured by ocean_proximity ---
    if "ocean_proximity" in df.columns:
        prox = df.loc[X.index, "ocean_proximity"]
        cats = prox.unique()
        color_map = {cat: i for i, cat in enumerate(sorted(cats))}
        prox_numeric = prox.map(color_map).values
        cmap = plt.get_cmap("tab10")
        fig, ax = plt.subplots(figsize=vis_cfg.get("figure_size_default", [10, 6]))
        for cat_name, cat_idx in color_map.items():
            mask = prox_numeric == cat_idx
            ax.scatter(pc1[mask], pc2[mask], label=cat_name, s=2, alpha=0.5, color=cmap(cat_idx / 10))
        ax.set_title("PCA: PC1 vs PC2 — coloured by Ocean Proximity", fontsize=11)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(markerscale=4, fontsize=8, loc="best")
        out_pca33 = dirs["figures"] / "fig_33_pca_by_proximity.png"
        _savefig(fig, out_pca33, dpi=dpi)
        logger.info("Saved → %s", out_pca33)


# ---------------------------------------------------------------------------
# 5. Cluster-Price Analysis
# ---------------------------------------------------------------------------

def _cluster_price_analysis(
    df: pd.DataFrame,
    geo_labels: pd.Series,
    target: str,
    dirs: dict[str, Path],
    vis_cfg: dict,
    logger: Any,
) -> None:
    """
    Violin plots of house price distribution per geo cluster (fig_34).
    Cross-tab cluster × ocean_proximity.
    Price stats per cluster CSV.
    """
    dpi = vis_cfg.get("figure_dpi", 120)

    valid_mask = geo_labels.notna()
    if valid_mask.sum() == 0 or target not in df.columns:
        logger.warning("No valid geo cluster labels or target column missing; skipping price analysis.")
        return

    cluster_col = geo_labels.loc[valid_mask].astype(int)
    target_vals  = df.loc[valid_mask, target]
    analysis_df  = pd.DataFrame({"cluster": cluster_col, target: target_vals})

    # --- price stats ---
    price_stats = (
        analysis_df.groupby("cluster")[target]
        .agg(["mean", "median", "std", "min", "max", "count"])
        .reset_index()
    )
    out_stats = dirs["stats"] / "28_cluster_price_stats.csv"
    price_stats.to_csv(out_stats, index=False)
    logger.info("Saved cluster price stats → %s", out_stats)

    # --- cross-tab with ocean_proximity ---
    if "ocean_proximity" in df.columns:
        prox_vals = df.loc[valid_mask, "ocean_proximity"]
        cross_df = pd.DataFrame({"cluster": cluster_col, "ocean_proximity": prox_vals})
        crosstab = pd.crosstab(cross_df["cluster"], cross_df["ocean_proximity"])
        out_crosstab = dirs["tables"] / "10_cluster_proximity_crosstab.csv"
        crosstab.to_csv(out_crosstab)
        logger.info("Saved cluster × proximity crosstab → %s", out_crosstab)

    # --- violin plot (fig_34) ---
    unique_clusters = sorted(analysis_df["cluster"].unique())
    groups = [analysis_df.loc[analysis_df["cluster"] == c, target].dropna().values
              for c in unique_clusters]
    # Filter out empty groups
    valid_pairs = [(c, g) for c, g in zip(unique_clusters, groups) if len(g) > 0]
    if not valid_pairs:
        logger.warning("No data groups for violin plot; skipping.")
        return

    cluster_labels_plot, groups_clean = zip(*valid_pairs)

    fig, ax = plt.subplots(figsize=vis_cfg.get("figure_size_large", [14, 10]))
    parts = ax.violinplot(groups_clean, showmedians=True)

    for pc in parts.get("bodies", []):
        pc.set_facecolor("#1E88E5")
        pc.set_alpha(0.6)

    ax.set_xticks(range(1, len(cluster_labels_plot) + 1))
    ax.set_xticklabels([f"Cluster {c}" for c in cluster_labels_plot], fontsize=9)
    ax.set_title("Median House Value Distribution per Geographic Cluster", fontsize=11)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Median House Value ($)")

    out_violin = dirs["figures"] / "fig_34_cluster_price_distribution.png"
    _savefig(fig, out_violin, dpi=dpi)
    logger.info("Saved → %s", out_violin)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(config: dict, base_dir: Path) -> dict:
    """
    Main entry point for the clustering EDA module.

    Args:
        config:   Full config dict loaded from config/eda.yaml.
        base_dir: Repository root (parent of eda/, config/, outputs/).

    Returns:
        Dict with cluster label Series:
          - 'geo_labels'      — geographic KMeans labels
          - 'full_labels'     — full feature-space KMeans labels
          - 'dbscan_labels'   — DBSCAN labels
    """
    _bootstrap_src_path(base_dir)

    from utils.logger import get_logger  # noqa: PLC0415

    logging_cfg = config.get("logging", {})
    log_file_rel = logging_cfg.get("log_file", "outputs/eda.log")
    logging_cfg  = {**logging_cfg, "log_file": str(base_dir / log_file_rel)}
    logger = get_logger("eda.clustering", logging_cfg)

    logger.info("=== Clustering EDA — START ===")

    # --- paths & output dirs ---------------------------------------------------
    paths_cfg = config.get("paths", {})
    data_path = base_dir / paths_cfg.get("input_data", "data/processed/house_price.parquet")
    dirs      = _ensure_output_dirs(base_dir, config)

    # --- config sections -------------------------------------------------------
    cluster_cfg  = config.get("clustering", {})
    vis_cfg      = config.get("visualizations", {})
    schema       = config.get("schema", {})
    eda_cfg      = config.get("eda", {})
    target       = schema.get("target", "median_house_value")
    random_state = int(eda_cfg.get("random_seed", 42))

    # --- load data -------------------------------------------------------------
    try:
        df = _load_data(data_path, logger)
    except FileNotFoundError as exc:
        logger.error("Data loading failed: %s", exc)
        raise

    # Ensure ratio features exist (needed for full-space clustering)
    df = _add_ratio_features(df, logger)

    result: dict[str, Any] = {}

    # --- 1. Geographic K-Means ------------------------------------------------
    try:
        geo_cfg    = cluster_cfg.get("geo_kmeans", {})
        geo_labels = _geo_kmeans(df, geo_cfg, random_state, dirs, vis_cfg, logger)
        result["geo_labels"] = geo_labels
    except Exception as exc:
        logger.error("Geographic KMeans failed: %s", exc, exc_info=True)
        geo_labels = pd.Series(np.nan, index=df.index)
        result["geo_labels"] = geo_labels

    # --- 2. Full Feature-Space K-Means ----------------------------------------
    try:
        full_cfg    = cluster_cfg.get("full_kmeans", {})
        full_labels = _full_kmeans(df, full_cfg, random_state, dirs, vis_cfg, logger)
        result["full_labels"] = full_labels
    except Exception as exc:
        logger.error("Full-space KMeans failed: %s", exc, exc_info=True)
        result["full_labels"] = pd.Series(np.nan, index=df.index)

    # --- 3. DBSCAN Geographic --------------------------------------------------
    try:
        dbscan_cfg    = cluster_cfg.get("dbscan", {})
        dbscan_labels = _dbscan_clustering(df, dbscan_cfg, dirs, vis_cfg, logger)
        result["dbscan_labels"] = dbscan_labels
    except Exception as exc:
        logger.error("DBSCAN clustering failed: %s", exc, exc_info=True)
        result["dbscan_labels"] = pd.Series(np.nan, index=df.index)

    # --- 4. PCA ----------------------------------------------------------------
    try:
        pca_cfg = cluster_cfg.get("pca", {})
        _pca_analysis(df, pca_cfg, schema, dirs, vis_cfg, logger)
    except Exception as exc:
        logger.error("PCA analysis failed: %s", exc, exc_info=True)

    # --- 5. Cluster-Price Analysis ---------------------------------------------
    try:
        _cluster_price_analysis(df, result["geo_labels"], target, dirs, vis_cfg, logger)
    except Exception as exc:
        logger.error("Cluster price analysis failed: %s", exc, exc_info=True)

    logger.info("=== Clustering EDA — END ===")
    return result


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml

    _this_dir = Path(__file__).resolve().parent
    _base_dir = _this_dir.parent
    _cfg_path = _base_dir / "config" / "eda.yaml"

    if not _cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {_cfg_path}")

    with _cfg_path.open("r", encoding="utf-8") as _fh:
        _config = yaml.safe_load(_fh)

    results = run(_config, _base_dir)
    print("Done. Keys returned:", list(results.keys()))
