"""
statistical_tests.py — Statistical hypothesis testing for the California Housing dataset.

Entry point: run(config, base_dir)

Performs and saves:
  09_normality_tests.csv                  — Shapiro-Wilk, D'Agostino K², Anderson-Darling
  10_anova_results.json                   — one-way ANOVA across ocean_proximity groups
  11_kruskal_results.json                 — Kruskal-Wallis (non-parametric ANOVA)
  12_tukey_hsd.csv                        — pairwise Tukey HSD post-hoc comparisons
  13_correlation_tests.csv                — Pearson and Spearman corr + p-values vs target
  14_levene_test.json                     — Levene test for variance equality
  15_mannwhitney_inland_vs_coastal.json   — Mann-Whitney U: INLAND vs coastal
  16_chi2_tests.csv                       — chi-squared independence tests
  17_effect_sizes.json                    — Cohen's d and eta-squared effect sizes

All parameters are read from config/eda.yaml — no hardcoded values.
Statsmodels is used for Tukey HSD; its absence is handled gracefully.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Statsmodels is optional — graceful fallback if not installed
try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    _STATSMODELS_AVAILABLE = True
except ImportError:
    _STATSMODELS_AVAILABLE = False


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


def _save_json(data: Any, path: Path, logger: Any) -> None:
    """Serialise dict to JSON, converting numpy types."""
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
    """Save DataFrame to CSV."""
    df.to_csv(path)
    logger.info("Saved CSV: %s  (%d rows × %d cols)", path.name, len(df), len(df.columns))


def _add_income_category(df: pd.DataFrame, config: dict, logger: Any) -> pd.DataFrame:
    """Add income_category binned column when not already present."""
    if "income_category" in df.columns:
        return df
    df = df.copy()
    bins_cfg = config.get("pivot_tables", {}).get("bins", {}).get("income", {})
    if not bins_cfg:
        logger.warning("Income bin config missing — income_category not added.")
        return df
    try:
        df["income_category"] = pd.cut(
            df["median_income"],
            bins=bins_cfg["cuts"],
            labels=bins_cfg["labels"],
            right=True,
            include_lowest=True,
        )
    except Exception as exc:
        logger.warning("Could not create income_category: %s", exc)
    return df


def _interpret(p_value: float, alpha: float, description: str) -> str:
    """Return a plain-English interpretation string for a hypothesis test."""
    if p_value < alpha:
        return (
            f"Significant (p={p_value:.4g} < α={alpha}): {description} — "
            "the null hypothesis is rejected."
        )
    return (
        f"Not significant (p={p_value:.4g} ≥ α={alpha}): {description} — "
        "the null hypothesis cannot be rejected."
    )


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def normality_tests(df: pd.DataFrame, config: dict, logger: Any) -> pd.DataFrame:
    """
    Run three normality tests for every numeric feature:
      - Shapiro-Wilk     (on a random subsample for performance)
      - D'Agostino K²    (normaltest)
      - Anderson-Darling  (uses 5% critical value)

    Returns a tidy DataFrame with one row per (feature, test) pair.
    """
    logger.info("Running normality tests ...")
    alpha = config.get("statistical_tests", {}).get("alpha", 0.05)
    sample_n = config.get("statistical_tests", {}).get("normality_sample_size", 5000)
    seed = config.get("eda", {}).get("random_seed", 42)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    records = []

    for col in numeric_cols:
        series = df[col].dropna()

        # --- Shapiro-Wilk (subsample) ---
        sw_sample = series.sample(min(sample_n, len(series)), random_state=seed)
        try:
            sw_stat, sw_p = scipy_stats.shapiro(sw_sample)
            records.append({
                "feature": col,
                "test": "shapiro_wilk",
                "statistic": round(float(sw_stat), 6),
                "p_value": round(float(sw_p), 6),
                "significant": bool(sw_p < alpha),
                "interpretation": _interpret(
                    sw_p, alpha, f"'{col}' deviates from normality (Shapiro-Wilk, n={len(sw_sample)})"
                ),
            })
        except Exception as exc:
            logger.warning("Shapiro-Wilk failed for '%s': %s", col, exc)

        # --- D'Agostino K² ---
        try:
            k2_stat, k2_p = scipy_stats.normaltest(series)
            records.append({
                "feature": col,
                "test": "dagostino_k2",
                "statistic": round(float(k2_stat), 6),
                "p_value": round(float(k2_p), 6),
                "significant": bool(k2_p < alpha),
                "interpretation": _interpret(
                    k2_p, alpha, f"'{col}' deviates from normality (D'Agostino K²)"
                ),
            })
        except Exception as exc:
            logger.warning("D'Agostino K² failed for '%s': %s", col, exc)

        # --- Anderson-Darling ---
        try:
            ad_result = scipy_stats.anderson(series, dist="norm")
            # Use the 5% significance level index (index 2 in the standard table)
            idx_5pct = 2
            ad_stat = float(ad_result.statistic)
            ad_crit = float(ad_result.critical_values[idx_5pct])
            ad_significant = ad_stat > ad_crit
            records.append({
                "feature": col,
                "test": "anderson_darling",
                "statistic": round(ad_stat, 6),
                "p_value": float("nan"),  # Anderson returns critical values, not p-value
                "critical_value_5pct": round(ad_crit, 6),
                "significant": bool(ad_significant),
                "interpretation": (
                    f"Statistic {ad_stat:.4f} {'>' if ad_significant else '<='} "
                    f"critical value {ad_crit:.4f} at 5%: '{col}' "
                    f"{'deviates from' if ad_significant else 'does not deviate from'} normality."
                ),
            })
        except Exception as exc:
            logger.warning("Anderson-Darling failed for '%s': %s", col, exc)

    result = pd.DataFrame(records)
    logger.info("Normality tests complete: %d records.", len(records))
    return result


def anova_results(df: pd.DataFrame, config: dict, logger: Any) -> dict:
    """
    One-way ANOVA: median_house_value across ocean_proximity groups.

    H0: all group means are equal.
    """
    logger.info("Running one-way ANOVA ...")
    alpha = config.get("statistical_tests", {}).get("alpha", 0.05)
    target = config.get("schema", {}).get("target", "median_house_value")

    groups = [
        grp[target].dropna().values
        for _, grp in df.groupby("ocean_proximity")
    ]
    group_names = df["ocean_proximity"].dropna().unique().tolist()

    f_stat, p_value = scipy_stats.f_oneway(*groups)
    f_stat = float(f_stat)
    p_value = float(p_value)

    result = {
        "test": "one_way_anova",
        "groups": group_names,
        "n_groups": len(group_names),
        "f_statistic": round(f_stat, 6),
        "p_value": round(p_value, 6),
        "significant": bool(p_value < alpha),
        "interpretation": _interpret(
            p_value, alpha,
            f"median_house_value differs across {len(group_names)} ocean_proximity groups (ANOVA)"
        ),
    }
    logger.info("ANOVA: F=%.4f, p=%.4g, significant=%s", f_stat, p_value, result["significant"])
    return result


def kruskal_results(df: pd.DataFrame, config: dict, logger: Any) -> dict:
    """
    Kruskal-Wallis H-test (non-parametric ANOVA): median_house_value across ocean_proximity.

    H0: all group distributions are identical.
    Does not assume normality — preferred when ANOVA assumptions are violated.
    """
    logger.info("Running Kruskal-Wallis test ...")
    alpha = config.get("statistical_tests", {}).get("alpha", 0.05)
    target = config.get("schema", {}).get("target", "median_house_value")

    groups = [
        grp[target].dropna().values
        for _, grp in df.groupby("ocean_proximity")
    ]
    group_names = df["ocean_proximity"].dropna().unique().tolist()

    h_stat, p_value = scipy_stats.kruskal(*groups)
    h_stat = float(h_stat)
    p_value = float(p_value)

    result = {
        "test": "kruskal_wallis",
        "groups": group_names,
        "n_groups": len(group_names),
        "h_statistic": round(h_stat, 6),
        "p_value": round(p_value, 6),
        "significant": bool(p_value < alpha),
        "interpretation": _interpret(
            p_value, alpha,
            f"median_house_value distribution differs across ocean_proximity groups (Kruskal-Wallis)"
        ),
    }
    logger.info("Kruskal-Wallis: H=%.4f, p=%.4g, significant=%s", h_stat, p_value, result["significant"])
    return result


def tukey_hsd(df: pd.DataFrame, config: dict, logger: Any) -> pd.DataFrame:
    """
    Tukey HSD post-hoc pairwise comparisons of median_house_value across ocean_proximity.

    Requires statsmodels. Returns a DataFrame of pairwise results.
    If statsmodels is unavailable, returns an informational placeholder DataFrame.
    """
    logger.info("Running Tukey HSD post-hoc test ...")
    alpha = config.get("statistical_tests", {}).get("alpha", 0.05)
    target = config.get("schema", {}).get("target", "median_house_value")

    if not _STATSMODELS_AVAILABLE:
        logger.warning(
            "statsmodels not installed — Tukey HSD skipped. "
            "Install with: pip install statsmodels"
        )
        return pd.DataFrame(
            [{"note": "statsmodels not available — Tukey HSD could not be computed"}]
        )

    subset = df[[target, "ocean_proximity"]].dropna()
    try:
        tukey = pairwise_tukeyhsd(
            endog=subset[target],
            groups=subset["ocean_proximity"],
            alpha=alpha,
        )
        summary_df = pd.DataFrame(
            data=tukey._results_table.data[1:],
            columns=tukey._results_table.data[0],
        )
        summary_df["significant"] = summary_df["reject"].astype(bool)
        summary_df["interpretation"] = summary_df.apply(
            lambda row: (
                f"Groups '{row['group1']}' and '{row['group2']}' differ significantly "
                f"(mean diff={float(row['meandiff']):.2f}, p_adj={float(row['p-adj']):.4g})."
                if row["significant"]
                else
                f"No significant difference between '{row['group1']}' and '{row['group2']}' "
                f"(mean diff={float(row['meandiff']):.2f}, p_adj={float(row['p-adj']):.4g})."
            ),
            axis=1,
        )
        logger.info("Tukey HSD: %d pairwise comparisons.", len(summary_df))
        return summary_df
    except Exception as exc:
        logger.error("Tukey HSD failed: %s", exc)
        raise


def correlation_tests(df: pd.DataFrame, config: dict, logger: Any) -> pd.DataFrame:
    """
    Pearson and Spearman correlation between each numeric feature and the target,
    including two-tailed p-values.
    """
    logger.info("Running Pearson and Spearman correlation tests ...")
    alpha = config.get("statistical_tests", {}).get("alpha", 0.05)
    target = config.get("schema", {}).get("target", "median_house_value")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target]
    target_series = df[target].dropna()

    records = []
    for col in feature_cols:
        paired = df[[col, target]].dropna()
        x = paired[col]
        y = paired[target]

        try:
            pr_stat, pr_p = scipy_stats.pearsonr(x, y)
            records.append({
                "feature": col,
                "method": "pearson",
                "statistic": round(float(pr_stat), 6),
                "p_value": round(float(pr_p), 6),
                "significant": bool(pr_p < alpha),
                "n": len(paired),
                "interpretation": _interpret(pr_p, alpha, f"linear correlation between '{col}' and target"),
            })
        except Exception as exc:
            logger.warning("Pearson corr failed for '%s': %s", col, exc)

        try:
            sp_stat, sp_p = scipy_stats.spearmanr(x, y)
            records.append({
                "feature": col,
                "method": "spearman",
                "statistic": round(float(sp_stat), 6),
                "p_value": round(float(sp_p), 6),
                "significant": bool(sp_p < alpha),
                "n": len(paired),
                "interpretation": _interpret(sp_p, alpha, f"monotonic correlation between '{col}' and target"),
            })
        except Exception as exc:
            logger.warning("Spearman corr failed for '%s': %s", col, exc)

    result = pd.DataFrame(records)
    logger.info("Correlation tests complete: %d records.", len(records))
    return result


def levene_test(df: pd.DataFrame, config: dict, logger: Any) -> dict:
    """
    Levene test for equality of variances across ocean_proximity groups.

    H0: all groups have equal variance.
    Uses the median centre (more robust than mean) by default.
    """
    logger.info("Running Levene test for variance equality ...")
    alpha = config.get("statistical_tests", {}).get("alpha", 0.05)
    target = config.get("schema", {}).get("target", "median_house_value")

    groups = [
        grp[target].dropna().values
        for _, grp in df.groupby("ocean_proximity")
    ]
    group_names = df["ocean_proximity"].dropna().unique().tolist()

    lev_stat, p_value = scipy_stats.levene(*groups, center="median")
    lev_stat = float(lev_stat)
    p_value = float(p_value)

    result = {
        "test": "levene",
        "groups": group_names,
        "n_groups": len(group_names),
        "center": "median",
        "statistic": round(lev_stat, 6),
        "p_value": round(p_value, 6),
        "significant": bool(p_value < alpha),
        "interpretation": _interpret(
            p_value, alpha,
            "variances of median_house_value differ across ocean_proximity groups (Levene)"
        ),
    }
    logger.info("Levene: W=%.4f, p=%.4g, significant=%s", lev_stat, p_value, result["significant"])
    return result


def mannwhitney_inland_vs_coastal(df: pd.DataFrame, config: dict, logger: Any) -> dict:
    """
    Mann-Whitney U test: INLAND vs all coastal groups combined.

    Coastal is defined as any ocean_proximity value that is NOT 'INLAND'.
    H0: the two distributions are identical (no location shift).
    """
    logger.info("Running Mann-Whitney U test: INLAND vs coastal ...")
    alpha = config.get("statistical_tests", {}).get("alpha", 0.05)
    target = config.get("schema", {}).get("target", "median_house_value")

    inland_vals = df.loc[df["ocean_proximity"] == "INLAND", target].dropna().values
    coastal_vals = df.loc[df["ocean_proximity"] != "INLAND", target].dropna().values

    if len(inland_vals) == 0 or len(coastal_vals) == 0:
        raise ValueError(
            "Mann-Whitney U test requires at least one value in each group. "
            f"INLAND n={len(inland_vals)}, coastal n={len(coastal_vals)}."
        )

    u_stat, p_value = scipy_stats.mannwhitneyu(inland_vals, coastal_vals, alternative="two-sided")
    u_stat = float(u_stat)
    p_value = float(p_value)

    result = {
        "test": "mann_whitney_u",
        "group_a": "INLAND",
        "group_b": "coastal (non-INLAND)",
        "n_inland": int(len(inland_vals)),
        "n_coastal": int(len(coastal_vals)),
        "u_statistic": round(u_stat, 6),
        "p_value": round(p_value, 6),
        "significant": bool(p_value < alpha),
        "median_inland": round(float(np.median(inland_vals)), 2),
        "median_coastal": round(float(np.median(coastal_vals)), 2),
        "interpretation": _interpret(
            p_value, alpha,
            "median_house_value distribution differs between INLAND and coastal properties"
        ),
    }
    logger.info(
        "Mann-Whitney U: U=%.2f, p=%.4g, median_inland=%.0f, median_coastal=%.0f",
        u_stat, p_value, result["median_inland"], result["median_coastal"],
    )
    return result


def chi2_tests(df: pd.DataFrame, config: dict, logger: Any) -> pd.DataFrame:
    """
    Chi-squared tests for independence between categorical variable pairs.

    Pairs are read from config.statistical_tests.chi2_pairs.
    Requires income_category and age_category to be present.
    """
    logger.info("Running chi-squared independence tests ...")
    alpha = config.get("statistical_tests", {}).get("alpha", 0.05)
    pairs = config.get("statistical_tests", {}).get(
        "chi2_pairs",
        [["ocean_proximity", "income_category"]],
    )

    records = []
    for pair in pairs:
        col_a, col_b = pair[0], pair[1]
        if col_a not in df.columns or col_b not in df.columns:
            logger.warning("Chi2 test skipped — column(s) missing: %s, %s", col_a, col_b)
            continue

        contingency = pd.crosstab(df[col_a], df[col_b])
        try:
            chi2_stat, p_value, dof, expected = scipy_stats.chi2_contingency(contingency)
            chi2_stat = float(chi2_stat)
            p_value = float(p_value)
            # Cramér's V as effect size
            n = contingency.values.sum()
            min_dim = min(contingency.shape) - 1
            cramers_v = float(np.sqrt(chi2_stat / (n * min_dim))) if min_dim > 0 else float("nan")

            records.append({
                "col_a": col_a,
                "col_b": col_b,
                "chi2_statistic": round(chi2_stat, 6),
                "p_value": round(p_value, 6),
                "degrees_of_freedom": int(dof),
                "cramers_v": round(cramers_v, 6),
                "significant": bool(p_value < alpha),
                "interpretation": _interpret(
                    p_value, alpha,
                    f"'{col_a}' and '{col_b}' are statistically associated (chi-squared)"
                ),
            })
            logger.info("Chi2 %s × %s: χ²=%.4f, p=%.4g", col_a, col_b, chi2_stat, p_value)
        except Exception as exc:
            logger.error("Chi2 test failed for (%s, %s): %s", col_a, col_b, exc)

    return pd.DataFrame(records)


def effect_sizes(
    df: pd.DataFrame,
    anova_result: dict,
    config: dict,
    logger: Any,
) -> dict:
    """
    Compute effect size measures:

    - Cohen's d: INLAND vs <1H OCEAN (standardised mean difference)
    - Eta-squared (η²): proportion of variance explained by ocean_proximity in ANOVA

    Cohen's d = (mean_a - mean_b) / pooled_std
    Eta-squared = SS_between / SS_total (approximated from F and group sizes)
    """
    logger.info("Computing effect sizes ...")
    target = config.get("schema", {}).get("target", "median_house_value")

    result: dict[str, Any] = {}

    # --- Cohen's d: INLAND vs <1H OCEAN ---
    try:
        inland = df.loc[df["ocean_proximity"] == "INLAND", target].dropna()
        near = df.loc[df["ocean_proximity"] == "<1H OCEAN", target].dropna()

        mean_diff = float(inland.mean() - near.mean())
        pooled_std = float(
            np.sqrt(
                ((len(inland) - 1) * inland.std() ** 2 + (len(near) - 1) * near.std() ** 2)
                / (len(inland) + len(near) - 2)
            )
        )
        cohens_d = round(mean_diff / pooled_std, 6) if pooled_std != 0 else float("nan")

        if abs(cohens_d) < 0.2:
            d_magnitude = "negligible"
        elif abs(cohens_d) < 0.5:
            d_magnitude = "small"
        elif abs(cohens_d) < 0.8:
            d_magnitude = "medium"
        else:
            d_magnitude = "large"

        result["cohens_d_inland_vs_1h_ocean"] = {
            "group_a": "INLAND",
            "group_b": "<1H OCEAN",
            "n_a": int(len(inland)),
            "n_b": int(len(near)),
            "mean_a": round(float(inland.mean()), 2),
            "mean_b": round(float(near.mean()), 2),
            "pooled_std": round(pooled_std, 4),
            "cohens_d": cohens_d,
            "magnitude": d_magnitude,
            "interpretation": (
                f"Cohen's d = {cohens_d:.4f} ({d_magnitude} effect). "
                f"INLAND mean (${inland.mean():,.0f}) vs <1H OCEAN mean (${near.mean():,.0f})."
            ),
        }
        logger.info("Cohen's d (INLAND vs <1H OCEAN): %.4f (%s)", cohens_d, d_magnitude)
    except Exception as exc:
        logger.warning("Cohen's d computation failed: %s", exc)
        result["cohens_d_inland_vs_1h_ocean"] = {"error": str(exc)}

    # --- Eta-squared from ANOVA ---
    try:
        f_stat = anova_result.get("f_statistic", float("nan"))
        k = anova_result.get("n_groups", 0)  # number of groups
        n = int(df[target].notna().sum())     # total observations

        if k > 0 and n > k and not np.isnan(f_stat):
            df_between = k - 1
            df_within = n - k
            # η² = SS_between / SS_total ≈ F * df_between / (F * df_between + df_within)
            eta_sq = round(
                (f_stat * df_between) / (f_stat * df_between + df_within), 6
            )
            if eta_sq < 0.01:
                eta_magnitude = "negligible"
            elif eta_sq < 0.06:
                eta_magnitude = "small"
            elif eta_sq < 0.14:
                eta_magnitude = "medium"
            else:
                eta_magnitude = "large"

            result["eta_squared_anova"] = {
                "source": "ocean_proximity",
                "f_statistic": round(float(f_stat), 6),
                "df_between": int(df_between),
                "df_within": int(df_within),
                "n_total": int(n),
                "eta_squared": eta_sq,
                "magnitude": eta_magnitude,
                "interpretation": (
                    f"η² = {eta_sq:.4f} ({eta_magnitude} effect). "
                    f"ocean_proximity explains {eta_sq * 100:.2f}% of variance in {target}."
                ),
            }
            logger.info("Eta-squared: %.4f (%s)", eta_sq, eta_magnitude)
        else:
            result["eta_squared_anova"] = {"note": "Could not compute — insufficient group/sample info."}
    except Exception as exc:
        logger.warning("Eta-squared computation failed: %s", exc)
        result["eta_squared_anova"] = {"error": str(exc)}

    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(config: dict, base_dir: Path) -> dict:
    """
    Execute the full statistical testing suite.

    Args:
        config:   Merged configuration dict loaded from eda.yaml.
        base_dir: Repository root used to resolve relative paths.

    Returns:
        Dict with all test results keyed by test name.
    """
    _bootstrap_src_path(base_dir)
    from utils.logger import get_logger  # noqa: PLC0415

    logger = get_logger("eda.statistical_tests", config.get("logging", {}))
    logger.info("=== Statistical Tests EDA started ===")

    if not _STATSMODELS_AVAILABLE:
        logger.warning(
            "statsmodels is NOT installed. Tukey HSD will be skipped. "
            "Install with: pip install statsmodels"
        )

    dirs = _ensure_output_dirs(base_dir, config)
    stats_dir = dirs["stats_dir"]
    data_path = base_dir / config.get("paths", {}).get(
        "input_data", "data/processed/house_price.parquet"
    )

    try:
        df_raw = _load_data(data_path, logger)
    except FileNotFoundError as exc:
        logger.error("Failed to load data: %s", exc)
        raise

    # Add income_category for chi-squared tests
    df = _add_income_category(df_raw, config, logger)

    results: dict[str, Any] = {}

    # 9. Normality tests
    try:
        norm_df = normality_tests(df, config, logger)
        _save_csv(norm_df, stats_dir / "09_normality_tests.csv", logger)
        results["normality_tests"] = norm_df
    except Exception as exc:
        logger.error("Normality tests failed: %s", exc)
        raise

    # 10. ANOVA
    try:
        anova = anova_results(df, config, logger)
        _save_json(anova, stats_dir / "10_anova_results.json", logger)
        results["anova_results"] = anova
    except Exception as exc:
        logger.error("ANOVA failed: %s", exc)
        raise

    # 11. Kruskal-Wallis
    try:
        kruskal = kruskal_results(df, config, logger)
        _save_json(kruskal, stats_dir / "11_kruskal_results.json", logger)
        results["kruskal_results"] = kruskal
    except Exception as exc:
        logger.error("Kruskal-Wallis failed: %s", exc)
        raise

    # 12. Tukey HSD
    try:
        tukey_df = tukey_hsd(df, config, logger)
        _save_csv(tukey_df, stats_dir / "12_tukey_hsd.csv", logger)
        results["tukey_hsd"] = tukey_df
    except Exception as exc:
        logger.error("Tukey HSD failed: %s", exc)
        raise

    # 13. Pearson + Spearman correlation tests
    try:
        corr_df = correlation_tests(df, config, logger)
        _save_csv(corr_df, stats_dir / "13_correlation_tests.csv", logger)
        results["correlation_tests"] = corr_df
    except Exception as exc:
        logger.error("Correlation tests failed: %s", exc)
        raise

    # 14. Levene test
    try:
        levene = levene_test(df, config, logger)
        _save_json(levene, stats_dir / "14_levene_test.json", logger)
        results["levene_test"] = levene
    except Exception as exc:
        logger.error("Levene test failed: %s", exc)
        raise

    # 15. Mann-Whitney U
    try:
        mw = mannwhitney_inland_vs_coastal(df, config, logger)
        _save_json(mw, stats_dir / "15_mannwhitney_inland_vs_coastal.json", logger)
        results["mannwhitney_inland_vs_coastal"] = mw
    except Exception as exc:
        logger.error("Mann-Whitney U test failed: %s", exc)
        raise

    # 16. Chi-squared tests
    try:
        chi2_df = chi2_tests(df, config, logger)
        _save_csv(chi2_df, stats_dir / "16_chi2_tests.csv", logger)
        results["chi2_tests"] = chi2_df
    except Exception as exc:
        logger.error("Chi-squared tests failed: %s", exc)
        raise

    # 17. Effect sizes (depends on anova_results being in results)
    try:
        eff = effect_sizes(df, results.get("anova_results", {}), config, logger)
        _save_json(eff, stats_dir / "17_effect_sizes.json", logger)
        results["effect_sizes"] = eff
    except Exception as exc:
        logger.error("Effect sizes computation failed: %s", exc)
        raise

    logger.info("=== Statistical Tests EDA complete — 9 outputs saved ===")
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
