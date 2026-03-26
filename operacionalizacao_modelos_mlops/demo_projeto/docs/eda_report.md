# California Housing Price — EDA Report

**Project:** California Housing Price Prediction
**Dataset:** Kaggle `shibumohapatra/house-price` (California Housing, FHFA 1990 census)
**Analysis Date:** 2026-03-17
**Pipeline:** `eda/run_eda.py` — 7 modules, 53.9s total
**Outputs:** 29 stats files · 10 tables · 33 figures

---

## Table of Contents

1. [Dataset Overview](#1-dataset-overview)
2. [Descriptive Statistics](#2-descriptive-statistics)
3. [Missing Values & Data Quality](#3-missing-values--data-quality)
4. [Target Variable Analysis](#4-target-variable-analysis)
5. [Feature Distributions](#5-feature-distributions)
6. [Correlation Analysis](#6-correlation-analysis)
7. [Ocean Proximity Analysis](#7-ocean-proximity-analysis)
8. [Pivot Tables & Cross-tabulations](#8-pivot-tables--cross-tabulations)
9. [Statistical Tests](#9-statistical-tests)
10. [Interaction Effects](#10-interaction-effects)
11. [Feature Engineering](#11-feature-engineering)
12. [Clustering Analysis](#12-clustering-analysis)
13. [Key Findings Summary](#13-key-findings-summary)
14. [Modeling Recommendations](#14-modeling-recommendations)

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| Rows | 20,640 |
| Columns | 10 |
| Memory usage | 1.73 MB |
| Missing values | 207 (only `total_bedrooms`) |
| Target column | `median_house_value` |
| Categorical columns | `ocean_proximity` |
| Numeric columns | 9 |

**Column descriptions:**

| Column | Type | Description |
|---|---|---|
| `longitude` | float64 | Block group longitude (negative = west) |
| `latitude` | float64 | Block group latitude |
| `housing_median_age` | int64 | Median age of houses in block (capped at 52) |
| `total_rooms` | int64 | Total number of rooms in block |
| `total_bedrooms` | float64 | Total bedrooms (207 NaN, ~1%) |
| `population` | int64 | Block population |
| `households` | int64 | Number of households |
| `median_income` | float64 | Median income in $10,000s (~$5k–$150k) |
| `ocean_proximity` | str | Distance to ocean (5 categories) |
| `median_house_value` | int64 | **TARGET** — Median house value ($) |

---

## 2. Descriptive Statistics

### Key Numeric Statistics

| Feature | Mean | Median | Std | Min | Max | Skewness | Distribution |
|---|---|---|---|---|---|---|---|
| `median_house_value` | $206,856 | $179,700 | $115,396 | $14,999 | $500,001 | 0.98 | right-skewed |
| `median_income` | $38,707 | $35,348 | $18,998 | $4,999 | $150,001 | 1.65 | right-skewed |
| `housing_median_age` | 28.6 yr | 29.0 yr | 12.6 yr | 1 | 52 | 0.06 | normal |
| `total_rooms` | 2,636 | 2,127 | 2,182 | 2 | 39,320 | 4.15 | right-skewed |
| `total_bedrooms` | 538 | 435 | 421 | 1 | 6,445 | 3.46 | right-skewed |
| `population` | 1,425 | 1,166 | 1,132 | 3 | 35,682 | 4.94 | right-skewed |
| `households` | 500 | 409 | 382 | 1 | 6,082 | 3.41 | right-skewed |
| `latitude` | 35.63 | 34.26 | 2.14 | 32.54 | 41.95 | 0.47 | normal |
| `longitude` | -119.57 | -118.49 | 2.00 | -124.35 | -114.31 | -0.30 | normal |

**Key observation:** `total_rooms`, `total_bedrooms`, `population`, `households`, and `median_income` are all **right-skewed** with heavy tails. Log transformations are strongly recommended for these features before modeling.

### IQR Outlier Detection

All features except `latitude`, `longitude`, and `housing_median_age` have significant IQR-flagged outliers. `population` (skew=4.94) and `total_rooms` (skew=4.15) are most extreme.

---

## 3. Missing Values & Data Quality

| Issue | Detail | Recommendation |
|---|---|---|
| Missing values | `total_bedrooms`: 207 rows (1.00%) | Impute with median per `ocean_proximity` group |
| Target capping | `median_house_value == $500,001`: **965 rows (4.68%)** | Use censored regression or log-transform; flag as truncated in model |
| Age capping | `housing_median_age == 52`: known maximum, not a data error | Create binary flag `age_at_cap = (housing_median_age == 52)` |
| Income encoding | `median_income` in $10,000 units; max is exactly $150,001 | Possible capping — check at modeling stage |

> **Critical:** The 965 capped target values (4.68% of data) represent true values ≥ $500,001. Any model trained on this data will systematically underpredict expensive homes. Consider using a truncated regression or treating $500,001 as a censoring threshold.

---

## 4. Target Variable Analysis

### Distribution Shape
- Mean: $206,856 | Median: $179,700 | Std: $115,396
- Skewness: **0.98** (right-skewed — log transform reduces to ~0.13)
- The distribution has a visible peak around $100k–$250k with a heavy upper tail
- **Vertical spike at $500,001** confirms the hard cap in the data collection

### Log-transformation Effect
Applying `log(median_house_value)` substantially normalizes the distribution:
- Skewness: 0.98 → 0.13
- The log-transformed target has near-symmetrical distribution
- **Recommendation: use `log(median_house_value)` as the modeling target**, then exponentiate predictions

---

## 5. Feature Distributions

### Distribution Classification

| Feature | Shape | Skewness | Action |
|---|---|---|---|
| `longitude`, `latitude` | Near-normal, bimodal (2 metro clusters) | ~±0.3 | No transform needed; geographic binning useful |
| `housing_median_age` | Approximately normal, slight spike at 52 | 0.06 | Binary flag for capped values |
| `total_rooms` | Heavy right-skewed | 4.15 | `log1p` transform |
| `total_bedrooms` | Heavy right-skewed | 3.46 | `log1p` transform; impute NaN first |
| `population` | Extreme right-skewed | 4.94 | `log1p` transform; extreme outliers present |
| `households` | Heavy right-skewed | 3.41 | `log1p` transform |
| `median_income` | Moderate right-skewed | 1.65 | `log1p` useful; already shows better linearity with target |

---

## 6. Correlation Analysis

### Pearson Correlations with Target

| Feature | Pearson r | Interpretation |
|---|---|---|
| `median_income` | **+0.688** | Strong positive — income is the most predictive raw feature |
| `latitude` | -0.144 | Moderate negative — northern CA has lower prices |
| `total_rooms` | +0.134 | Weak positive — larger blocks, higher prices |
| `housing_median_age` | +0.106 | Weak positive — older neighborhoods (SF/Bay Area) |
| `households` | +0.066 | Very weak |
| `total_bedrooms` | +0.050 | Very weak |
| `longitude` | -0.046 | Very weak |
| `population` | -0.025 | Negligible |

**Key insight:** Only `median_income` has a meaningfully strong linear correlation with the target. All other raw features have weak Pearson r. This suggests **nonlinear relationships** and the importance of **feature engineering** to reveal predictive signal.

### Feature Correlation Matrix Findings
- `total_rooms`, `total_bedrooms`, `population`, `households` are strongly inter-correlated (r > 0.80) — **multicollinearity risk**
- `latitude` and `longitude` are nearly independent (r ≈ -0.12)
- `median_income` is moderately correlated with `total_rooms` (r = 0.20)

---

## 7. Ocean Proximity Analysis

### Mean House Values by Ocean Proximity

| Category | Count | Mean Value | Median Value | Mean Income |
|---|---|---|---|---|
| ISLAND | 5 | **$380,440** | $414,700 | $27,444 |
| NEAR BAY | 2,290 | **$259,212** | $233,800 | $41,729 |
| NEAR OCEAN | 2,658 | **$249,434** | $229,450 | $40,058 |
| <1H OCEAN | 9,136 | **$240,084** | $214,850 | $42,307 |
| INLAND | 6,551 | **$124,805** | $108,500 | $32,090 |

> `ISLAND` has only 5 observations — treat as rare class. `INLAND` properties cost on average **$115,279 less** than `<1H OCEAN` properties.

### Effect Size
- **One-Way ANOVA:** F=1,612.1, p≈0 — `ocean_proximity` significantly differentiates house values
- **Eta-squared η² = 0.238** — `ocean_proximity` alone **explains 23.8% of variance** in house value (large effect)
- **Cohen's d (INLAND vs <1H OCEAN) = -1.24** — large effect size; the two distributions have minimal overlap
- **Mann-Whitney U (INLAND vs all coastal):** median INLAND = $108,500 vs coastal = $219,500 (p≈0)

### Tukey HSD Post-hoc Comparisons

All pairs are significantly different **except ISLAND vs NEAR BAY** (p=0.056, marginally non-significant due to ISLAND's tiny sample size of n=5).

---

## 8. Pivot Tables & Cross-tabulations

### House Value by Income Category × Ocean Proximity

Crosstab of **mean house value** by income bin and ocean proximity reveals:

| Income | <1H OCEAN | INLAND | NEAR BAY | NEAR OCEAN |
|---|---|---|---|---|
| Very Low (<$15k) | $151,327 | $75,157 | $161,810 | $138,593 |
| Low ($15–30k) | $171,359 | $92,774 | $174,743 | $164,916 |
| Medium ($30–45k) | $219,573 | $134,161 | $250,418 | $237,973 |
| High ($45–60k) | $265,148 | $187,071 | $295,800 | $307,929 |
| Very High (>$60k) | $380,249 | $284,973 | $400,559 | $430,026 |

**Key observation:** At every income level, INLAND properties are substantially cheaper. The income effect is **amplified** at the high end: the gap between `NEAR OCEAN` and `INLAND` for Very High income is $430k–$285k = $145k, vs $139k–$75k = $64k for Very Low income. This confirms a significant **income × ocean proximity interaction**.

### Top 10 Most Expensive Block Groups
All concentrated in coastal areas (NEAR OCEAN, NEAR BAY), high income, short median age. Located geographically near Malibu / San Francisco waterfront.

### Bottom 10 Least Expensive Block Groups
All INLAND, very low income (<$10k), high population density.

---

## 9. Statistical Tests

### Normality Tests
All numeric features **fail normality** (Shapiro-Wilk, D'Agostino K², Anderson-Darling — all p≈0). This is expected for census block data:
- `total_rooms`, `total_bedrooms`, `population`, `households` — extreme right skew
- `housing_median_age` — closest to normal but fails at n=20,640 (normality tests are very sensitive at large N)
- **Implication:** Prefer **non-parametric** tests and **robust** estimators

### ANOVA & Kruskal-Wallis
| Test | Statistic | p-value | Significant | Interpretation |
|---|---|---|---|---|
| One-Way ANOVA | F=1,612.1 | ≈0 | Yes | House values differ significantly across ocean_proximity groups |
| Kruskal-Wallis | H=6,634.6 | ≈0 | Yes | Non-parametric confirmation: distributions differ |
| Levene (variance) | W=439.2 | ≈0 | Yes | Unequal variances across groups (ANOVA assumption violated) |

→ ANOVA result is robust even with the Levene violation due to large N (Central Limit Theorem), but **Kruskal-Wallis confirms the finding non-parametrically**.

### Chi-Squared Tests
- `ocean_proximity × income_category`: χ²=1,287.7, p≈0 — **strong dependence** between location type and income level. Coastal areas are wealthier, which partially explains their higher house prices.

---

## 10. Interaction Effects

### 2-Way: Income × Ocean Proximity
- The relationship between income and house price **differs in slope by ocean proximity**
- NEAR OCEAN and NEAR BAY show the steepest income–price slopes
- INLAND has a flatter slope — income matters less because of a lower price ceiling
- **Interaction term `median_income × op_NEAR OCEAN`** should be included in linear models

### 2-Way: Income × Housing Age (Heatmap)
- **Young blocks (<10yr) × High income (>$60k)**: highest mean prices ($350–$400k)
- **Old blocks (>40yr) × Low income**: lowest prices ($75–100k)
- Age has a **non-linear** relationship with price: blocks aged 20–35yr show the lowest prices on average, while old (>40yr) blocks in coastal areas show high prices (historical Bay Area neighborhoods)

### 2-Way: Income × Rooms per Household
- Higher rooms per household (more spacious) + higher income = substantially higher prices
- Income explains more variance than spaciousness; spaciousness is a secondary amplifier
- The `rooms_per_household × median_income` interaction term captures spacious high-income neighborhoods

### 3-Way: Income × Ocean Proximity × Age Category
The 3-way interaction confirms:
- For low-income groups, house age barely matters
- For high-income groups in coastal areas, the age-price relationship inverts: **older neighborhoods are MORE valuable** (historic districts, SF/Bay Area)

---

## 11. Feature Engineering

### Created Features and Their Correlation with Target

| Feature | Type | Correlation with Target | Interpretation |
|---|---|---|---|
| `log_median_house_value` | Log transform | **+0.949** | Self-correlation — confirms log target is nearly linear |
| `median_income` | Raw | +0.688 | Strongest raw predictor |
| `log_median_income` | Log transform | +0.670 | Slightly less predictive than raw (income is already ~linear to target) |
| `median_income_squared` | Polynomial | +0.625 | Captures diminishing returns at high income |
| `median_income_x_housing_median_age` | Interaction | +0.589 | Strong interaction feature |
| `op_INLAND` (dummy) | Categorical encoding | -0.485 | Location is a strong negative predictor |
| `ocean_proximity_encoded` | Ordinal encoding | -0.397 | Ordinal location encoding |
| `nearest_city_distance` | Geographic | -0.384 | Closer to city center = more expensive |
| `bedrooms_per_room` | Ratio | -0.256 | Lower ratio = more spacious = higher value |
| `log_population_per_household` | Log ratio | -0.247 | Higher density = lower value |
| `log_rooms_per_household` | Log ratio | +0.241 | More spacious = higher value |
| `rooms_per_household` | Ratio | +0.152 | Spaciousness measure |
| `dist_los_angeles` | Geographic | -0.117 | Distance from LA correlates with price |

### Top Engineered Feature Insights

1. **`median_income × housing_median_age`** (r=0.589): Old + affluent neighborhoods (SF, Berkeley) command premiums. This is the key interaction term.
2. **`bedrooms_per_room`** (r=-0.256): Lower ratio means larger rooms = more luxury. Each percentage point reduction in this ratio corresponds to meaningful price premium.
3. **`population_per_household`** (r=-0.247): Crowded blocks are cheaper. This proxies for neighborhood density/quality.
4. **`nearest_city_distance`** (r=-0.384): Pure geographic value — proximity to major employment centers.
5. **`rooms_per_household`** (r=0.152): More rooms per household reflects suburban/larger housing.

### Recommended Feature Set for Modeling

**Must include:**
- `median_income` (or `log_median_income`)
- `ocean_proximity` (one-hot encoded, `INLAND` dummy is especially important)
- `latitude` and `longitude` (or geographic cluster assignments)
- `bedrooms_per_room`
- `rooms_per_household` (or log version)
- `population_per_household` (or log version)

**Should include:**
- `nearest_city_distance` (distance to nearest of SF, LA, SJ, SD, Sacramento)
- `housing_median_age` (with cap flag)
- `median_income_squared` (polynomial for nonlinearity)
- `median_income × ocean_proximity_encoded` (interaction)

**Skip or transform:**
- `total_rooms`, `total_bedrooms`, `population`, `households` — replaced by ratio features
- Raw counts without normalization add noise and collinearity

---

## 12. Clustering Analysis

### Geographic K-Means (k=3 optimal, silhouette=0.647)

Three geographically coherent clusters:

| Cluster | Location | Mean House Value | Mean Income | Dominant Proximity |
|---|---|---|---|---|
| 0 | Los Angeles Basin | **$177,495** | $28,435 | <1H OCEAN, INLAND |
| 1 | Bay Area / Central Coast | **$191,703** | $36,542 | NEAR BAY, <1H OCEAN |
| 2 | Affluent Coastal | **$278,566** | $58,457 | <1H OCEAN, NEAR OCEAN |

The 3-cluster geographic structure reflects California's housing market geography:
- Cluster 0: Inland valleys and suburban LA, moderate prices
- Cluster 1: SF Bay Area inland and Sacramento corridor
- Cluster 2: Affluent coastal neighborhoods (Malibu, Pacific Heights, Monterey)

### Feature-Space K-Means (k=3 optimal, silhouette=0.302)

Multi-feature clustering (income, age, spaciousness, density, location) also finds 3 clusters but with lower cohesion (silhouette=0.30), suggesting the data does not form tight clusters in high-dimensional space.

### DBSCAN (Geographic)
DBSCAN found only **1 cluster** across all tested epsilon values (0.5°–1.5°), indicating the California housing data forms a **single continuous geographic distribution** without distinct density gaps. This is expected — California has continuous suburban development from San Diego to the Bay Area.

### PCA Analysis

**Principal Component Interpretation:**

| Component | Explained Variance | Dominant Features | Interpretation |
|---|---|---|---|
| PC1 | ~55% | `households`, `total_bedrooms`, `total_rooms`, `population` | **Block Scale** — large vs small blocks |
| PC2 | ~17% | `bedrooms_per_room`, `rooms_per_household`, `median_income` | **Quality/Efficiency** — spacious high-income vs dense low-income |
| PC3 | ~10% | `latitude`, `longitude` | **Geography** — north-south and east-west positioning |

**Total variance explained by PC1-PC3:** ~82%

In the PC1–PC2 scatter plot colored by house value, expensive properties cluster in the **upper right** (small block scale + high quality/income), confirming that the quality dimension is the most predictive.

---

## 13. Key Findings Summary

### What Predicts House Prices in California?

**Finding 1: Median Income is the dominant individual predictor**
- Pearson r = 0.688 with house value
- Income alone explains ~47% of variance (r²)
- Log income and income² add marginal improvements

**Finding 2: Location type (ocean_proximity) explains 23.8% of variance**
- ANOVA eta² = 0.238 — each location category has a distinct price distribution
- The coastal premium: INLAND is ~$115k cheaper than <1H OCEAN (Cohen's d = -1.24)
- NEAR BAY (San Francisco area) and NEAR OCEAN command the highest premiums

**Finding 3: Engineered ratio features are more predictive than raw counts**
- `bedrooms_per_room` (r=-0.256) and `rooms_per_household` (r=0.152) outperform raw `total_rooms` (r=0.134) and `total_bedrooms` (r=0.050)
- `population_per_household` (r=-0.247) captures neighborhood density better than raw `population`

**Finding 4: Geography beyond ocean_proximity matters**
- `nearest_city_distance` (r=-0.384) — proximity to major employment centers is highly predictive
- Geographic clusters identify 3 housing markets: LA Basin, Bay Area, Affluent Coastal
- Distance from San Francisco is the most predictive of the 5 city distances

**Finding 5: Key interaction effects**
- `median_income × housing_median_age` (r=0.589 with target) — old affluent neighborhoods command premium
- `median_income × ocean_proximity` — coastal areas amplify the income premium
- INLAND properties show a flatter income–price slope

**Finding 6: Non-linear relationships dominate**
- Most features fail normality tests
- `median_income_squared` adds predictive power beyond linear income
- Log target (`log_median_house_value`) substantially normalizes distributions

**Finding 7: Target censoring is a modeling challenge**
- 965 observations (4.68%) have `median_house_value == $500,001` (right-censored)
- Standard regression will systematically under-predict expensive homes
- Use log-transform + residual analysis to diagnose, or consider Tobit regression

### Priority Feature List for ML Model

Ranked by expected importance:

1. `median_income` / `log_median_income`
2. `op_INLAND` (dummy for inland location)
3. `latitude`, `longitude`
4. `nearest_city_distance`
5. `bedrooms_per_room`
6. `log_population_per_household`
7. `log_rooms_per_household`
8. `ocean_proximity_encoded`
9. `housing_median_age`
10. `median_income_squared`
11. `median_income_x_housing_median_age`
12. Geographic cluster label (k=3)

---

## 14. Modeling Recommendations

### Target Variable
- **Use `log(median_house_value)` as target**; back-transform predictions with `exp()`
- Acknowledge the $500,001 censoring; add `is_capped` as a feature or use survival analysis

### Missing Data
- Impute `total_bedrooms` with **median per `ocean_proximity` group** (207 rows, 1%)
- Do not use mean imputation — the feature is right-skewed

### Feature Preprocessing
1. **Log-transform:** `total_rooms`, `total_bedrooms`, `population`, `households`
2. **Create ratio features:** `bedrooms_per_room`, `rooms_per_household`, `population_per_household`
3. **Geographic:** `nearest_city_distance` (or distance to each of 5 cities separately)
4. **One-hot encode:** `ocean_proximity` (5 categories)
5. **Scale:** StandardScaler or MinMaxScaler after log transforms
6. **Drop raw counts:** `total_rooms`, `total_bedrooms`, `population`, `households` — replaced by ratios

### Recommended Models (in order of expected performance)
1. **Gradient Boosting (XGBoost/LightGBM)** — handles non-linearity, interactions, missing values
2. **Random Forest** — good baseline, interpretable feature importance
3. **Ridge Regression on log-transformed features** — fast baseline, interpretable
4. **Neural Network (MLP)** — can learn complex interactions but needs tuning

### Evaluation Metrics
- Primary: **RMSE on log scale** (log-RMSE, penalizes relative error)
- Secondary: **MAE** on original scale, **MAPE**
- Check residuals around $500,001 cap separately

### Expected Baseline Performance
Based on income alone (r=0.688): **RMSE ~$84,000 (~41% of mean)**
With all engineered features: target **RMSE ~$45,000–60,000** (Random Forest/GBM range)

---

## Outputs Reference

All analysis outputs are in `operacionalizacao_modelos_mlops/aula02/outputs/`:

```
outputs/
├── stats/           # 29 JSON + CSV statistical files
│   ├── 00_run_summary.json           Pipeline execution summary
│   ├── 01_basic_info.json            Dataset metadata
│   ├── 02_descriptive_stats.csv      Full descriptive statistics
│   ├── 03_missing_values.json        Missing value analysis
│   ├── 04_ocean_proximity_counts.csv Category distribution
│   ├── 05_outliers_iqr.csv           IQR outlier counts per feature
│   ├── 06_distribution_summary.csv   Distribution shape classification
│   ├── 07_correlation_matrix.csv     Feature–feature Pearson correlations
│   ├── 08_target_correlation.csv     Feature–target correlation ranking
│   ├── 09_normality_tests.csv        Shapiro-Wilk, D'Agostino, Anderson
│   ├── 10_anova_results.json         One-way ANOVA by ocean_proximity
│   ├── 11_kruskal_results.json       Kruskal-Wallis (non-parametric)
│   ├── 12_tukey_hsd.csv              Tukey post-hoc pairwise comparisons
│   ├── 13_correlation_tests.csv      Pearson + Spearman with p-values
│   ├── 14_levene_test.json           Levene variance equality test
│   ├── 15_mannwhitney_inland_vs_coastal.json  INLAND vs coastal comparison
│   ├── 16_chi2_tests.csv             Chi-squared independence tests
│   ├── 17_effect_sizes.json          Cohen's d and Eta-squared
│   ├── 18_interaction_2way_means.csv 2-way interaction mean tables
│   ├── 19_interaction_3way_means.csv 3-way interaction mean tables
│   ├── 20_engineered_feature_correlations.csv All features ranked by |r|
│   ├── 21_feature_engineering_summary.json    Created features metadata
│   ├── 22_geo_kmeans_scores.csv      Geographic KMeans silhouette scores
│   ├── 23_fullspace_kmeans_scores.csv Feature-space KMeans scores
│   ├── 24_cluster_profiles.csv       Feature means per cluster
│   ├── 25_dbscan_results.json        DBSCAN results per epsilon
│   ├── 26_pca_loadings.csv           PCA component feature loadings
│   ├── 27_pca_explained_variance.csv PCA cumulative explained variance
│   └── 28_cluster_price_stats.csv    Price stats per geographic cluster
│
├── tables/          # 10 CSV tables
│   ├── 01_pivot_proximity_income.csv    Pivot: proximity × income
│   ├── 02_pivot_proximity_age.csv       Pivot: proximity × age
│   ├── 03_pivot_income_age.csv          Pivot: income × age
│   ├── 04_stats_by_proximity.csv        Full stats by ocean_proximity
│   ├── 05_contingency_proximity_income.csv  Contingency table
│   ├── 06_top10_blocks.csv              Top 10 most expensive blocks
│   ├── 07_bottom10_blocks.csv           Bottom 10 cheapest blocks
│   ├── 08_crosstab_value_by_income_proximity.csv  Cross-tab
│   ├── 09_enriched_dataset_sample.csv   1000-row enriched dataset sample
│   └── 10_cluster_proximity_crosstab.csv  Cluster × ocean_proximity
│
└── figures/         # 33 PNG figures
    ├── fig_01 – Target distribution (raw)
    ├── fig_02 – Target distribution (log)
    ├── fig_03 – Feature distributions (grid)
    ├── fig_04 – Feature boxplots
    ├── fig_05 – Correlation heatmap
    ├── fig_06 – Pairplot (key features)
    ├── fig_07 – Geographic map (house value)
    ├── fig_08 – Geographic map (log house value)
    ├── fig_09 – Income vs price scatter
    ├── fig_10 – Price by proximity (boxplots)
    ├── fig_11 – Price by proximity (violins)
    ├── fig_12 – Age vs price (LOWESS)
    ├── fig_13 – Income distribution by proximity
    ├── fig_14 – Missing values bar chart
    ├── fig_15 – Outlier analysis (z-scores)
    ├── fig_16 – Interaction: income × proximity
    ├── fig_17 – Interaction: income × age (heatmap)
    ├── fig_18 – Interaction: income × rooms (heatmap)
    ├── fig_19 – Interaction: age × proximity (line)
    ├── fig_20 – Interaction: population density × income
    ├── fig_21 – 3-way: income × proximity × age
    ├── fig_23 – Engineered features distributions
    ├── fig_24 – Log vs raw distributions
    ├── fig_25 – Distance features geographic map
    ├── fig_26 – Feature importance (correlation)
    ├── fig_27 – KMeans elbow + silhouette
    ├── fig_28 – Geographic clusters (optimal k=3)
    ├── fig_29 – Full-space KMeans map
    ├── fig_30 – DBSCAN clusters map
    ├── fig_31 – PCA explained variance
    ├── fig_32 – PCA scatter (PC1 vs PC2)
    ├── fig_33 – PCA scatter (by ocean_proximity)
    └── fig_34 – Cluster price distributions
```

---

*Report generated by `eda/run_eda.py` — California Housing EDA Pipeline v1.0.0*
*Configuration: `config/eda.yaml` — All parameters externalized, no hardcoded values*
