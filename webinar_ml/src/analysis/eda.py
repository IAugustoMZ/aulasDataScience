"""
EDA computation module for webinar_ml.
All functions receive DataFrames and config dicts; none produce side effects
beyond returning figures or DataFrames. Display and saving happen in the notebook.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
import yaml


# ── Config helpers ─────────────────────────────────────────────────────────────

def load_config(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_root() -> Path:
    # src/analysis/eda.py  ->  ../../  =  project root
    return Path(__file__).parent.parent.parent


def apply_plot_style(config: dict) -> None:
    style = config["plots"].get("style", "seaborn-v0_8-whitegrid")
    try:
        plt.style.use(style)
    except OSError:
        plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("notebook", font_scale=config["plots"].get("font_scale", 1.1))


# ── 1. Class distribution ──────────────────────────────────────────────────────

def plot_class_distribution(
    df: pd.DataFrame,
    config: dict,
    title: str = "Distribuição das Classes de Risco",
) -> plt.Figure:
    class_order = config["class_order"]
    palette = config["class_palette"]
    annotated = df[df["anotado"]].copy()

    counts = annotated["classe_risco"].value_counts()
    fracs = annotated["classe_risco"].value_counts(normalize=True)

    data = pd.DataFrame({"count": counts, "fraction": fracs}).reindex(
        [c for c in class_order if c in counts.index]
    )

    fig, axes = plt.subplots(1, 2, figsize=config["plots"]["figsize_wide"])

    colors = [palette[c] for c in data.index]

    axes[0].bar(data.index, data["count"], color=colors, edgecolor="white", linewidth=0.8)
    axes[0].set_title("Contagem absoluta", fontsize=12)
    axes[0].set_xlabel("Classe de risco")
    axes[0].set_ylabel("Registros")
    for i, (cls, row) in enumerate(data.iterrows()):
        axes[0].text(i, row["count"] + 100, f"{row['count']:,}", ha="center", fontsize=9)

    wedges, texts, autotexts = axes[1].pie(
        data["fraction"],
        labels=data.index,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
    )
    for at in autotexts:
        at.set_fontsize(9)
    axes[1].set_title("Proporção (%)", fontsize=12)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def imbalance_summary(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    class_order = config["class_order"]
    annotated = df[df["anotado"]]
    counts = annotated["classe_risco"].value_counts()
    fracs = annotated["classe_risco"].value_counts(normalize=True)
    majority = counts.max()

    rows = []
    for cls in class_order:
        if cls not in counts:
            continue
        rows.append({
            "classe": cls,
            "n": counts[cls],
            "fracção": f"{fracs[cls]:.1%}",
            "ratio_vs_majoritária": round(majority / counts[cls], 1),
        })
    return pd.DataFrame(rows).set_index("classe")


# ── 2. Noise / annotation analysis ───────────────────────────────────────────

def plot_annotation_breakdown(df: pd.DataFrame, config: dict) -> plt.Figure:
    noise_cfg = config["noise"]
    class_order = config["class_order"]
    palette = config["class_palette"]
    n = len(df)

    n_annotated = int(df[noise_cfg["annotated_col"]].sum())
    n_unannotated = n - n_annotated

    fig, axes = plt.subplots(1, 2, figsize=config["plots"]["figsize_wide"])

    # Esquerda: cobertura de anotação
    categories = ["Anotados", "Sem rótulo"]
    values = [n_annotated, n_unannotated]
    colors = ["#4CAF50", "#9E9E9E"]
    bars = axes[0].barh(categories, values, color=colors, edgecolor="white")
    axes[0].set_xlabel("Registros")
    axes[0].set_title("Cobertura de anotação")
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
                     f"{val:,} ({val/n:.1%})", va="center", fontsize=9)
    axes[0].set_xlim(0, max(values) * 1.3)

    # Direita: distribuição de classes nos registros anotados
    annotated_df = df[df[noise_cfg["annotated_col"]]].copy()
    class_counts = annotated_df["classe_risco"].value_counts().reindex(class_order).fillna(0)
    bar_colors = [palette.get(c, "#888888") for c in class_order]
    axes[1].bar(class_order, class_counts.values, color=bar_colors, edgecolor="white")
    axes[1].set_title("Distribuição de classes nos registros anotados")
    axes[1].set_xlabel("Classe de risco")
    axes[1].set_ylabel("Registros")
    for i, (cls, val) in enumerate(zip(class_order, class_counts.values)):
        axes[1].text(i, val + 30, f"{int(val):,}", ha="center", fontsize=9)

    fig.suptitle("Composição do dataset: cobertura e distribuição de classes", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_annotation_coverage_by_feature(df: pd.DataFrame, feature: str, config: dict) -> plt.Figure:
    """Compara distribuição de uma feature entre registros anotados e não anotados.

    Em projetos reais, esta análise responde: 'O conjunto não anotado é uma
    amostra representativa do que já rotulamos, ou tem viés de área/equipamento?'
    """
    noise_cfg = config["noise"]

    annotated = df[df[noise_cfg["annotated_col"]]][feature].value_counts(normalize=True).rename("anotado")
    unannotated = df[~df[noise_cfg["annotated_col"]]][feature].value_counts(normalize=True).rename("nao_anotado")

    comparison = pd.concat([annotated, unannotated], axis=1).fillna(0).sort_values("anotado", ascending=False)

    fig, ax = plt.subplots(figsize=config["plots"]["figsize_wide"])
    x = np.arange(len(comparison))
    width = 0.4
    ax.bar(x - width / 2, comparison["anotado"], width, label="Anotados", color="#4CAF50", edgecolor="white")
    ax.bar(x + width / 2, comparison["nao_anotado"], width, label="Sem rótulo", color="#9E9E9E", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison.index, rotation=30, ha="right")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_title(f"Distribuição de '{feature}': anotados vs. sem rótulo")
    ax.set_ylabel("Proporção")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_noise_by_class(df: pd.DataFrame, config: dict) -> plt.Figure:
    return plot_annotation_coverage_by_feature(df, "area_fpso", config)


# ── 3. Text length analysis ────────────────────────────────────────────────────

def plot_text_length_distribution(df: pd.DataFrame, config: dict) -> plt.Figure:
    text_cfg = config["text"]
    class_order = config["class_order"]
    palette = config["class_palette"]

    df = df.copy()
    df["n_words"] = df[text_cfg["column"]].str.split().str.len()

    annotated = df[df["anotado"]].copy()

    fig, axes = plt.subplots(1, 2, figsize=config["plots"]["figsize_wide"])

    # Overall distribution
    bins = text_cfg["word_length_bins"]
    axes[0].hist(df["n_words"].dropna(), bins=bins, color="#5C6BC0",
                 edgecolor="white", linewidth=0.6)
    axes[0].set_title("Distribuição de comprimento de relatos (dataset completo)")
    axes[0].set_xlabel("Número de palavras")
    axes[0].set_ylabel("Frequência")

    # By class
    for cls in class_order:
        subset = annotated[annotated["classe_risco"] == cls]["n_words"].dropna()
        if len(subset) == 0:
            continue
        axes[1].hist(subset, bins=bins, alpha=0.5, label=cls,
                     color=palette.get(cls, None), edgecolor="white", linewidth=0.4)
    axes[1].set_title("Comprimento de relatos por classe de risco")
    axes[1].set_xlabel("Número de palavras")
    axes[1].set_ylabel("Frequência")
    axes[1].legend()

    fig.suptitle("Análise de comprimento dos relatos", fontweight="bold")
    fig.tight_layout()
    return fig


def text_length_stats_by_class(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    text_col = config["text"]["column"]
    class_order = config["class_order"]
    annotated = df[df["anotado"]].copy()
    annotated["n_words"] = annotated[text_col].str.split().str.len()
    stats = (
        annotated.groupby("classe_risco")["n_words"]
        .agg(["min", "max", "mean", "median", "std"])
        .round(1)
    )
    present = [c for c in class_order if c in stats.index]
    return stats.reindex(present)


# ── 4. Token frequency analysis ───────────────────────────────────────────────

def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return text.lower()


def _tokenize(text: str, min_len: int) -> list[str]:
    tokens = re.findall(r"\b[a-zA-Z]+\b", _normalize(str(text)))
    return [t for t in tokens if len(t) >= min_len]


def build_stopwords(config: dict) -> set[str]:
    extra = set(config["text"].get("stopwords_extra", []))
    try:
        from nltk.corpus import stopwords as nltk_sw
        import nltk
        try:
            base = set(nltk_sw.words("portuguese"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            base = set(nltk_sw.words("portuguese"))
        return base | extra
    except ImportError:
        return extra


def top_tokens(
    df: pd.DataFrame,
    config: dict,
    class_filter: str | None = None,
) -> pd.DataFrame:
    text_col = config["text"]["column"]
    min_len = config["text"]["min_token_length"]
    top_n = config["text"]["top_n_tokens"]
    stopwords = build_stopwords(config)

    subset = df if class_filter is None else df[df["classe_risco"] == class_filter]
    counter: Counter = Counter()
    for text in subset[text_col].dropna():
        tokens = _tokenize(text, min_len)
        counter.update(t for t in tokens if t not in stopwords)

    rows = counter.most_common(top_n)
    return pd.DataFrame(rows, columns=["token", "count"])


def top_bigrams(df: pd.DataFrame, config: dict, class_filter: str | None = None) -> pd.DataFrame:
    text_col = config["text"]["column"]
    min_len = config["text"]["min_token_length"]
    top_n = config["text"]["top_n_bigrams"]
    stopwords = build_stopwords(config)

    subset = df if class_filter is None else df[df["classe_risco"] == class_filter]
    counter: Counter = Counter()
    for text in subset[text_col].dropna():
        tokens = [t for t in _tokenize(text, min_len) if t not in stopwords]
        counter.update(zip(tokens, tokens[1:]))

    rows = [(" ".join(bg), cnt) for bg, cnt in counter.most_common(top_n)]
    return pd.DataFrame(rows, columns=["bigrama", "count"])


def plot_top_tokens_by_class(df: pd.DataFrame, config: dict) -> plt.Figure:
    class_order = config["class_order"]
    palette = config["class_palette"]
    annotated = df[df["anotado"]].copy()

    present = [c for c in class_order if c in annotated["classe_risco"].unique()]
    n_classes = len(present)
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 6))
    if n_classes == 1:
        axes = [axes]

    for ax, cls in zip(axes, present):
        tokens = top_tokens(annotated, config, class_filter=cls).head(15)
        ax.barh(tokens["token"][::-1], tokens["count"][::-1],
                color=palette.get(cls, "#888"), edgecolor="white")
        ax.set_title(f"Classe: {cls}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Freq.")
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle("Top 15 tokens por classe de risco (sem stopwords)", fontweight="bold")
    fig.tight_layout()
    return fig


# ── 5. Categorical feature analysis ───────────────────────────────────────────

def plot_categorical_vs_class(
    df: pd.DataFrame,
    feature: str,
    config: dict,
) -> plt.Figure:
    class_order = config["class_order"]
    palette = config["class_palette"]
    top_n = config["categorical_features"]["top_n_per_feature"]

    annotated = df[df["anotado"]].copy()

    top_cats = annotated[feature].value_counts().head(top_n).index.tolist()
    subset = annotated[annotated[feature].isin(top_cats)]

    ct = pd.crosstab(
        subset[feature],
        subset["classe_risco"],
        normalize="index",
    )
    present_order = [c for c in class_order if c in ct.columns]
    ct = ct[present_order]
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=config["plots"]["figsize_wide"])
    colors = [palette[c] for c in present_order]
    ct.plot(kind="barh", stacked=True, ax=ax, color=colors,
            edgecolor="white", linewidth=0.5)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_title(f"Distribuição de classe de risco por '{feature}' (normalizada)")
    ax.set_xlabel("Proporção")
    ax.set_ylabel(feature)
    ax.legend(title="Classe", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    return fig


# ── 6. Cramér's V association ──────────────────────────────────────────────────

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    from scipy.stats import chi2_contingency
    ct = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(ct)
    n = ct.sum().sum()
    r, k = ct.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1) + 1e-9)))


def association_table(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    features = config["association"]["cross_tab_features"]
    threshold = config["association"]["cramers_v_threshold"]
    annotated = df[df["anotado"]].copy()

    rows = []
    for feat in features:
        if feat not in annotated.columns:
            continue
        v = cramers_v(annotated[feat], annotated["classe_risco"])
        rows.append({
            "feature": feat,
            "cramers_v": round(v, 4),
            "associacao_forte": v >= threshold,
        })
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)


def plot_association_heatmap(df: pd.DataFrame, config: dict) -> plt.Figure:
    features = config["association"]["cross_tab_features"]
    annotated = df[df["anotado"]].copy()
    all_features = [f for f in features if f in annotated.columns] + ["classe_risco"]

    matrix = pd.DataFrame(index=all_features, columns=all_features, dtype=float)
    for f1, f2 in [(a, b) for a in all_features for b in all_features]:
        matrix.loc[f1, f2] = cramers_v(annotated[f1], annotated[f2])

    fig, ax = plt.subplots(figsize=config["plots"]["figsize_default"])
    sns.heatmap(
        matrix.astype(float),
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        ax=ax,
        linewidths=0.4,
        annot_kws={"size": 8},
    )
    ax.set_title("Cramér's V — Associação entre features categóricas e classe de risco")
    fig.tight_layout()
    return fig


# ── 7. Temporal analysis ───────────────────────────────────────────────────────

def plot_temporal_trend(df: pd.DataFrame, config: dict) -> plt.Figure:
    temp_cfg = config["temporal"]
    class_order = config["class_order"]
    palette = config["class_palette"]

    annotated = df[df["anotado"]].copy()
    annotated[temp_cfg["date_col"]] = pd.to_datetime(annotated[temp_cfg["date_col"]])
    annotated = annotated.set_index(temp_cfg["date_col"])

    monthly = (
        annotated.groupby([pd.Grouper(freq=temp_cfg["freq"]), "classe_risco"])
        .size()
        .unstack(fill_value=0)
    )
    present = [c for c in class_order if c in monthly.columns]
    monthly = monthly[present]
    rolling = monthly.rolling(window=temp_cfg["rolling_window"], min_periods=1).mean()

    fig, axes = plt.subplots(2, 1, figsize=config["plots"]["figsize_tall"], sharex=True)

    for cls in present:
        axes[0].plot(monthly.index, monthly[cls], alpha=0.3, color=palette[cls], linewidth=1)
        axes[0].plot(rolling.index, rolling[cls], label=cls,
                     color=palette[cls], linewidth=2)
    axes[0].set_title(f"Ocorrências mensais por classe (média móvel {temp_cfg['rolling_window']} meses)")
    axes[0].set_ylabel("Registros")
    axes[0].legend(title="Classe")

    monthly_total = monthly.sum(axis=1)
    axes[1].fill_between(monthly_total.index, monthly_total, alpha=0.4, color="#5C6BC0")
    axes[1].plot(monthly_total.index, monthly_total, color="#3949AB", linewidth=1.5)
    axes[1].set_title("Volume total de registros por mês")
    axes[1].set_ylabel("Total registros")
    axes[1].set_xlabel("Data")

    fig.suptitle("Evolução temporal dos registros de segurança", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_hour_of_day(df: pd.DataFrame, config: dict) -> plt.Figure:
    """Incident count by hour-of-day, split by risk class.

    The handover spike (hours 6-7 and 18-19) is the key teaching moment:
    timestamps are a predictive feature, not just metadata.
    """
    class_order = config["class_order"]
    palette = config["class_palette"]

    annotated = df[df["anotado"]].copy()
    dt_col = config["temporal"].get("datetime_col", "data_hora_ocorrencia")
    annotated[dt_col] = pd.to_datetime(annotated[dt_col], utc=True)
    annotated["_hora"] = annotated[dt_col].dt.hour

    fig, axes = plt.subplots(2, 1, figsize=config["plots"]["figsize_tall"])

    # Top: stacked count per hour per class
    pivot = (
        annotated.groupby(["_hora", "classe_risco"])
        .size()
        .unstack(fill_value=0)
    )
    present = [c for c in class_order if c in pivot.columns]
    pivot = pivot[present]
    bottom = np.zeros(len(pivot))
    for cls in present:
        axes[0].bar(pivot.index, pivot[cls], bottom=bottom,
                    color=palette[cls], label=cls, edgecolor="white", linewidth=0.3)
        bottom += pivot[cls].values
    axes[0].set_title("Incidentes por hora do dia (contagem absoluta por classe)")
    axes[0].set_xlabel("Hora (BRT)")
    axes[0].set_ylabel("Registros")
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].axvspan(6.5, 7.5, alpha=0.12, color="red", label="Passagem de turno")
    axes[0].axvspan(18.5, 19.5, alpha=0.12, color="red")
    axes[0].legend(title="Classe", fontsize=8)

    # Bottom: proportion of critico+alto per hour (risk intensity curve)
    high_risk = annotated[annotated["classe_risco"].isin(["critico", "alto"])].copy()
    hr_by_hour = high_risk.groupby("_hora").size()
    total_by_hour = annotated.groupby("_hora").size().replace(0, np.nan)
    risk_rate = (hr_by_hour / total_by_hour).reindex(range(24), fill_value=0)

    axes[1].fill_between(risk_rate.index, risk_rate.values, alpha=0.35, color="#EF5350")
    axes[1].plot(risk_rate.index, risk_rate.values, color="#C62828", linewidth=2)
    axes[1].axvspan(6.5, 7.5, alpha=0.12, color="red")
    axes[1].axvspan(18.5, 19.5, alpha=0.12, color="red")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    axes[1].set_title("Taxa de risco alto/crítico por hora (% do total da hora)")
    axes[1].set_xlabel("Hora (BRT)")
    axes[1].set_ylabel("Taxa alto + crítico")
    axes[1].set_xticks(range(0, 24, 2))

    fig.suptitle(
        "Distribuição temporal de incidentes por hora do dia\n"
        "Barras vermelhas = janela de passagem de turno",
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_shift_vs_class_heatmap(df: pd.DataFrame, config: dict) -> plt.Figure:
    """Normalised heatmap: turno × classe_risco (proportions, not counts).

    Removes shift-size artifacts. Shows relative risk intensity per shift.
    """
    class_order = config["class_order"]
    palette = config["class_palette"]

    annotated = df[df["anotado"]].copy()
    ct = pd.crosstab(annotated["turno"], annotated["classe_risco"], normalize="index")
    present = [c for c in class_order if c in ct.columns]
    ct = ct[present].reindex(["A", "B", "C"])

    fig, ax = plt.subplots(figsize=config["plots"]["figsize_default"])
    sns.heatmap(
        ct,
        annot=True,
        fmt=".1%",
        cmap="YlOrRd",
        vmin=0,
        ax=ax,
        linewidths=0.5,
        annot_kws={"size": 10},
    )
    ax.set_title(
        "Proporção de classe de risco por turno\n"
        "(normalizado por linha — remove efeito do tamanho do turno)",
        fontweight="bold",
    )
    ax.set_xlabel("Classe de risco")
    ax.set_ylabel("Turno")
    fig.tight_layout()
    return fig


def temporal_feature_preview(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Return a DataFrame showing derived temporal features — candidates for Notebook 02.

    Demonstrates hour, day-of-week, cyclical sin/cos encoding, and shift flag.
    """
    dt_col = config["temporal"].get("datetime_col", "data_hora_ocorrencia")
    sample = df.copy()
    sample[dt_col] = pd.to_datetime(sample[dt_col], utc=True)

    sample["hora"] = sample[dt_col].dt.hour
    sample["dia_semana"] = sample[dt_col].dt.dayofweek
    # Cyclical encoding: avoids discontinuity at hour 23→0
    sample["hora_sin"] = np.sin(2 * np.pi * sample["hora"] / 24).round(4)
    sample["hora_cos"] = np.cos(2 * np.pi * sample["hora"] / 24).round(4)
    sample["passagem_turno"] = (
        sample["hora"].isin([6, 7, 18, 19])
    ).astype(int)

    cols = ["id", "data_hora_ocorrencia", "turno", "hora", "dia_semana",
            "hora_sin", "hora_cos", "passagem_turno", "classe_risco"]
    cols = [c for c in cols if c in sample.columns]
    return sample[cols].head(8)


# ── 8. Unannotated opportunity ────────────────────────────────────────────────

def plot_unannotated_profile(
    df_full: pd.DataFrame,
    config: dict,
    feature: str = "area_fpso",
) -> plt.Figure:
    annotated = df_full[df_full["anotado"]].copy()
    unannotated = df_full[~df_full["anotado"]].copy()

    ann_dist = annotated[feature].value_counts(normalize=True)
    unann_dist = unannotated[feature].value_counts(normalize=True)

    top_n = 12
    top_cats = ann_dist.head(top_n).index.tolist()
    compare = pd.DataFrame({
        "anotado": ann_dist.reindex(top_cats, fill_value=0),
        "nao_anotado": unann_dist.reindex(top_cats, fill_value=0),
    })

    fig, ax = plt.subplots(figsize=config["plots"]["figsize_wide"])
    x = np.arange(len(compare))
    width = 0.35
    ax.bar(x - width / 2, compare["anotado"], width, label="Anotado",
           color="#42A5F5", edgecolor="white")
    ax.bar(x + width / 2, compare["nao_anotado"], width, label="Nao anotado",
           color="#EF5350", edgecolor="white", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(compare.index, rotation=30, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_title(
        f"Perfil de '{feature}': registros anotados vs. nao anotados\n"
        "(distribuicoes similares => viés de anotação baixo)"
    )
    ax.set_ylabel("Proporcao")
    ax.legend()
    fig.tight_layout()
    return fig


# ── 9. Split summary ──────────────────────────────────────────────────────────

def plot_split_summary(
    train: pd.DataFrame,
    test: pd.DataFrame,
    unannotated: pd.DataFrame,
    config: dict,
) -> plt.Figure:
    class_order = config["class_order"]
    palette = config["class_palette"]

    splits = {"Train": train, "Test": test}
    present = [c for c in class_order if c in train["classe_risco"].unique()]

    fig, axes = plt.subplots(1, 2, figsize=config["plots"]["figsize_wide"])

    for ax, (name, df) in zip(axes, splits.items()):
        counts = df["classe_risco"].value_counts(normalize=True).reindex(present, fill_value=0)
        colors = [palette[c] for c in present]
        ax.bar(present, counts.values, color=colors, edgecolor="white")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        ax.set_title(f"{name}  ({len(df):,} registros)")
        ax.set_xlabel("Classe")
        ax.set_ylabel("Proporcao")
        for i, v in enumerate(counts.values):
            ax.text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=8)

    fig.suptitle(
        f"Distribuicao de classes por split  |  "
        f"Unannotated: {len(unannotated):,} registros (nao usados no treino)",
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


# ── Save helper ────────────────────────────────────────────────────────────────

def save_figure(fig: plt.Figure, filename: str, config: dict, root: Path) -> Path:
    out_dir = root / config["paths"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = config["plots"].get("save_format", "png")
    dpi = config["plots"].get("dpi", 120)
    path = out_dir / f"{filename}.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
