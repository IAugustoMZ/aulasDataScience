"""
Feature analysis module for webinar_ml.

Provides statistical and visual analyses that answer the question:
'Which features — and in which form — carry predictive signal for risk class?'

All functions are pure: receive DataFrames + config dicts, return figures or DataFrames.
Side effects (display, save) happen exclusively in the notebook.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


# ── 1. Class separability — categorical features ───────────────────────────────

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    from scipy.stats import chi2_contingency
    ct = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(ct)
    n = ct.sum().sum()
    r, k = ct.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1) + 1e-9)))


def feature_importance_table(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Cramér's V for every categorical feature vs. classe_risco.

    Tells us which features carry the most associative signal — directly
    informing which columns belong in the ColumnTransformer.
    """
    from scipy.stats import chi2_contingency

    annotated = df[df["anotado"]].dropna(subset=["classe_risco"])
    features = config["feature_analysis"]["categorical_features"]
    threshold = config["feature_analysis"]["cramers_v_threshold"]
    rows = []
    for feat in features:
        if feat not in annotated.columns:
            continue
        ct = pd.crosstab(annotated[feat], annotated["classe_risco"])
        chi2, p_value, _, _ = chi2_contingency(ct)
        n = ct.sum().sum()
        r, k = ct.shape
        v = float(np.sqrt(chi2 / (n * (min(r, k) - 1) + 1e-9)))
        rows.append({
            "feature": feat,
            "cramers_v": round(v, 4),
            "p_value": round(p_value, 6),
            "significativo": p_value < 0.05,
            "sinal_forte": v >= threshold,
            "recomendacao": _recommend_encoder(feat, v, threshold),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("cramers_v", ascending=False)
        .reset_index(drop=True)
    )


def _recommend_encoder(feature: str, v: float, threshold: float) -> str:
    if v >= threshold:
        return "OneHotEncoder (sinal forte — incluir no pipeline)"
    elif v >= threshold * 0.5:
        return "OneHotEncoder (sinal moderado — incluir, monitorar)"
    else:
        return "Omitir ou BinaryEncoder (sinal fraco)"


def plot_cramers_v_ranking(df: pd.DataFrame, config: dict) -> plt.Figure:
    """Bar chart: Cramér's V por feature, ordenado, com linha de threshold."""
    table = feature_importance_table(df, config)
    threshold = config["feature_analysis"]["cramers_v_threshold"]
    palette = config["plots"].get("bar_color", "#1976D2")

    fig, ax = plt.subplots(figsize=config["plots"]["figsize_default"])
    colors = [
        "#1976D2" if v >= threshold else "#90CAF9"
        for v in table["cramers_v"]
    ]
    bars = ax.barh(table["feature"][::-1], table["cramers_v"][::-1],
                   color=colors[::-1], edgecolor="white")
    ax.axvline(threshold, color="#FF5722", linestyle="--", linewidth=1.5,
               label=f"Threshold ({threshold})")
    for bar, val in zip(bars, table["cramers_v"][::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Cramér's V")
    ax.set_title("Associação (Cramér's V) entre features categóricas e classe de risco")
    ax.legend()
    ax.set_xlim(0, max(table["cramers_v"]) * 1.2)
    fig.tight_layout()
    return fig


# ── 2. Categorical interactions ────────────────────────────────────────────────

def interaction_heatmap(
    df: pd.DataFrame,
    feature_a: str,
    feature_b: str,
    target_class: str,
    config: dict,
) -> plt.Figure:
    """Heatmap de proporção de `target_class` para cada combinação feature_a × feature_b.

    Identifica combinações (ex: área_fpso=deck_de_perfuracao + produto=h2s)
    que concentram desproporcionalmente incidentes graves — candidatos a features
    de interação no pipeline.
    """
    annotated = df[df["anotado"]].dropna(subset=["classe_risco"])
    top_a = annotated[feature_a].value_counts().head(
        config["feature_analysis"]["interaction_top_n"]
    ).index
    top_b = annotated[feature_b].value_counts().head(
        config["feature_analysis"]["interaction_top_n"]
    ).index

    sub = annotated[annotated[feature_a].isin(top_a) & annotated[feature_b].isin(top_b)]
    pivot = (
        sub.assign(_is_target=sub["classe_risco"] == target_class)
        .groupby([feature_a, feature_b])["_is_target"]
        .mean()
        .unstack(fill_value=0)
    )
    pivot = pivot.reindex(index=top_a, columns=top_b, fill_value=0)

    fig, ax = plt.subplots(figsize=config["plots"]["figsize_grid"])
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1%",
        cmap="YlOrRd",
        vmin=0,
        vmax=pivot.values.max(),
        ax=ax,
        linewidths=0.3,
        annot_kws={"size": 8},
        cbar_kws={"format": mtick.PercentFormatter(xmax=1)},
    )
    ax.set_title(
        f"Proporção de '{target_class}' por {feature_a} × {feature_b}\n"
        f"(células mais escuras = maior risco relativo)"
    )
    ax.set_xlabel(feature_b)
    ax.set_ylabel(feature_a)
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    return fig


def top_risky_combinations(
    df: pd.DataFrame,
    feature_a: str,
    feature_b: str,
    target_class: str,
    config: dict,
    top_n: int = 10,
) -> pd.DataFrame:
    """Ranking das combinações feature_a × feature_b com maior proporção de target_class."""
    annotated = df[df["anotado"]].dropna(subset=["classe_risco"])
    result = (
        annotated.assign(_is_target=annotated["classe_risco"] == target_class)
        .groupby([feature_a, feature_b])
        .agg(
            total=("_is_target", "count"),
            n_target=("_is_target", "sum"),
        )
        .assign(proporcao=lambda x: x["n_target"] / x["total"])
        .query("total >= 20")
        .sort_values("proporcao", ascending=False)
        .head(top_n)
        .reset_index()
    )
    result["proporcao"] = result["proporcao"].map("{:.1%}".format)
    return result


# ── 3. Text separability — TF-IDF + dimensionality reduction ──────────────────

def tfidf_pca_projection(
    df: pd.DataFrame,
    config: dict,
    n_components: int = 2,
    sample_n: Optional[int] = 3000,
) -> pd.DataFrame:
    """TF-IDF → SVD → 2D projection.

    Returns a DataFrame with columns [pc1, pc2, classe_risco] ready for plotting.
    Uses TruncatedSVD (LSA) — works directly on sparse TF-IDF matrix, no densification.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import Normalizer
    from sklearn.pipeline import make_pipeline

    annotated = df[df["anotado"]].dropna(subset=["classe_risco"]).copy()
    if sample_n and len(annotated) > sample_n:
        per_class = sample_n // 4
        annotated = (
            annotated.groupby("classe_risco", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), per_class), random_state=42),
                   include_groups=True)
            .reset_index(drop=True)
        )

    text_col = config["text"]["column"]
    cfg = config["feature_analysis"]["tfidf"]

    vectorizer = TfidfVectorizer(
        max_features=cfg["max_features"],
        ngram_range=tuple(cfg["ngram_range"]),
        min_df=cfg["min_df"],
        sublinear_tf=True,
    )
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    pipe = make_pipeline(vectorizer, svd, Normalizer(copy=False))

    coords = pipe.fit_transform(annotated[text_col])
    result = annotated[["classe_risco"]].copy().reset_index(drop=True)
    for i in range(n_components):
        result[f"pc{i+1}"] = coords[:, i]
    return result


def plot_tfidf_projection(df: pd.DataFrame, config: dict) -> plt.Figure:
    """Scatter 2D: projeção TF-IDF+SVD colorida por classe de risco.

    Se as classes formam clusters visíveis, BoW (TF-IDF + LogReg) vai funcionar.
    Se há sobreposição forte, precisamos de embeddings mais ricos.
    """
    proj = tfidf_pca_projection(df, config)
    class_order = config["class_order"]
    palette = config["class_palette"]

    fig, ax = plt.subplots(figsize=config["plots"]["figsize_default"])
    for cls in class_order:
        sub = proj[proj["classe_risco"] == cls]
        ax.scatter(sub["pc1"], sub["pc2"], label=cls, alpha=0.4, s=12,
                   color=palette[cls], rasterized=True)
    ax.set_xlabel("Componente 1 (LSA/SVD)")
    ax.set_ylabel("Componente 2 (LSA/SVD)")
    ax.set_title("Projeção TF-IDF → SVD 2D: separabilidade linear das classes")
    ax.legend(title="Classe", markerscale=2)
    fig.tight_layout()
    return fig


# ── 4. BERT embeddings + t-SNE ────────────────────────────────────────────────

def bert_embeddings(
    df: pd.DataFrame,
    config: dict,
    sample_n: Optional[int] = 1000,
) -> tuple[np.ndarray, pd.Series]:
    """Gera embeddings BERT para os relatos via sentence-transformers.

    Retorna (embeddings_array, labels_series) — mantido separado para permitir
    cache: embeddings são caros, t-SNE é barato e pode ser reexecutado.

    Modelo padrão: 'paraphrase-multilingual-MiniLM-L12-v2'
    — suporta português, roda em CPU em ~2min para 1k registros.
    """
    from sentence_transformers import SentenceTransformer

    annotated = df[df["anotado"]].dropna(subset=["classe_risco"]).copy()
    if sample_n and len(annotated) > sample_n:
        per_class = sample_n // 4
        annotated = (
            annotated.groupby("classe_risco", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), per_class), random_state=42),
                   include_groups=True)
            .reset_index(drop=True)
        )

    model_name = config["feature_analysis"]["bert"]["model_name"]
    model = SentenceTransformer(model_name)
    texts = annotated[config["text"]["column"]].tolist()
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True,
                              convert_to_numpy=True)
    return embeddings, annotated["classe_risco"].reset_index(drop=True)


def tsne_projection(
    embeddings: np.ndarray,
    labels: pd.Series,
    config: dict,
) -> pd.DataFrame:
    """Reduz embeddings BERT para 2D via t-SNE.

    Separa o cálculo dos embeddings da redução dimensional para permitir
    experimentar diferentes hiperparâmetros de t-SNE sem re-encodar.
    """
    from sklearn.manifold import TSNE

    tsne_cfg = config["feature_analysis"]["tsne"]
    tsne = TSNE(
        n_components=2,
        perplexity=tsne_cfg["perplexity"],
        max_iter=tsne_cfg["n_iter"],
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(embeddings)
    return pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "classe_risco": labels.values,
    })


def plot_tsne(tsne_df: pd.DataFrame, config: dict, title_suffix: str = "") -> plt.Figure:
    """Scatter t-SNE colorido por classe.

    Clusters densos e separados = embeddings carregam sinal semântico além do BoW.
    Sobreposição nas bordas alto/critico é esperada — é onde o problema é difícil.
    """
    class_order = config["class_order"]
    palette = config["class_palette"]

    fig, ax = plt.subplots(figsize=config["plots"]["figsize_default"])
    for cls in class_order:
        sub = tsne_df[tsne_df["classe_risco"] == cls]
        ax.scatter(sub["x"], sub["y"], label=cls, alpha=0.5, s=14,
                   color=palette[cls], rasterized=True)
    title = "t-SNE sobre embeddings BERT: estrutura semântica das classes"
    if title_suffix:
        title += f"\n{title_suffix}"
    ax.set_title(title)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(title="Classe", markerscale=2)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def plot_tfidf_vs_bert_side_by_side(
    df: pd.DataFrame,
    tsne_df: pd.DataFrame,
    config: dict,
) -> plt.Figure:
    """Painel lado a lado: TF-IDF+SVD (esquerda) vs. BERT+t-SNE (direita).

    Visualização pedagógica central: mostra o ganho de representação semântica
    em relação ao modelo de bag-of-words — justifica por que embeddings pré-treinados
    existem nos notebooks seguintes.
    """
    proj = tfidf_pca_projection(df, config, sample_n=len(tsne_df))
    class_order = config["class_order"]
    palette = config["class_palette"]

    fig, axes = plt.subplots(1, 2, figsize=config["plots"]["figsize_wide"])

    for cls in class_order:
        sub = proj[proj["classe_risco"] == cls]
        axes[0].scatter(sub["pc1"], sub["pc2"], label=cls, alpha=0.4, s=10,
                        color=palette[cls], rasterized=True)
    axes[0].set_title("TF-IDF → SVD\n(bag-of-words)")
    axes[0].set_xlabel("Componente 1")
    axes[0].set_ylabel("Componente 2")
    axes[0].legend(title="Classe", markerscale=2, fontsize=8)

    for cls in class_order:
        sub = tsne_df[tsne_df["classe_risco"] == cls]
        axes[1].scatter(sub["x"], sub["y"], label=cls, alpha=0.5, s=10,
                        color=palette[cls], rasterized=True)
    axes[1].set_title("BERT embeddings → t-SNE\n(semântica contextual)")
    axes[1].set_xlabel("t-SNE dim 1")
    axes[1].set_ylabel("t-SNE dim 2")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].legend(title="Classe", markerscale=2, fontsize=8)

    fig.suptitle("Representação de texto: BoW vs. semântica contextual", fontweight="bold")
    fig.tight_layout()
    return fig


# ── 5. Temporal features ───────────────────────────────────────────────────────

def temporal_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Deriva features temporais a partir do timestamp completo.

    Retorna o DataFrame enriquecido com:
    - hora, dia_semana, mes
    - hora_sin, hora_cos  (encoding cíclico)
    - passagem_turno      (flag para janelas de handover)
    - turno_noturno       (flag para turno B / horário 19h–07h)
    """
    dt_col = config["temporal"]["datetime_col"]
    handover = config["feature_analysis"]["temporal"]["handover_windows"]

    out = df.copy()
    ts = pd.to_datetime(out[dt_col], utc=True).dt.tz_convert("America/Sao_Paulo")

    out["hora"] = ts.dt.hour
    out["dia_semana"] = ts.dt.dayofweek
    out["mes"] = ts.dt.month
    out["hora_sin"] = np.sin(2 * np.pi * out["hora"] / 24)
    out["hora_cos"] = np.cos(2 * np.pi * out["hora"] / 24)
    out["passagem_turno"] = out["hora"].apply(
        lambda h: any(lo <= h <= hi for lo, hi in handover)
    )
    out["turno_noturno"] = out["hora"].apply(lambda h: h >= 19 or h < 7)
    return out


def temporal_signal_table(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Tabela: por feature temporal derivada, proporção de cada classe.

    Mostra quais features temporais têm distribuição de classe diferente da média —
    evidência de sinal preditivo antes de qualquer modelo.
    """
    enriched = temporal_features(df, config)
    annotated = enriched[enriched["anotado"]].dropna(subset=["classe_risco"])
    class_order = config["class_order"]

    features = ["passagem_turno", "turno_noturno"]
    rows = []
    global_dist = annotated["classe_risco"].value_counts(normalize=True)

    for feat in features:
        for val in [True, False]:
            sub = annotated[annotated[feat] == val]
            if len(sub) < 50:
                continue
            dist = sub["classe_risco"].value_counts(normalize=True).reindex(class_order).fillna(0)
            row = {"feature": feat, "valor": str(val), "n": len(sub)}
            for cls in class_order:
                row[cls] = f"{dist[cls]:.1%}"
                row[f"delta_{cls}"] = round(dist[cls] - global_dist.get(cls, 0), 4)
            rows.append(row)
    return pd.DataFrame(rows)


def plot_temporal_features_vs_class(df: pd.DataFrame, config: dict) -> plt.Figure:
    """2×1: taxa de 'critico' e 'alto' por hora do dia, com encoding cíclico sobreposto."""
    enriched = temporal_features(df, config)
    annotated = enriched[enriched["anotado"]].dropna(subset=["classe_risco"])
    palette = config["class_palette"]
    handover = config["feature_analysis"]["temporal"]["handover_windows"]

    hourly = (
        annotated.groupby("hora")["classe_risco"]
        .value_counts(normalize=False)
        .unstack(fill_value=0)
    )
    hourly_pct = hourly.div(hourly.sum(axis=1), axis=0)

    fig, axes = plt.subplots(2, 1, figsize=config["plots"]["figsize_tall"], sharex=True)

    for cls in ["critico", "alto"]:
        if cls in hourly_pct.columns:
            axes[0].plot(hourly_pct.index, hourly_pct[cls], label=cls,
                         color=palette[cls], linewidth=2)
    for lo, hi in handover:
        axes[0].axvspan(lo, hi, alpha=0.12, color="#FF5722", label="_nolegend_")
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    axes[0].set_ylabel("Proporção da hora")
    axes[0].set_title("Taxa de incidentes alto+crítico por hora do dia")
    axes[0].legend()

    hours = np.linspace(0, 23, 100)
    axes[1].plot(hours, np.sin(2 * np.pi * hours / 24), label="hora_sin", color="#1976D2")
    axes[1].plot(hours, np.cos(2 * np.pi * hours / 24), label="hora_cos",
                 color="#1976D2", linestyle="--")
    axes[1].set_xlabel("Hora do dia")
    axes[1].set_ylabel("Valor")
    axes[1].set_title("Encoding cíclico: sin(2π·h/24) e cos(2π·h/24)")
    axes[1].legend()
    axes[1].set_xticks(range(0, 24, 2))

    fig.tight_layout()
    return fig


# ── 6. Statistical hypothesis tests ───────────────────────────────────────────

def hypothesis_tests_summary(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Tabela de testes de hipótese para cada feature vs. classe de risco.

    - Texto (comprimento): Kruskal-Wallis entre classes (não assume normalidade)
    - Features categóricas: chi² de independência
    - Features temporais binárias: chi² de independência

    Retorna p-value, estatística do teste, e interpretação direta.
    """
    annotated = df[df["anotado"]].dropna(subset=["classe_risco"]).copy()
    class_order = config["class_order"]
    text_col = config["text"]["column"]
    rows = []

    # Kruskal-Wallis: comprimento do relato entre classes
    annotated["_n_words"] = annotated[text_col].str.split().str.len()
    groups = [annotated[annotated["classe_risco"] == c]["_n_words"].dropna()
              for c in class_order if c in annotated["classe_risco"].values]
    if len(groups) >= 2:
        stat, p = stats.kruskal(*groups)
        rows.append({
            "feature": "comprimento_relato",
            "teste": "Kruskal-Wallis",
            "estatistica": round(stat, 2),
            "p_value": round(p, 6),
            "significativo": p < 0.05,
            "interpretacao": "Comprimento difere entre classes" if p < 0.05
                             else "Comprimento NÃO difere entre classes",
        })

    # Chi² para features categóricas e binárias
    test_features = (
        config["feature_analysis"]["categorical_features"]
        + ["passagem_turno", "turno_noturno"]
    )
    enriched = temporal_features(annotated, config)
    for feat in test_features:
        if feat not in enriched.columns:
            continue
        ct = pd.crosstab(enriched[feat], enriched["classe_risco"])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        rows.append({
            "feature": feat,
            "teste": "chi²",
            "estatistica": round(chi2, 2),
            "p_value": round(p, 6),
            "significativo": p < 0.05,
            "interpretacao": f"Associação com classe (dof={dof})" if p < 0.05
                             else "Sem associação significativa",
        })

    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


def plot_text_length_by_class_boxplot(df: pd.DataFrame, config: dict) -> plt.Figure:
    """Boxplot do comprimento do relato por classe — apoia o teste Kruskal-Wallis."""
    annotated = df[df["anotado"]].dropna(subset=["classe_risco"]).copy()
    text_col = config["text"]["column"]
    class_order = config["class_order"]
    palette = config["class_palette"]

    annotated["n_palavras"] = annotated[text_col].str.split().str.len()
    data_by_class = [
        annotated[annotated["classe_risco"] == c]["n_palavras"].dropna()
        for c in class_order
    ]

    fig, ax = plt.subplots(figsize=config["plots"]["figsize_default"])
    bp = ax.boxplot(data_by_class, labels=class_order, patch_artist=True,
                    medianprops={"color": "white", "linewidth": 2})
    for patch, cls in zip(bp["boxes"], class_order):
        patch.set_facecolor(palette[cls])
        patch.set_alpha(0.7)
    ax.set_xlabel("Classe de risco")
    ax.set_ylabel("Número de palavras")
    ax.set_title("Distribuição do comprimento do relato por classe\n"
                 "(Kruskal-Wallis testa se as medianas são iguais)")
    fig.tight_layout()
    return fig


# ── 7. Pipeline specification ──────────────────────────────────────────────────

def pipeline_spec_table(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Tabela final: feature → transformador recomendado para o ColumnTransformer.

    É a saída operacional deste notebook: especifica o que construir no Notebook 03.
    """
    importance = feature_importance_table(df, config)
    threshold = config["feature_analysis"]["cramers_v_threshold"]

    rows = []

    # Texto
    rows.append({
        "feature": "relato",
        "tipo": "texto",
        "transformador": "TfidfVectorizer(sublinear_tf=True, ngram_range=(1,2))",
        "motivo": "Feature principal — vocabulário diverge por classe (visto nos tokens)",
        "prioridade": 1,
    })
    rows.append({
        "feature": "relato",
        "tipo": "texto (avançado)",
        "transformador": "SentenceTransformer → vetor denso 384d",
        "motivo": "t-SNE mostrou clusters semânticos não capturados pelo BoW",
        "prioridade": 2,
    })

    # Features categóricas com sinal
    for _, row in importance.iterrows():
        prioridade = 3 if row["sinal_forte"] else 5
        rows.append({
            "feature": row["feature"],
            "tipo": "categórica",
            "transformador": "OneHotEncoder(handle_unknown='ignore')"
                             if row["sinal_forte"]
                             else "Omitir (baixo sinal)",
            "motivo": f"Cramér's V = {row['cramers_v']:.3f} "
                      f"({'≥' if row['sinal_forte'] else '<'} threshold {threshold})",
            "prioridade": prioridade,
        })

    # Features temporais
    rows.append({
        "feature": "hora_sin, hora_cos",
        "tipo": "temporal (cíclica)",
        "transformador": "FunctionTransformer (derivar do timestamp)",
        "motivo": "Encoding cíclico preserva continuidade hora 23→0; pico crítico em passagem de turno",
        "prioridade": 3,
    })
    rows.append({
        "feature": "passagem_turno",
        "tipo": "temporal (binária)",
        "transformador": "FunctionTransformer → bool → int",
        "motivo": "Janelas 06h-07h / 18h-19h concentram maior proporção de crítico",
        "prioridade": 3,
    })

    return (
        pd.DataFrame(rows)
        .sort_values("prioridade")
        .reset_index(drop=True)
    )
