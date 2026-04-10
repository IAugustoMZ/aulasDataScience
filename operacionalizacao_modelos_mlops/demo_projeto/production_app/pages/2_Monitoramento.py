"""
pages/2_Monitoramento.py — Dashboard Streamlit: monitoramento de modelo em produção.

Simula monitoramento de inferência em lote:
  1. Carrega N amostras do parquet de features.
  2. Realiza predição em lote via modelo MLflow local (sem servidor REST).
  3. Divide as predições em K lotes sequenciais para simular ingestão temporal.
  4. Calcula métricas por lote: RMSE, MAE, R², MAPE.
  5. Visualiza:
       - Séries temporais de cada métrica com média móvel (janela configurável).
       - Histograma de resíduos + KDE.
       - Scatter de real vs previsto.
       - Tabela de métricas por lote (expansível).

As métricas monitoradas espelham modeling.yaml → metrics:
  primary   : rmse
  additional: mae, r2, mape
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)

# ── Bootstrap de paths ────────────────────────────────────────────────────────
_PAGE_DIR     = Path(__file__).resolve().parent   # pages/
_APP_DIR      = _PAGE_DIR.parent                  # production_app/
_PROJECT_ROOT = _APP_DIR.parent                   # demo_projeto/

for _p in [str(_APP_DIR), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.pipeline_utils import (
    obter_parquet_features,
    obter_colunas_features_brutas,
    _TARGET_COL,
)
from utils.model_utils import carregar_modelo, prever_lote

# ─────────────────────────────────────────────────────────────────────────────
# Configuração da página
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Monitoramento do Modelo",
    page_icon="📡",
    layout="wide",
)

st.title("📡 Dashboard de Monitoramento do Modelo")
st.markdown(
    """
    Simula **monitoramento de produção em lote**: amostra N pontos do parquet de features,
    executa o modelo sobre eles e calcula métricas ao longo de K lotes sequenciais.
    Use para detectar drift, degradação ou vieses sistemáticos ao longo do tempo.
    O modelo é carregado diretamente do **banco SQLite do MLflow** — sem servidor externo.
    """
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — configurações
# ─────────────────────────────────────────────────────────────────────────────
_URI_PADRAO = f"sqlite:///{_PROJECT_ROOT / 'mlruns.db'}"

with st.sidebar:
    st.header("⚙️ Configurações")
    db_uri = st.text_input(
        "URI do banco SQLite",
        value=_URI_PADRAO,
        help="URI do banco SQLite gerado por modelagem.py.",
    )
    n_amostras = st.slider(
        "Total de amostras",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
        help="Número de pontos a amostrar do parquet de features.",
    )
    n_lotes = st.slider(
        "Número de lotes",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        help="Divide o total de amostras em N lotes sequenciais.",
    )
    janela_movel = st.slider(
        "Janela de média móvel",
        min_value=2,
        max_value=10,
        value=3,
        help="Tamanho da janela para a média móvel nos gráficos de série temporal.",
    )
    semente = st.number_input("Semente aleatória", value=42, step=1)

# ─────────────────────────────────────────────────────────────────────────────
# Cache do modelo
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Carregando modelo do MLflow...")
def _modelo_em_cache(uri: str):
    """Carrega e armazena em cache o modelo MLflow para evitar recarregamentos."""
    return carregar_modelo(uri)


# ─────────────────────────────────────────────────────────────────────────────
# Funções auxiliares
# ─────────────────────────────────────────────────────────────────────────────

def _calcular_metricas(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula RMSE, MAE, R² e MAPE para um par de arrays."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
    }


def _formatar_usd(x, _):
    """Formatador de eixo Matplotlib para valores em USD."""
    if abs(x) >= 1_000:
        return f"${x / 1_000:.0f}k"
    return f"${x:.0f}"


def _plotar_serie_temporal(
    ax: plt.Axes,
    lotes: list[int],
    valores: list[float],
    movel: pd.Series,
    nome_metrica: str,
    cor: str,
    eh_usd: bool = False,
) -> None:
    """Plota métrica por lote + média móvel em um eixo Matplotlib."""
    ax.plot(lotes, valores, "o-", color=cor, alpha=0.5, linewidth=1.5,
            markersize=4, label="Por lote")
    ax.plot(lotes, movel, "-", color=cor, linewidth=2.5,
            label=f"Média móvel (j={janela_movel})")
    ax.set_xlabel("Lote #", fontsize=9)
    ax.set_title(nome_metrica.upper(), fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    if eh_usd:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_formatar_usd))

    # Faixa de ±1 desvio padrão em torno da média móvel
    arr_movel = movel.values.astype(float)
    desvio = np.nanstd(valores)
    ax.fill_between(
        lotes,
        arr_movel - desvio,
        arr_movel + desvio,
        color=cor,
        alpha=0.08,
        label="±1 dp",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Execução do monitoramento
# ─────────────────────────────────────────────────────────────────────────────
btn_executar = st.button(
    "▶️ Executar Análise de Monitoramento",
    type="primary",
    use_container_width=True,
)

if btn_executar:

    # ── Carregamento dos dados de features ────────────────────────────────────
    with st.spinner("Carregando parquet de features..."):
        try:
            df_features = obter_parquet_features()
        except Exception as exc:
            st.error(f"❌ Erro ao carregar o parquet de features: {exc}")
            st.stop()

    # ── Amostragem e separação X / y ──────────────────────────────────────────
    rng = np.random.default_rng(int(semente))
    idx_amostra = rng.choice(
        len(df_features),
        size=min(n_amostras, len(df_features)),
        replace=False,
    )
    df_amostra = df_features.iloc[idx_amostra].reset_index(drop=True)

    y_true_total = df_amostra[_TARGET_COL].values

    # Selecionar colunas de features (excluindo target)
    colunas_brutas = obter_colunas_features_brutas()
    colunas_disponiveis = [c for c in colunas_brutas if c in df_amostra.columns]
    colunas_ausentes    = [c for c in colunas_brutas if c not in df_amostra.columns]

    if colunas_ausentes:
        st.warning(f"⚠️ Colunas ausentes no parquet (ignoradas): {colunas_ausentes}")

    X_amostra = df_amostra[colunas_disponiveis].copy()

    # Sanitizar nomes para XGBoost (< → lt_, [ → (, ] → ))
    mapa_rename = {
        c: c.replace("<", "lt_").replace("[", "(").replace("]", ")")
        for c in X_amostra.columns
        if any(ch in c for ch in ("<", "[", "]"))
    }
    if mapa_rename:
        X_amostra = X_amostra.rename(columns=mapa_rename)

    # ── Predição em lote via modelo MLflow ────────────────────────────────────
    with st.spinner(f"Realizando predição em lote ({len(X_amostra)} linhas)..."):
        try:
            modelo = _modelo_em_cache(db_uri)
            y_pred_total = np.array(prever_lote(X_amostra, modelo))
        except Exception as exc:
            st.error(f"❌ Erro na predição em lote: {exc}")
            st.stop()

    # ── Métricas por lote ─────────────────────────────────────────────────────
    tamanho_lote = len(X_amostra) // n_lotes
    resto        = len(X_amostra) % n_lotes

    metricas_lotes: list[dict] = []
    for i in range(n_lotes):
        inicio = i * tamanho_lote
        fim    = inicio + tamanho_lote + (1 if i < resto else 0)
        if fim > len(X_amostra):
            break
        m = _calcular_metricas(y_true_total[inicio:fim], y_pred_total[inicio:fim])
        m["lote"] = i + 1
        metricas_lotes.append(m)

    df_metricas = pd.DataFrame(metricas_lotes).set_index("lote")
    df_movel    = df_metricas.rolling(window=janela_movel, min_periods=1).mean()
    geral       = _calcular_metricas(y_true_total[:len(y_pred_total)], y_pred_total)

    # ═══════════════════════════════════════════════════════════════════════════
    # SEÇÃO 1 — KPIs gerais
    # ═══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader(f"Métricas Gerais ({len(X_amostra)} amostras)")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("RMSE",  f"${geral['rmse']:,.0f}")
    kpi2.metric("MAE",   f"${geral['mae']:,.0f}")
    kpi3.metric("R²",    f"{geral['r2']:.4f}")
    kpi4.metric("MAPE",  f"{geral['mape']:.2f}%")

    # ═══════════════════════════════════════════════════════════════════════════
    # SEÇÃO 2 — Séries temporais por lote
    # ═══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("📈 Métricas por Lote ao Longo do Tempo")
    st.caption(
        f"{len(metricas_lotes)} lotes × ~{tamanho_lote} amostras cada  |  "
        f"Janela de média móvel = {janela_movel} lotes"
    )

    lotes   = df_metricas.index.tolist()
    paleta  = {"rmse": "#e74c3c", "mae": "#e67e22", "r2": "#27ae60", "mape": "#2980b9"}

    fig_ts, eixos_ts = plt.subplots(2, 2, figsize=(14, 7), tight_layout=True)
    fig_ts.patch.set_facecolor("#0e1117")

    pares_metrica = [
        ("rmse", eixos_ts[0, 0], True),
        ("mae",  eixos_ts[0, 1], True),
        ("r2",   eixos_ts[1, 0], False),
        ("mape", eixos_ts[1, 1], False),
    ]

    for metrica, ax, eh_usd in pares_metrica:
        ax.set_facecolor("#1a1a2e")
        _plotar_serie_temporal(
            ax=ax,
            lotes=lotes,
            valores=df_metricas[metrica].tolist(),
            movel=df_movel[metrica],
            nome_metrica=metrica,
            cor=paleta[metrica],
            eh_usd=eh_usd,
        )
        for borda in ax.spines.values():
            borda.set_edgecolor("#333")
        ax.tick_params(colors="white")
        ax.title.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

    st.pyplot(fig_ts, use_container_width=True)
    plt.close(fig_ts)

    # ═══════════════════════════════════════════════════════════════════════════
    # SEÇÃO 3 — Distribuição de resíduos
    # ═══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("📊 Distribuição dos Resíduos")
    st.caption(
        "Resíduos = y_real − y_previsto. "
        "Um modelo bem calibrado apresenta resíduos centrados em 0, sem assimetria acentuada."
    )

    residuos = y_true_total[:len(y_pred_total)] - y_pred_total

    fig_hist, eixos_hist = plt.subplots(1, 2, figsize=(14, 4), tight_layout=True)
    fig_hist.patch.set_facecolor("#0e1117")

    # Histograma + KDE
    ax_hist = eixos_hist[0]
    ax_hist.set_facecolor("#1a1a2e")
    sns.histplot(
        residuos,
        bins=30,
        kde=True,
        color="#e74c3c",
        ax=ax_hist,
        line_kws={"linewidth": 2},
    )
    ax_hist.axvline(0, color="white", linestyle="--", linewidth=1.2, label="Erro zero")
    ax_hist.axvline(
        float(np.mean(residuos)),
        color="#f1c40f",
        linestyle="-.",
        linewidth=1.5,
        label=f"Resíduo médio: ${np.mean(residuos):,.0f}",
    )
    ax_hist.set_title("Histograma de Resíduos + KDE", color="white", fontsize=11, fontweight="bold")
    ax_hist.set_xlabel("Resíduo (USD)", color="white")
    ax_hist.set_ylabel("Contagem", color="white")
    ax_hist.tick_params(colors="white")
    for borda in ax_hist.spines.values():
        borda.set_edgecolor("#333")
    ax_hist.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    ax_hist.xaxis.set_major_formatter(mticker.FuncFormatter(_formatar_usd))

    # Scatter real vs previsto
    ax_scatter = eixos_hist[1]
    ax_scatter.set_facecolor("#1a1a2e")
    ax_scatter.scatter(
        y_true_total[:len(y_pred_total)],
        y_pred_total,
        alpha=0.35,
        s=12,
        color="#3498db",
        edgecolors="none",
    )
    val_min = min(y_true_total.min(), y_pred_total.min())
    val_max = max(y_true_total.max(), y_pred_total.max())
    ax_scatter.plot(
        [val_min, val_max],
        [val_min, val_max],
        "r--",
        linewidth=1.5,
        label="Predição perfeita",
    )
    ax_scatter.set_title("Real vs Previsto", color="white", fontsize=11, fontweight="bold")
    ax_scatter.set_xlabel("Real (USD)", color="white")
    ax_scatter.set_ylabel("Previsto (USD)", color="white")
    ax_scatter.tick_params(colors="white")
    for borda in ax_scatter.spines.values():
        borda.set_edgecolor("#333")
    ax_scatter.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    ax_scatter.xaxis.set_major_formatter(mticker.FuncFormatter(_formatar_usd))
    ax_scatter.yaxis.set_major_formatter(mticker.FuncFormatter(_formatar_usd))

    st.pyplot(fig_hist, use_container_width=True)
    plt.close(fig_hist)

    # ═══════════════════════════════════════════════════════════════════════════
    # SEÇÃO 4 — Tabela de métricas por lote
    # ═══════════════════════════════════════════════════════════════════════════
    with st.expander("📋 Tabela de métricas por lote"):
        df_exibir = df_metricas.copy()
        df_exibir["rmse"] = df_exibir["rmse"].map("${:,.0f}".format)
        df_exibir["mae"]  = df_exibir["mae"].map("${:,.0f}".format)
        df_exibir["r2"]   = df_exibir["r2"].map("{:.4f}".format)
        df_exibir["mape"] = df_exibir["mape"].map("{:.2f}%".format)
        df_exibir.columns = ["RMSE", "MAE", "R²", "MAPE"]
        st.dataframe(df_exibir, use_container_width=True)
