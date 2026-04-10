"""
pages/1_Predicao.py — Interface Streamlit: predição de preço imobiliário.

O usuário insere as 9 features originais do California Housing.
Ao submeter:
  1. A cadeia completa de pré-processamento (utils/pipeline_utils.py) converte as
     entradas brutas nas ~28 features engenheiradas que o modelo espera.
  2. O modelo é carregado diretamente do banco SQLite do MLflow (sem servidor REST).
  3. As métricas de IC (cv_rmse_std, holdout_rmse) são recuperadas via MlflowClient.
  4. O IC de 95% é calculado como: y_hat ± 1,96 × (cv_rmse_std / √n_folds)
  5. Os resultados são exibidos com gauge visual e barra de intervalo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# ── Bootstrap de paths ────────────────────────────────────────────────────────
_PAGE_DIR     = Path(__file__).resolve().parent   # pages/
_APP_DIR      = _PAGE_DIR.parent                  # production_app/
_PROJECT_ROOT = _APP_DIR.parent                   # demo_projeto/

for _p in [str(_APP_DIR), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.pipeline_utils import preprocessar_entradas
from utils.model_utils import (
    carregar_modelo,
    prever_individual,
    obter_params_ic,
    calcular_intervalo_confianca,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuração da página
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predição de Preço Imobiliário",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 Predição de Preço Imobiliário — Califórnia")
st.markdown(
    """
    Informe as features originais do bloco residencial abaixo.
    A aplicação executa o **pipeline completo de feature engineering**
    (razões, transformações log, distâncias geográficas, encoding) e
    carrega o modelo diretamente do **banco SQLite do MLflow** — sem servidor externo.
    """
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — configuração do banco MLflow
# ─────────────────────────────────────────────────────────────────────────────
_URI_PADRAO = f"sqlite:///{_PROJECT_ROOT / 'mlruns.db'}"

with st.sidebar:
    st.header("⚙️ Configurações MLflow")
    db_uri = st.text_input(
        "URI do banco SQLite",
        value=_URI_PADRAO,
        help=(
            "URI do banco SQLite gerado por modelagem.py.\n"
            "Exemplos:\n"
            "  sqlite:///mlruns.db\n"
            "  sqlite:////caminho/absoluto/mlruns.db"
        ),
    )

    st.divider()
    st.markdown(
        """
        **Como gerar o banco:**
        ```bash
        cd demo_projeto
        python notebooks/ingestao.py
        python notebooks/preprocessamento.py
        python notebooks/modelagem.py
        ```
        Depois execute a aplicação:
        ```bash
        streamlit run production_app/app.py
        ```
        """
    )

# ─────────────────────────────────────────────────────────────────────────────
# Cache do modelo (recarregado apenas quando o URI muda)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Carregando modelo do MLflow...")
def _modelo_em_cache(uri: str):
    """Carrega e armazena em cache o modelo MLflow para evitar recarregamentos."""
    return carregar_modelo(uri)


# ─────────────────────────────────────────────────────────────────────────────
# Formulário de entrada — apenas as 9 features originais
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Features do bloco residencial")
st.caption(
    "Todas as features derivadas (log, distâncias geo, razões, polinomiais, encoding) "
    "são calculadas automaticamente pelo pipeline de pré-processamento."
)

col1, col2, col3 = st.columns(3)

with col1:
    median_income = st.number_input(
        "Renda mediana (dezenas de milhares USD)",
        min_value=0.1,
        max_value=20.0,
        value=4.5,
        step=0.1,
        help="Ex: 4.5 significa renda mediana de $45.000",
    )
    housing_median_age = st.number_input(
        "Idade mediana das residências (anos)",
        min_value=1,
        max_value=52,
        value=25,
        step=1,
        help="Idade mediana das casas no bloco. Máx = 52 (censurado).",
    )
    ocean_proximity = st.selectbox(
        "Proximidade ao oceano",
        options=["<1H OCEAN", "INLAND", "NEAR BAY", "NEAR OCEAN", "ISLAND"],
        index=0,
        help="Localização categórica em relação ao oceano.",
    )

with col2:
    total_rooms = st.number_input(
        "Total de cômodos",
        min_value=1,
        max_value=40_000,
        value=2_500,
        step=50,
        help="Número total de cômodos no bloco.",
    )
    total_bedrooms = st.number_input(
        "Total de quartos",
        min_value=0,
        max_value=7_000,
        value=500,
        step=10,
        help="Total de quartos. Informe 0 para deixar o pipeline imputar.",
    )
    households = st.number_input(
        "Domicílios",
        min_value=1,
        max_value=7_000,
        value=450,
        step=10,
        help="Número de domicílios no bloco.",
    )

with col3:
    population = st.number_input(
        "População",
        min_value=1,
        max_value=40_000,
        value=1_200,
        step=50,
        help="População total do bloco.",
    )
    latitude = st.number_input(
        "Latitude",
        min_value=32.0,
        max_value=42.0,
        value=37.75,
        step=0.01,
        format="%.4f",
        help="Latitude do bloco (Califórnia: ~32°N – 42°N).",
    )
    longitude = st.number_input(
        "Longitude",
        min_value=-125.0,
        max_value=-114.0,
        value=-122.42,
        step=0.01,
        format="%.4f",
        help="Longitude do bloco (Califórnia: ~-114° – -125°).",
    )

# ─────────────────────────────────────────────────────────────────────────────
# Predição
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
btn_prever = st.button(
    "🔮 Calcular Preço Imobiliário",
    type="primary",
    use_container_width=True,
)

if btn_prever:
    entradas_brutas = {
        "median_income":      median_income,
        "housing_median_age": housing_median_age,
        "total_rooms":        total_rooms,
        "total_bedrooms":     float(total_bedrooms) if total_bedrooms > 0 else float("nan"),
        "population":         population,
        "households":         households,
        "latitude":           latitude,
        "longitude":          longitude,
        "ocean_proximity":    ocean_proximity,
    }

    # ── Passo 1: pipeline de pré-processamento ────────────────────────────────
    with st.spinner("Executando pipeline de pré-processamento..."):
        try:
            features_df = preprocessar_entradas(entradas_brutas)
            pipeline_ok = True
        except Exception as exc:
            st.error(f"❌ Erro no pré-processamento: {exc}")
            pipeline_ok = False

    # ── Passo 2: carregamento do modelo e predição ────────────────────────────
    if pipeline_ok:
        with st.spinner("Carregando modelo e realizando predição..."):
            try:
                modelo = _modelo_em_cache(db_uri)
                y_hat  = prever_individual(features_df, modelo)
                predicao_ok = True
            except Exception as exc:
                st.error(
                    f"❌ Erro ao carregar o modelo ou realizar predição: {exc}\n\n"
                    f"Verifique se o banco SQLite está em: `{db_uri}`"
                )
                predicao_ok = False

    # ── Passo 3: parâmetros de IC via MLflow ──────────────────────────────────
    if pipeline_ok and predicao_ok:
        with st.spinner("Recuperando parâmetros de IC do MLflow..."):
            try:
                params_ic = obter_params_ic(db_uri)
                inferior, superior = calcular_intervalo_confianca(
                    y_hat=y_hat,
                    cv_rmse_std=params_ic["cv_rmse_std"],
                )
                ic_ok = True
            except Exception as exc:
                st.warning(
                    f"⚠️ Não foi possível recuperar o IC do MLflow: {exc}\n\n"
                    "A predição pontual é exibida abaixo sem intervalo de confiança."
                )
                ic_ok = False

    # ── Passo 4: exibição dos resultados ──────────────────────────────────────
    if pipeline_ok and predicao_ok:
        st.divider()
        st.subheader("📊 Resultados da Predição")

        col_res1, col_res2, col_res3 = st.columns([2, 1, 1])

        with col_res1:
            st.metric(
                label="Valor Imobiliário Mediano Previsto",
                value=f"${y_hat:,.0f}",
                help="Estimativa pontual do modelo carregado do MLflow.",
            )
            if ic_ok:
                from utils.model_utils import _N_FOLDS_CV
                st.markdown(
                    f"""
                    **Intervalo de Confiança de 95%:**
                    &nbsp;&nbsp; ${inferior:,.0f} &nbsp; – &nbsp; ${superior:,.0f}

                    *EP = cv\\_rmse\\_std / √{_N_FOLDS_CV} =
                    {params_ic["cv_rmse_std"]:,.0f} / √{_N_FOLDS_CV} =
                    {params_ic["cv_rmse_std"] / (_N_FOLDS_CV ** 0.5):,.0f}*
                    """
                )

                # Barra visual do IC
                amplitude = superior - inferior
                st.progress(
                    min(int((y_hat - inferior) / (amplitude + 1e-9) * 100), 100),
                    text=f"Predição dentro do intervalo  |  Amplitude: ${amplitude:,.0f}",
                )

        with col_res2:
            if ic_ok:
                st.metric("Limite inferior (IC 95%)", f"${inferior:,.0f}")
                st.metric("Limite superior (IC 95%)", f"${superior:,.0f}")

        with col_res3:
            if ic_ok:
                st.metric("RMSE holdout", f"${params_ic['holdout_rmse']:,.0f}")
                st.metric("cv_rmse_std", f"${params_ic['cv_rmse_std']:,.0f}")
                st.caption(f"Versão do modelo: {params_ic['versao_modelo']}")
                st.caption(f"Run ID: `{params_ic['run_id'][:8]}...`")

        # ── Inspeção das features engenheiradas ───────────────────────────────
        with st.expander("🔍 Inspecionar features engenheiradas"):
            st.caption(
                f"{len(features_df.columns)} features enviadas ao modelo "
                f"(9 originais → {len(features_df.columns)} após pipeline completo)"
            )
            st.dataframe(
                features_df.T.rename(columns={0: "valor"}).style.format("{:.4f}"),
                use_container_width=True,
                height=600,
            )
