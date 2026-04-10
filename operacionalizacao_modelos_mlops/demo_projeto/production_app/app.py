"""
production_app/app.py — Ponto de entrada da aplicação Streamlit.

Redireciona automaticamente para a página de Predição ao abrir a aplicação.

Como executar:
    cd demo_projeto
    streamlit run production_app/app.py

Páginas disponíveis:
    1_Predicao.py      — Predição individual de preço com IC 95%
    2_Monitoramento.py — Dashboard de monitoramento por lotes
"""
import streamlit as st

st.switch_page("pages/1_Predicao.py")
