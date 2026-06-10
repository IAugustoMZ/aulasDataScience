"""
Constrói o ColumnTransformer a partir de params.yaml.

Design: a spec das features vive no config, não no código.
Trocar features = editar params.yaml, não este arquivo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def _add_temporal_features(X: pd.DataFrame) -> pd.DataFrame:
    """Deriva features temporais a partir de data_hora_ocorrencia."""
    out = X.copy()
    if "data_hora_ocorrencia" in out.columns:
        ts = pd.to_datetime(out["data_hora_ocorrencia"], utc=True).dt.tz_convert("America/Sao_Paulo")
        hora = ts.dt.hour
        out["hora_sin"] = np.sin(2 * np.pi * hora / 24)
        out["hora_cos"] = np.cos(2 * np.pi * hora / 24)
        # janelas de passagem de turno: 06h-07h e 18h-19h
        out["passagem_turno"] = hora.apply(lambda h: int(6 <= h <= 7 or 18 <= h <= 19))
    return out


def _word_count(X: pd.DataFrame) -> np.ndarray:
    """Conta palavras no relato — retorna array 2D para StandardScaler."""
    col = X.iloc[:, 0] if isinstance(X, pd.DataFrame) else pd.Series(X.ravel())
    return col.str.split().str.len().fillna(0).values.reshape(-1, 1)


def build_feature_pipeline(params: dict) -> Pipeline:
    """Retorna um Pipeline sklearn com ColumnTransformer + preprocessamento.

    O Pipeline expõe fit/transform/fit_transform — pode ser usado diretamente
    ou embutido num Pipeline maior com o classificador.

    Args:
        params: dict com chave 'features' (seção features do params.yaml).
    """
    cfg = params["features"]
    text_col = cfg["text_col"]
    cat_cols = cfg["categorical"]
    num_cols = cfg["numeric"]

    tfidf_cfg = cfg["tfidf"]

    tfidf = TfidfVectorizer(
        max_features=tfidf_cfg["max_features"],
        ngram_range=tuple(tfidf_cfg["ngram_range"]),
        min_df=tfidf_cfg["min_df"],
        sublinear_tf=tfidf_cfg["sublinear_tf"],
        strip_accents="unicode",
        analyzer="word",
    )

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    # features numéricas: passagem_turno (bool→int), hora_sin, hora_cos, n_palavras
    # StandardScaler apenas nas contínuas (sin/cos já estão em [-1,1]; aplicar não faz mal)
    numeric_passthrough = ["passagem_turno"]
    numeric_scale = [c for c in num_cols if c != "passagem_turno"]

    transformers = [
        ("tfidf", tfidf, text_col),
        ("ohe", ohe, cat_cols),
    ]
    if numeric_scale:
        transformers.append(("num_scale", StandardScaler(), numeric_scale))
    if numeric_passthrough:
        transformers.append(("num_pass", "passthrough", numeric_passthrough))

    ct = ColumnTransformer(transformers=transformers, remainder="drop")

    return Pipeline([("preprocessor", ct)])


def prepare_dataframe(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Adiciona features derivadas ao DataFrame antes de entrar no ColumnTransformer.

    Deve ser chamado tanto no treino quanto na inferência, antes de fit/transform.
    """
    cfg = params["features"]
    out = _add_temporal_features(df)
    out["n_palavras"] = out[cfg["text_col"]].str.split().str.len().fillna(0)
    return out
