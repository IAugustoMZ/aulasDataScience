"""
modeling/reducer.py — Transformador sklearn-compatível para redução de features.

Suporta quatro estratégias configuráveis via config/modeling.yaml
(seção feature_reduction):

    none   → passthrough (sem redução)
    rfe    → Eliminação Recursiva de Features (supervisionada, k tunável)
    pca    → Análise de Componentes Principais (linear, não supervisionada)
    kpca   → Kernel PCA (não-linear; suporta rbf, poly, cosine)

Princípios de design:
    • Herda BaseEstimator + TransformerMixin para compatibilidade total com sklearn:
        - clone() em folds de CV funciona corretamente
        - get_params() / set_params() para Optuna e GridSearchCV
        - fit_transform() fornecido automaticamente pelo TransformerMixin
    • Todos os hiperparâmetros são argumentos de primeiro nível no __init__
      (sem dicts aninhados) para que o Optuna possa variá-los diretamente.
    • method='rfe' requer estimador supervisionado. Como o FeatureReducer fica
      dentro de um Pipeline, o sklearn propaga y automaticamente no fit().
    • method='none' retorna o DataFrame/array original inalterado, preservando
      nomes de colunas para extração de importância de features posterior.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor


# ── Fábrica de estimadores para RFE ──────────────────────────────────────────

_ESTIMADORES_RFE: dict[str, Any] = {
    'ridge'         : Ridge(alpha=1.0),
    'random_forest' : RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42),
}


def _resolver_estimador_rfe(spec: Any) -> Any:
    """
    Resolve o parâmetro rfe_estimator.

    Aceita:
        str  → chave em _ESTIMADORES_RFE (ex: 'ridge', 'random_forest')
        obj  → qualquer estimador sklearn passado diretamente
    """
    if isinstance(spec, str):
        if spec not in _ESTIMADORES_RFE:
            raise ValueError(
                f"rfe_estimator='{spec}' não reconhecido. "
                f"Valores válidos: {list(_ESTIMADORES_RFE.keys())}"
            )
        return clone(_ESTIMADORES_RFE[spec])
    return clone(spec)


# ─────────────────────────────────────────────────────────────────────────────
# FeatureReducer
# ─────────────────────────────────────────────────────────────────────────────

class FeatureReducer(BaseEstimator, TransformerMixin):
    """
    Transformador unificado e configurável para redução de features.

    Parâmetros
    ----------
    method : str
        Estratégia de redução: 'none' | 'rfe' | 'pca' | 'kpca'.
        Padrão 'none' (identidade — sem redução).

    n_features_to_select : int
        [Somente RFE] Número de features a manter. Padrão 15.

    rfe_estimator : str ou estimador sklearn
        [Somente RFE] Estimador base para ranqueamento por importância.
        Aceita 'ridge' ou 'random_forest' (strings) ou qualquer estimador sklearn.
        Padrão 'ridge'.

    n_components : int ou None
        [Somente PCA / kPCA] Número de componentes de saída.
        None significa min(n_amostras, n_features). Padrão 15.

    kernel : str
        [Somente kPCA] Tipo de kernel. Um de 'rbf', 'poly', 'cosine', 'linear',
        'sigmoid'. Padrão 'rbf'.

    gamma : float ou None
        [Somente kPCA] Coeficiente do kernel para 'rbf', 'poly', 'sigmoid'.
        None → 1/n_features. Padrão None.

    degree : int
        [Somente kPCA] Grau para kernel 'poly'. Padrão 3.

    coef0 : float
        [Somente kPCA] Termo independente para kernels 'poly' e 'sigmoid'.
        Padrão 1.0.

    logger : logging.Logger ou None
        Logger opcional para mensagens diagnósticas.

    Atributos (definidos após fit)
    ------------------------------
    reducer_      : redutor interno ajustado (RFE, PCA, KernelPCA) ou None para 'none'
    feature_names_in_  : list[str] | None
        Nomes das features de entrada (de colunas do DataFrame), se disponíveis.
    feature_names_out_ : list[str] | None
        Nomes das features de saída após redução (RFE preserva nomes originais;
        PCA/kPCA usa 'pc_0', 'pc_1', ...).
    """

    def __init__(
        self,
        method: str = 'none',
        n_features_to_select: int = 15,
        rfe_estimator: Any = 'ridge',
        n_components: int = 15,
        kernel: str = 'rbf',
        gamma: float | None = None,
        degree: int = 3,
        coef0: float = 1.0,
        logger: Any = None,
    ) -> None:
        self.method = method
        self.n_features_to_select = n_features_to_select
        self.rfe_estimator = rfe_estimator
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.logger = logger

    # ── Helpers internos ──────────────────────────────────────────────────────

    def _logar(self, msg: str, *args: Any) -> None:
        """Loga mensagem se logger estiver configurado."""
        if self.logger:
            self.logger.info(msg, *args)

    def _construir_redutor_interno(self):
        """Instancia o redutor interno a partir dos parâmetros atuais."""
        if self.method == 'none':
            return None
        if self.method == 'rfe':
            estimador = _resolver_estimador_rfe(self.rfe_estimator)
            return RFE(
                estimator=estimador,
                n_features_to_select=self.n_features_to_select,
            )
        if self.method == 'pca':
            return PCA(n_components=self.n_components, random_state=42)
        if self.method == 'kpca':
            return KernelPCA(
                n_components=self.n_components,
                kernel=self.kernel,
                gamma=self.gamma,
                degree=self.degree,
                coef0=self.coef0,
            )
        raise ValueError(
            f"FeatureReducer: method='{self.method}' desconhecido. "
            "Opções válidas: 'none', 'rfe', 'pca', 'kpca'."
        )

    # ── API sklearn ───────────────────────────────────────────────────────────

    def fit(self, X, y=None) -> "FeatureReducer":
        """
        Ajusta o redutor interno em X (e y para RFE).

        Parâmetros
        ----------
        X : array-like ou pd.DataFrame de shape (n_amostras, n_features)
        y : array-like de shape (n_amostras,) — obrigatório para method='rfe',
            ignorado para 'pca' / 'kpca' / 'none'.
        """
        # Armazena nomes das features de entrada, se disponíveis
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = None

        # Limita n_components ao intervalo válido antes de construir o redutor.
        # O Optuna pode sugerir valores maiores que n_features; o sklearn levantaria
        # ValueError se n_components >= min(n_amostras, n_features).
        if self.method in ('pca', 'kpca'):
            n_features = X.shape[1]
            if self.n_components >= n_features:
                self.n_components = n_features - 1
                self._logar(
                    "FeatureReducer.fit: n_components limitado a %d (< n_features=%d).",
                    self.n_components, n_features,
                )

        self.reducer_ = self._construir_redutor_interno()

        if self.reducer_ is None:
            # 'none' — sem ajuste necessário
            self.feature_names_out_ = self.feature_names_in_
            self._logar("FeatureReducer.fit: method='none' — passthrough, sem redução.")
            return self

        if self.method == 'rfe':
            if y is None:
                raise ValueError(
                    "FeatureReducer com method='rfe' requer y. "
                    "Certifique-se de que está dentro de um Pipeline que recebe y."
                )
            self.reducer_.fit(X, y)
            if self.feature_names_in_ is not None:
                self.feature_names_out_ = [
                    col for col, sel in zip(self.feature_names_in_, self.reducer_.support_)
                    if sel
                ]
            else:
                self.feature_names_out_ = None
            self._logar(
                "FeatureReducer.fit: RFE selecionou %d/%d features: %s",
                self.n_features_to_select,
                len(self.feature_names_in_) if self.feature_names_in_ else '?',
                self.feature_names_out_,
            )

        elif self.method in ('pca', 'kpca'):
            self.reducer_.fit(X)
            n_out = self.n_components
            self.feature_names_out_ = [f'pc_{i}' for i in range(n_out)]
            variancia_explicada = None
            if self.method == 'pca' and hasattr(self.reducer_, 'explained_variance_ratio_'):
                variancia_explicada = float(self.reducer_.explained_variance_ratio_.sum())
            self._logar(
                "FeatureReducer.fit: %s ajustado → %d componentes%s.",
                self.method.upper(), n_out,
                f' (variância explicada: {variancia_explicada:.3f})' if variancia_explicada is not None else '',
            )

        return self

    def transform(self, X, y=None):
        """
        Aplica a redução ajustada em X.

        Retorna pd.DataFrame quando a entrada é pd.DataFrame e o método é 'none' ou 'rfe'
        (nomes de features são preservados/atualizados). Retorna np.ndarray para PCA/kPCA
        (nomes de colunas mudam para pc_0, pc_1, ...) empacotado em pd.DataFrame.
        """
        if not hasattr(self, 'reducer_'):
            raise RuntimeError(
                "FeatureReducer não foi ajustado. Chame fit() antes de transform()."
            )

        if self.reducer_ is None:
            # 'none' — identidade
            return X

        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        X_out = self.reducer_.transform(X_arr)

        # Empacota resultado em DataFrame com nomes de colunas significativos
        if self.feature_names_out_ is not None:
            return pd.DataFrame(X_out, columns=self.feature_names_out_, index=(
                X.index if isinstance(X, pd.DataFrame) else None
            ))
        return X_out

    # ── Utilitário ────────────────────────────────────────────────────────────

    @property
    def features_selecionadas(self) -> list[str] | None:
        """
        Retorna a lista de nomes de features de saída após a redução.
        Disponível após fit(). Retorna None para PCA/kPCA (componentes não têm
        nomes de features originais) ou se a entrada não tinha nomes de colunas.
        """
        return getattr(self, 'feature_names_out_', None)

    # Alias em inglês para compatibilidade com código existente
    @property
    def selected_features(self) -> list[str] | None:
        """Alias de features_selecionadas para compatibilidade com código legado."""
        return self.features_selecionadas
