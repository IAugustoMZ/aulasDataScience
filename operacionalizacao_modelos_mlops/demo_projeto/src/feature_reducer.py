"""
feature_reducer.py — Custom sklearn-compatible transformer for feature reduction.

Supports three configurable strategies driven by config/modeling.yaml
(feature_reduction section):

    none   → identity passthrough (no reduction)
    rfe    → Recursive Feature Elimination (supervised, tunable k)
    pca    → Principal Component Analysis (linear, unsupervised)
    kpca   → Kernel PCA (non-linear, unsupervised; supports rbf, poly, cosine)

Design principles:
    • Inherits BaseEstimator + TransformerMixin for full sklearn compatibility:
        - clone() in CV folds works correctly
        - get_params() / set_params() for Optuna and GridSearchCV
        - fit_transform() provided automatically by TransformerMixin
    • All hyperparameters are first-class __init__ args (no nested dicts) so
      Optuna can vary them directly via _suggest_param().
    • method='rfe' requires a supervised estimator.  Because FeatureReducer sits
      inside a Pipeline, sklearn calls fit(X, y) propagating y automatically —
      no manual wiring needed.
    • method='none' returns the original DataFrame/array unchanged so that
      downstream feature-importance extraction still works with named columns.

Usage in Pipeline (modelagem_walkthrough.py):
    from src.feature_reducer import FeatureReducer
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        ('imputer',   GroupMedianImputer(...)),
        ('scaler',    StandardScalerTransformer(...)),
        ('reducer',   FeatureReducer(method='rfe', n_features_to_select=15)),
        ('estimator', Ridge()),
    ])
    pipe.fit(X_train, y_train)

RFE estimator choices (rfe_estimator param):
    'ridge'          → Ridge(alpha=1.0)          — fast, good general baseline
    'random_forest'  → RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
    Any sklearn regressor instance can also be passed directly.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor


# ── RFE estimator factory ─────────────────────────────────────────────────────

_RFE_ESTIMATORS: dict[str, Any] = {
    'ridge'        : Ridge(alpha=1.0),
    'random_forest': RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42),
}


def _resolve_rfe_estimator(spec: Any) -> Any:
    """
    Resolves the rfe_estimator parameter.

    Accepts:
        str  → key into _RFE_ESTIMATORS (e.g. 'ridge', 'random_forest')
        obj  → any fitted/unfitted sklearn estimator passed directly
    """
    if isinstance(spec, str):
        if spec not in _RFE_ESTIMATORS:
            raise ValueError(
                f"rfe_estimator='{spec}' not recognised. "
                f"Valid strings: {list(_RFE_ESTIMATORS.keys())}"
            )
        return clone(_RFE_ESTIMATORS[spec])
    return clone(spec)


# ─────────────────────────────────────────────────────────────────────────────
# FeatureReducer
# ─────────────────────────────────────────────────────────────────────────────

class FeatureReducer(BaseEstimator, TransformerMixin):
    """
    Unified, config-driven feature reduction transformer.

    Parameters
    ----------
    method : str
        Reduction strategy: 'none' | 'rfe' | 'pca' | 'kpca'.
        Default 'none' (identity — no reduction).

    n_features_to_select : int
        [RFE only] Number of features to keep. Default 15.

    rfe_estimator : str or sklearn estimator
        [RFE only] Base estimator for feature importance ranking.
        Accepts 'ridge' or 'random_forest' (strings) or any sklearn estimator.
        Default 'ridge'.

    n_components : int or None
        [PCA / kPCA only] Number of output components.
        None means min(n_samples, n_features). Default 15.

    kernel : str
        [kPCA only] Kernel type. One of 'rbf', 'poly', 'cosine', 'linear',
        'sigmoid'. Default 'rbf'.

    gamma : float or None
        [kPCA only] Kernel coefficient for 'rbf', 'poly', 'sigmoid'.
        None → 1/n_features. Default None.

    degree : int
        [kPCA only] Degree for 'poly' kernel. Default 3.

    coef0 : float
        [kPCA only] Independent term for 'poly' and 'sigmoid' kernels.
        Default 1.0.

    logger : logging.Logger or None
        Optional logger for diagnostic messages.

    Attributes (set after fit)
    --------------------------
    reducer_     : fitted inner reducer (RFE, PCA, KernelPCA) or None for 'none'
    feature_names_in_ : list[str] | None
        Input feature names (from DataFrame column names), if available.
    feature_names_out_ : list[str] | None
        Output feature names after reduction (RFE keeps original names;
        PCA/kPCA use 'pc_0', 'pc_1', ...).
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

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _log(self, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(msg, *args)

    def _build_inner(self):
        """Instantiates the inner reducer from current params."""
        if self.method == 'none':
            return None
        if self.method == 'rfe':
            estimator = _resolve_rfe_estimator(self.rfe_estimator)
            return RFE(
                estimator=estimator,
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
            f"FeatureReducer: unknown method='{self.method}'. "
            "Valid options: 'none', 'rfe', 'pca', 'kpca'."
        )

    # ── sklearn API ───────────────────────────────────────────────────────────

    def fit(self, X, y=None) -> "FeatureReducer":
        """
        Fit the inner reducer on X (and y for RFE).

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) — required for method='rfe',
            ignored for 'pca' / 'kpca' / 'none'.
        """
        # Store input feature names if available
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = None

        # Clamp n_components to valid range before building the inner reducer.
        # Optuna may suggest values exceeding n_features; sklearn will raise a
        # ValueError if n_components >= min(n_samples, n_features).
        if self.method in ('pca', 'kpca'):
            n_features = X.shape[1]
            if self.n_components >= n_features:
                self.n_components = n_features - 1
                self._log(
                    "FeatureReducer.fit: n_components clamped to %d (< n_features=%d).",
                    self.n_components, n_features,
                )

        self.reducer_ = self._build_inner()

        if self.reducer_ is None:
            # 'none' — no fitting needed
            self.feature_names_out_ = self.feature_names_in_
            self._log("FeatureReducer.fit: method='none' — passthrough, no reduction.")
            return self

        if self.method == 'rfe':
            if y is None:
                raise ValueError(
                    "FeatureReducer with method='rfe' requires y. "
                    "Ensure it is inside a Pipeline that receives y."
                )
            self.reducer_.fit(X, y)
            if self.feature_names_in_ is not None:
                self.feature_names_out_ = [
                    col for col, sel in zip(self.feature_names_in_, self.reducer_.support_)
                    if sel
                ]
            else:
                self.feature_names_out_ = None
            self._log(
                "FeatureReducer.fit: RFE selected %d/%d features: %s",
                self.n_features_to_select,
                len(self.feature_names_in_) if self.feature_names_in_ else '?',
                self.feature_names_out_,
            )

        elif self.method in ('pca', 'kpca'):
            self.reducer_.fit(X)
            n_out = self.n_components
            self.feature_names_out_ = [f'pc_{i}' for i in range(n_out)]
            explained = None
            if self.method == 'pca' and hasattr(self.reducer_, 'explained_variance_ratio_'):
                explained = float(self.reducer_.explained_variance_ratio_.sum())
            self._log(
                "FeatureReducer.fit: %s fitted → %d components%s.",
                self.method.upper(), n_out,
                f' (explained variance: {explained:.3f})' if explained is not None else '',
            )

        return self

    def transform(self, X, y=None):
        """
        Apply the fitted reduction to X.

        Returns pd.DataFrame when input is pd.DataFrame and method is 'none' or 'rfe'
        (feature names are preserved/updated).  Returns np.ndarray for PCA/kPCA
        (column names change to pc_0, pc_1, ...) wrapped in a pd.DataFrame.
        """
        if not hasattr(self, 'reducer_'):
            raise RuntimeError(
                "FeatureReducer has not been fitted. Call fit() before transform()."
            )

        if self.reducer_ is None:
            # 'none' — identity
            return X

        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        X_out = self.reducer_.transform(X_arr)

        # Wrap result in DataFrame with meaningful column names
        if self.feature_names_out_ is not None:
            return pd.DataFrame(X_out, columns=self.feature_names_out_, index=(
                X.index if isinstance(X, pd.DataFrame) else None
            ))
        return X_out

    # ── Utility ───────────────────────────────────────────────────────────────

    @property
    def selected_features(self) -> list[str] | None:
        """
        Returns the list of output feature names after reduction.
        Available after fit(). Returns None for PCA/kPCA (components have no
        original-feature names) or if input had no column names.
        """
        return getattr(self, 'feature_names_out_', None)
