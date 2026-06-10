"""
Definições dos modelos clássicos de baseline.

Cada função retorna um estimador sklearn configurado a partir do params.yaml.
Os modelos são intencionalmente simples — baseline sólido e interpretável
antes de partir para spaCy e LLMs.
"""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def build_logistic_regression(params: dict) -> LogisticRegression:
    cfg = params["models"]["logistic_regression"]
    return LogisticRegression(
        max_iter=cfg["max_iter"],
        class_weight=cfg["class_weight"],
        solver=cfg["solver"],
        random_state=params["base"]["random_seed"],
    )


def build_linear_svc(params: dict) -> LinearSVC:
    cfg = params["models"]["linear_svc"]
    return LinearSVC(
        max_iter=cfg["max_iter"],
        class_weight=cfg["class_weight"],
        random_state=params["base"]["random_seed"],
    )


def build_random_forest(params: dict):
    from sklearn.ensemble import RandomForestClassifier
    cfg = params["models"]["random_forest"]
    return RandomForestClassifier(
        min_samples_leaf=cfg["min_samples_leaf"],
        class_weight=cfg["class_weight"],
        random_state=params["base"]["random_seed"],
        n_jobs=-1,
    )


def build_decision_tree(params: dict):
    from sklearn.tree import DecisionTreeClassifier
    cfg = params["models"]["decision_tree"]
    return DecisionTreeClassifier(
        class_weight=cfg["class_weight"],
        random_state=params["base"]["random_seed"],
    )


class _XGBWithLabelEncoder(BaseEstimator):
    """Wrapper que encodifica labels string antes de passar ao XGBClassifier.

    XGBoost >= 2.x exige labels inteiros em multiclasse — este wrapper mantém
    a interface sklearn (fit/predict/predict_proba) sem alterar o restante do pipeline.
    """

    def __init__(self, eval_metric="mlogloss", random_state=None, n_jobs=-1, verbosity=0,
                 n_estimators=100, learning_rate=0.1, max_depth=6):
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbosity = verbosity
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def _make_xgb(self):
        from xgboost import XGBClassifier
        return XGBClassifier(
            eval_metric=self.eval_metric,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=self.verbosity,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
        )

    def fit(self, X, y, **kwargs):
        from sklearn.preprocessing import LabelEncoder
        self._le = LabelEncoder().fit(y)
        self._xgb = self._make_xgb()
        self._xgb.fit(X, self._le.transform(y), **kwargs)
        self.classes_ = self._le.classes_
        return self

    def predict(self, X):
        return self._le.inverse_transform(self._xgb.predict(X))

    def predict_proba(self, X):
        return self._xgb.predict_proba(X)


def build_xgboost(params: dict):
    return _XGBWithLabelEncoder(
        eval_metric="mlogloss",
        random_state=params["base"]["random_seed"],
        n_jobs=-1,
        verbosity=0,
    )



def build_knn(params: dict):
    from sklearn.neighbors import KNeighborsClassifier
    return KNeighborsClassifier(n_jobs=-1)


# Mapa nome → builder — iterar sobre isso no train script
MODEL_BUILDERS = {
    "logistic_regression": build_logistic_regression,
    "linear_svc": build_linear_svc,
    "random_forest": build_random_forest,
    "decision_tree": build_decision_tree,
    "xgboost": build_xgboost,
    "knn": build_knn,
}


def get_param_grid(model_name: str, params: dict) -> dict:
    """Grids de hiperparâmetros por modelo. Chaves no formato 'classifier__<param>'."""
    cfg = params["models"][model_name]

    if model_name == "logistic_regression":
        return {"classifier__C": cfg["C"]}

    if model_name == "linear_svc":
        return {"classifier__C": cfg["C"]}

    if model_name == "random_forest":
        return {
            "classifier__n_estimators": cfg["n_estimators"],
            "classifier__max_depth": cfg["max_depth"],
        }

    if model_name == "decision_tree":
        return {
            "classifier__max_depth": cfg["max_depth"],
            "classifier__min_samples_leaf": cfg["min_samples_leaf"],
        }

    if model_name == "xgboost":
        return {
            "classifier__n_estimators": cfg["n_estimators"],
            "classifier__learning_rate": cfg["learning_rate"],
            "classifier__max_depth": cfg["max_depth"],
        }

    if model_name == "knn":
        return {
            "classifier__n_neighbors": cfg["n_neighbors"],
            "classifier__weights": cfg["weights"],
        }

    raise ValueError(f"Modelo desconhecido: {model_name}")
