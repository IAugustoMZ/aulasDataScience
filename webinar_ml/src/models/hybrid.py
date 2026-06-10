"""
Modelo híbrido: predição ML clássico + sinal spaCy.
Também contém TripleHybridClassifier: ML + spaCy BOW + spaCy tok2vec.

Três estratégias de fusão configuráveis em params.yaml → hybrid.fusion_strategy:

  "override"  — se o score spaCy['CRITICO'] >= override_threshold, a predição
                final é 'critico' independentemente do que o ML disse.
                Maximiza recall a custo de precision.

  "weighted"  — média ponderada dos scores de ambos os modelos; argmax decide.
                Requer que o ML exponha predict_proba (LogReg, RF, XGB).
                Balanceado: recall e precision crescem juntos.

  "stack"     — meta-modelo (LogReg leve) treinado sobre a concatenação dos
                scores de ambos os modelos. Aprende pesos ótimos a partir dos dados.
                Melhor quando treino e produção têm distribuição estável.

Todos os três são avaliados em conjunto no train_hybrid.py para comparação.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


CLASS_ORDER = ["baixo", "medio", "alto", "critico"]
LABEL_MAP = {c: c.upper() for c in CLASS_ORDER}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ml_proba(ml_pipeline, X: pd.DataFrame) -> np.ndarray:
    """Retorna array (n, 4) de probabilidades do modelo ML na ordem CLASS_ORDER.

    Funciona com qualquer estimador sklearn que tenha predict_proba.
    Para LinearSVC (sem proba), usa decision_function + softmax.
    """
    if hasattr(ml_pipeline, "predict_proba"):
        proba = ml_pipeline.predict_proba(X)
        classes = list(ml_pipeline.classes_)
    else:
        # LinearSVC: decision_function → softmax
        scores = ml_pipeline.decision_function(X)
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_s = np.exp(scores)
        proba = exp_s / exp_s.sum(axis=1, keepdims=True)
        classes = list(ml_pipeline.classes_)

    # reordena colunas para CLASS_ORDER
    idx = [classes.index(c) for c in CLASS_ORDER if c in classes]
    return proba[:, idx]


def _spacy_proba(trainer, texts: Sequence[str]) -> np.ndarray:
    """Retorna array (n, 4) de scores spaCy na ordem CLASS_ORDER."""
    scores = trainer.predict_proba(texts)
    return np.column_stack([scores[LABEL_MAP[c]] for c in CLASS_ORDER])


# ── Estratégia: override ──────────────────────────────────────────────────────

class OverrideFusion:
    """Se o score spaCy para 'critico' >= threshold, sobrescreve a predição ML.

    Intuição: o spaCy detectou padrão léxico forte → confia na regra.
    """

    def __init__(self, override_threshold: float):
        self.override_threshold = override_threshold

    def predict(
        self,
        ml_preds: np.ndarray,
        spacy_proba: np.ndarray,
    ) -> np.ndarray:
        critico_idx = CLASS_ORDER.index("critico")
        spacy_critico_score = spacy_proba[:, critico_idx]
        result = ml_preds.copy()
        result[spacy_critico_score >= self.override_threshold] = "critico"
        return result


# ── Estratégia: weighted ──────────────────────────────────────────────────────

class WeightedFusion:
    """Média ponderada dos scores ML e spaCy; argmax decide a classe final."""

    def __init__(self, spacy_weight: float):
        self.spacy_weight = spacy_weight
        self.ml_weight = 1.0 - spacy_weight

    def predict(
        self,
        ml_proba: np.ndarray,
        spacy_proba: np.ndarray,
    ) -> np.ndarray:
        combined = self.ml_weight * ml_proba + self.spacy_weight * spacy_proba
        indices = np.argmax(combined, axis=1)
        return np.array([CLASS_ORDER[i] for i in indices])

    def predict_proba(
        self,
        ml_proba: np.ndarray,
        spacy_proba: np.ndarray,
    ) -> np.ndarray:
        return self.ml_weight * ml_proba + self.spacy_weight * spacy_proba


# ── Estratégia: stack ─────────────────────────────────────────────────────────

class StackFusion:
    """Meta-modelo treinado sobre a concatenação dos scores de ML e spaCy.

    Features de entrada: [ml_proba(4) | spacy_proba(4)] → 8 features.
    Estimador: LogisticRegression leve (sem regularização pesada).
    """

    def __init__(self, random_state: int = 42):
        self._meta = LogisticRegression(
            C=1.0,
            max_iter=500,
            class_weight="balanced",
            random_state=random_state,
            solver="lbfgs",
        )
        self._le = LabelEncoder()
        self._fitted = False

    def fit(
        self,
        ml_proba: np.ndarray,
        spacy_proba: np.ndarray,
        y: Sequence[str],
    ) -> "StackFusion":
        X_meta = np.hstack([ml_proba, spacy_proba])
        y_enc = self._le.fit_transform(y)
        self._meta.fit(X_meta, y_enc)
        self._fitted = True
        return self

    def predict(
        self,
        ml_proba: np.ndarray,
        spacy_proba: np.ndarray,
    ) -> np.ndarray:
        assert self._fitted, "StackFusion não foi treinado. Chame fit() primeiro."
        X_meta = np.hstack([ml_proba, spacy_proba])
        y_enc = self._meta.predict(X_meta)
        return self._le.inverse_transform(y_enc)

    def predict_proba(
        self,
        ml_proba: np.ndarray,
        spacy_proba: np.ndarray,
    ) -> np.ndarray:
        assert self._fitted
        X_meta = np.hstack([ml_proba, spacy_proba])
        return self._meta.predict_proba(X_meta)


# ── Façade: HybridClassifier ──────────────────────────────────────────────────

class HybridClassifier:
    """Unifica as três estratégias numa interface única.

    Uso:
        clf = HybridClassifier(params, ml_pipeline, spacy_trainer)
        clf.fit_stack(X_train, texts_train, y_train)   # só para stack
        preds = clf.predict(X_test, texts_test)
    """

    def __init__(self, params: dict, ml_pipeline, spacy_trainer):
        self._params = params
        self._ml = ml_pipeline
        self._spacy = spacy_trainer
        h_cfg = params["hybrid"]

        self._strategy = h_cfg["fusion_strategy"]
        self._override = OverrideFusion(h_cfg["override_threshold"])
        self._weighted = WeightedFusion(h_cfg["spacy_weight"])
        self._stack = StackFusion(params["base"]["random_seed"])

    def fit_stack(
        self,
        X_train: pd.DataFrame,
        texts_train: Sequence[str],
        y_train: Sequence[str],
    ) -> None:
        """Treina o meta-modelo do stack (necessário apenas se strategy=stack)."""
        ml_p = _ml_proba(self._ml, X_train)
        sp_p = _spacy_proba(self._spacy, texts_train)
        self._stack.fit(ml_p, sp_p, y_train)

    def predict(
        self,
        X: pd.DataFrame,
        texts: Sequence[str],
        strategy: str | None = None,
    ) -> np.ndarray:
        strategy = strategy or self._strategy
        ml_p = _ml_proba(self._ml, X)
        sp_p = _spacy_proba(self._spacy, texts)

        ml_preds = np.array([CLASS_ORDER[i] for i in np.argmax(ml_p, axis=1)])

        if strategy == "override":
            return self._override.predict(ml_preds, sp_p)
        elif strategy == "weighted":
            return self._weighted.predict(ml_p, sp_p)
        elif strategy == "stack":
            return self._stack.predict(ml_p, sp_p)
        else:
            raise ValueError(f"fusion_strategy desconhecida: {strategy!r}")

    def predict_all_strategies(
        self,
        X: pd.DataFrame,
        texts: Sequence[str],
    ) -> dict[str, np.ndarray]:
        """Retorna predições para todas as três estratégias de uma vez."""
        ml_p = _ml_proba(self._ml, X)
        sp_p = _spacy_proba(self._spacy, texts)
        ml_preds = np.array([CLASS_ORDER[i] for i in np.argmax(ml_p, axis=1)])

        return {
            "override": self._override.predict(ml_preds, sp_p),
            "weighted": self._weighted.predict(ml_p, sp_p),
            "stack": self._stack.predict(ml_p, sp_p),
        }


# ── Híbrido triplo: ML + spaCy BOW + spaCy tok2vec ───────────────────────────

def _average_spacy_proba(
    bow_trainer,
    deep_trainer,
    texts: Sequence[str],
    bow_weight: float = 0.4,
    deep_weight: float = 0.6,
) -> np.ndarray:
    """Média ponderada dos scores dos dois modelos spaCy (BOW e tok2vec)."""
    bow_p = _spacy_proba(bow_trainer, texts)
    deep_p = _spacy_proba(deep_trainer, texts)
    return bow_weight * bow_p + deep_weight * deep_p


class TripleHybridClassifier:
    """Combina ML clássico + spaCy BOW + spaCy tok2vec.

    Estratégias:
      "override_deep"  — override usando score tok2vec (mais preciso)
      "weighted_avg"   — média ponderada: ML + (BOW*0.4 + tok2vec*0.6)
      "stack_triple"   — meta-LogReg sobre [ml_proba(4) | bow_proba(4) | deep_proba(4)] = 12 features
    """

    def __init__(
        self,
        params: dict,
        ml_pipeline,
        bow_trainer,
        deep_trainer,
    ):
        self._params = params
        self._ml = ml_pipeline
        self._bow = bow_trainer
        self._deep = deep_trainer
        h_cfg = params["hybrid"]

        self._override = OverrideFusion(h_cfg["override_threshold"])
        self._weighted_ml = 1.0 - h_cfg["spacy_weight"]
        self._weighted_sp = h_cfg["spacy_weight"]

        self._stack_meta = LogisticRegression(
            C=1.0,
            max_iter=500,
            class_weight="balanced",
            random_state=params["base"]["random_seed"],
            solver="lbfgs",
        )
        self._le = LabelEncoder()
        self._stack_fitted = False

    def fit_stack(
        self,
        X_train: pd.DataFrame,
        texts_train: Sequence[str],
        y_train: Sequence[str],
    ) -> None:
        ml_p = _ml_proba(self._ml, X_train)
        bow_p = _spacy_proba(self._bow, texts_train)
        deep_p = _spacy_proba(self._deep, texts_train)
        X_meta = np.hstack([ml_p, bow_p, deep_p])  # 12 features
        y_enc = self._le.fit_transform(y_train)
        self._stack_meta.fit(X_meta, y_enc)
        self._stack_fitted = True

    def predict(
        self,
        X: pd.DataFrame,
        texts: Sequence[str],
        strategy: str = "stack_triple",
    ) -> np.ndarray:
        ml_p = _ml_proba(self._ml, X)
        bow_p = _spacy_proba(self._bow, texts)
        deep_p = _spacy_proba(self._deep, texts)
        ml_preds = np.array([CLASS_ORDER[i] for i in np.argmax(ml_p, axis=1)])

        if strategy == "override_deep":
            return self._override.predict(ml_preds, deep_p)
        elif strategy == "weighted_avg":
            sp_avg = _average_spacy_proba(self._bow, self._deep, texts)
            combined = self._weighted_ml * ml_p + self._weighted_sp * sp_avg
            return np.array([CLASS_ORDER[i] for i in np.argmax(combined, axis=1)])
        elif strategy == "stack_triple":
            assert self._stack_fitted, "Chame fit_stack() primeiro."
            X_meta = np.hstack([ml_p, bow_p, deep_p])
            y_enc = self._stack_meta.predict(X_meta)
            return self._le.inverse_transform(y_enc)
        else:
            raise ValueError(f"strategy desconhecida: {strategy!r}")

    def predict_all_strategies(
        self,
        X: pd.DataFrame,
        texts: Sequence[str],
    ) -> dict[str, np.ndarray]:
        return {
            "override_deep": self.predict(X, texts, "override_deep"),
            "weighted_avg": self.predict(X, texts, "weighted_avg"),
            "stack_triple": self.predict(X, texts, "stack_triple"),
        }
