"""
Módulo spaCy para classificação de incidentes de segurança em FPSOs.

Duas camadas complementares:
  1. RuleBasedCriticoDetector  — Matcher de padrões léxicos (configs/spacy.yaml)
     Alta precisão para termos críticos conhecidos; recall limitado.

  2. SpacyTextCatTrainer  — textcat supervisionado (bow ou ensemble)
     Aprende de dados; maximiza recall_critico via threshold tuning.

Design:
  - Sem side effects além de retornar predições e modelos treinados.
  - Threshold armazenado no modelo para serialização simples (model.meta).
  - Scripts de pipeline controlam I/O e MLflow logging.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import spacy
from spacy.matcher import Matcher
from spacy.training import Example
from spacy.util import compounding, minibatch


# ── Constantes ────────────────────────────────────────────────────────────────

CLASS_ORDER = ["baixo", "medio", "alto", "critico"]
LABEL_MAP = {c: c.upper() for c in CLASS_ORDER}   # spaCy usa labels em UPPER


# ── Detector por regras ────────────────────────────────────────────────────────

class RuleBasedCriticoDetector:
    """Identifica relatos com padrões léxicos de risco crítico.

    Retorna flag binário (0/1) por documento — não é um classificador
    multi-classe, mas um feature extractor de alto sinal.
    """

    def __init__(self, spacy_cfg: dict):
        self._nlp = spacy.blank("pt")
        self._matcher = Matcher(self._nlp.vocab)
        self._negation = set(spacy_cfg.get("negation_tokens", []))
        self._add_patterns(spacy_cfg.get("critico_patterns", []))

    def _add_patterns(self, patterns: list[list[dict]]) -> None:
        for i, pattern in enumerate(patterns):
            self._matcher.add(f"CRITICO_{i}", [pattern])

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Retorna array de 0/1: 1 se padrão crítico detectado sem negação próxima."""
        flags = []
        for doc in self._nlp.pipe(texts, batch_size=64):
            matches = self._matcher(doc)
            flagged = False
            for _, start, end in matches:
                # verifica negação nos 3 tokens anteriores
                window = [t.lower_ for t in doc[max(0, start - 3): start]]
                if not any(neg in window for neg in self._negation):
                    flagged = True
                    break
            flags.append(int(flagged))
        return np.array(flags)

    def predict_scores(self, texts: Sequence[str]) -> np.ndarray:
        """Retorna score contínuo: razão de padrões encontrados / max_possível (≥0)."""
        scores = []
        for doc in self._nlp.pipe(texts, batch_size=64):
            matches = self._matcher(doc)
            valid = sum(
                1 for _, start, _ in matches
                if not any(
                    t.lower_ in self._negation
                    for t in doc[max(0, start - 3): start]
                )
            )
            # normaliza pelo comprimento do doc para não penalizar relatos longos
            score = valid / max(len(doc), 1)
            scores.append(score)
        return np.array(scores)


# ── Trainer textcat ──────────────────────────────────────────────────────────

class SpacyTextCatTrainer:
    """Treina um textcat spaCy para classificação de risco (4 classes).

    Foco em recall_critico: após treino, encontra o threshold ótimo
    que maximiza recall sem deixar precision_critico cair abaixo de min_precision.
    """

    def __init__(self, params: dict, spacy_cfg: dict | None = None):
        self._params = params
        self._spacy_cfg = spacy_cfg or {}
        self._nlp: spacy.language.Language | None = None
        self._threshold: float = params["spacy"]["critico_threshold"]

    # ── Preparação dos dados ──────────────────────────────────────────────────

    @staticmethod
    def _to_spacy_examples(
        nlp: spacy.language.Language,
        texts: Sequence[str],
        labels: Sequence[str],
    ) -> list[Example]:
        """Converte texts + labels para list[Example] do spaCy."""
        examples = []
        all_labels = list(LABEL_MAP.values())
        for text, label in zip(texts, labels):
            doc = nlp.make_doc(text)
            cats = {lbl: 0.0 for lbl in all_labels}
            cats[LABEL_MAP[label]] = 1.0
            example = Example.from_dict(doc, {"cats": cats})
            examples.append(example)
        return examples

    # ── Treino ────────────────────────────────────────────────────────────────

    def fit(
        self,
        texts_train: Sequence[str],
        labels_train: Sequence[str],
        texts_val: Sequence[str] | None = None,
        labels_val: Sequence[str] | None = None,
    ) -> list[dict]:
        """Treina o textcat e retorna histórico de métricas por época.

        O histórico é usado para plotar a curva de aprendizado.
        """
        sp_cfg = self._params["spacy"]
        tc_cfg = sp_cfg["textcat"]

        base_model = sp_cfg.get("base_model", "pt_core_news_sm")
        try:
            self._nlp = spacy.load(base_model, exclude=["ner", "parser"])
        except OSError:
            # fallback: modelo em branco (sem vetores pré-treinados)
            self._nlp = spacy.blank("pt")

        # Adiciona textcat ao pipeline
        arch = tc_cfg["architecture"]
        if arch == "bow":
            arch_name = "spacy.TextCatBOW.v3"
        elif arch == "ensemble":
            arch_name = "spacy.TextCatEnsemble.v2"
        else:
            arch_name = arch  # aceita nome completo direto do config

        exclusive = tc_cfg.get("exclusive_classes", True)
        if "textcat" not in self._nlp.pipe_names:
            if arch == "bow":
                model_cfg = {
                    "@architectures": arch_name,
                    "exclusive_classes": exclusive,
                    "ngram_size": 1,
                    "no_output_layer": False,
                }
            elif arch == "ensemble":
                # TextCatEnsemble.v2 = tok2vec (CNN) + linear_model (BOW)
                # exclusive_classes fica no linear_model interno, não no topo
                width = tc_cfg.get("tok2vec_width", 96)
                depth = tc_cfg.get("tok2vec_depth", 4)
                model_cfg = {
                    "@architectures": arch_name,
                    "tok2vec": {
                        "@architectures": "spacy.Tok2Vec.v2",
                        "embed": {
                            "@architectures": "spacy.MultiHashEmbed.v2",
                            "width": width,
                            "attrs": ["NORM", "PREFIX", "SUFFIX", "SHAPE"],
                            "rows": [5000, 2500, 2500, 2500],
                            "include_static_vectors": True,
                        },
                        "encode": {
                            "@architectures": "spacy.MaxoutWindowEncoder.v2",
                            "width": width,
                            "depth": depth,
                            "window_size": 1,
                            "maxout_pieces": 3,
                        },
                    },
                    "linear_model": {
                        "@architectures": "spacy.TextCatBOW.v3",
                        "exclusive_classes": exclusive,
                        "ngram_size": 1,
                        "no_output_layer": False,
                    },
                }
            else:
                model_cfg = {
                    "@architectures": arch_name,
                    "exclusive_classes": exclusive,
                }
            textcat = self._nlp.add_pipe(
                "textcat",
                config={"model": model_cfg},
                last=True,
            )
        else:
            textcat = self._nlp.get_pipe("textcat")

        for label in LABEL_MAP.values():
            textcat.add_label(label)

        train_examples = self._to_spacy_examples(self._nlp, texts_train, labels_train)
        val_examples = (
            self._to_spacy_examples(self._nlp, texts_val, labels_val)
            if texts_val is not None else None
        )

        # Inicializa o modelo
        self._nlp.initialize(lambda: train_examples)

        opt_cfg = sp_cfg.get("optimizer", {})
        optimizer = self._nlp.select_pipes(enable=["textcat"])
        # configura o otimizador
        self._nlp.config["training"]["optimizer"]["learn_rate"] = opt_cfg.get("learn_rate", 0.001)

        seed = self._params["base"]["random_seed"]
        random.seed(seed)
        np.random.seed(seed)

        history: list[dict] = []
        n_iter = tc_cfg["n_iter"]
        dropout = tc_cfg["dropout"]
        batch_size = tc_cfg["batch_size"]
        batch_compound = tc_cfg.get("batch_size_compound", 1.001)

        for epoch in range(n_iter):
            random.shuffle(train_examples)
            losses: dict = {}

            batches = minibatch(
                train_examples,
                size=compounding(batch_size, batch_size * 32, batch_compound),
            )
            with self._nlp.select_pipes(enable=["textcat"]):
                for batch in batches:
                    self._nlp.update(batch, drop=dropout, losses=losses)

            epoch_rec: dict = {"epoch": epoch + 1, "loss": round(losses.get("textcat", 0.0), 4)}

            if val_examples is not None:
                val_metrics = self._evaluate_examples(val_examples)
                epoch_rec.update(val_metrics)

            history.append(epoch_rec)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                val_info = (
                    f" | val recall_critico={epoch_rec.get('recall_critico', 0):.4f}"
                    if val_examples else ""
                )
                print(f"  epoch {epoch+1:>3}/{n_iter} | loss={epoch_rec['loss']:.4f}{val_info}")

        return history

    # ── Threshold tuning ──────────────────────────────────────────────────────

    def tune_threshold(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        min_precision: float = 0.30,
        grid: Sequence[float] | None = None,
    ) -> float:
        """Varre thresholds e escolhe o que maximiza recall_critico com
        precision_critico >= min_precision.

        Salva o threshold escolhido em self._threshold.
        """
        if grid is None:
            grid = np.arange(0.10, 0.90, 0.05).tolist()

        scores = self._raw_scores(texts)
        best_threshold = self._threshold
        best_recall = 0.0

        results = []
        for thr in grid:
            preds = self._scores_to_labels(scores, thr)
            rc = _recall_critico(labels, preds)
            pc = _precision_critico(labels, preds)
            results.append({"threshold": round(thr, 3), "recall_critico": rc, "precision_critico": pc})
            if pc >= min_precision and rc > best_recall:
                best_recall = rc
                best_threshold = thr

        self._threshold = round(float(best_threshold), 3)
        self._nlp.meta["critico_threshold"] = self._threshold
        print(f"  [threshold] otimo = {self._threshold:.3f} -> recall_critico={best_recall:.4f}")
        return self._threshold

    # ── Predição ──────────────────────────────────────────────────────────────

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Prediz classe com o threshold tuned."""
        scores = self._raw_scores(texts)
        return self._scores_to_labels(scores, self._threshold)

    def predict_proba(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        """Retorna dict label → array de scores para cada documento."""
        assert self._nlp is not None, "Modelo não treinado. Chame fit() primeiro."
        result: dict[str, list] = {lbl: [] for lbl in LABEL_MAP.values()}
        for doc in self._nlp.pipe(texts, batch_size=64):
            for lbl in LABEL_MAP.values():
                result[lbl].append(doc.cats.get(lbl, 0.0))
        return {lbl: np.array(v) for lbl, v in result.items()}

    def _raw_scores(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        return self.predict_proba(texts)

    def _scores_to_labels(
        self, scores: dict[str, np.ndarray], threshold: float
    ) -> np.ndarray:
        """Converte scores em labels.

        Se score['CRITICO'] >= threshold → 'critico'.
        Caso contrário → argmax dos demais scores.
        """
        n = len(next(iter(scores.values())))
        preds = []
        critico_scores = scores[LABEL_MAP["critico"]]
        for i in range(n):
            if critico_scores[i] >= threshold:
                preds.append("critico")
            else:
                best_lbl = max(
                    (lbl for lbl in LABEL_MAP.values() if lbl != LABEL_MAP["critico"]),
                    key=lambda l: scores[l][i],
                )
                # volta para lowercase (convenção do projeto)
                preds.append(best_lbl.lower())
        return np.array(preds)

    # ── Avaliação interna ─────────────────────────────────────────────────────

    def _evaluate_examples(self, examples: list[Example]) -> dict:
        texts = [eg.reference.text for eg in examples]
        labels = [
            max(eg.reference.cats, key=eg.reference.cats.get).lower()
            for eg in examples
        ]
        preds = self.predict(texts)
        return {
            "recall_critico": round(_recall_critico(labels, preds), 4),
            "precision_critico": round(_precision_critico(labels, preds), 4),
            "f1_critico": round(_f1_critico(labels, preds), 4),
        }

    def evaluate(
        self, texts: Sequence[str], labels: Sequence[str]
    ) -> dict:
        """Avalia no conjunto fornecido com o threshold atual."""
        from src.evaluation.metrics import classification_metrics
        preds = self.predict(texts)
        return classification_metrics(labels, preds)

    # ── Persistência ─────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Salva o modelo spaCy completo (inclui threshold em meta)."""
        assert self._nlp is not None, "Modelo não treinado."
        self._nlp.meta["critico_threshold"] = self._threshold
        self._nlp.to_disk(path)

    @classmethod
    def load(cls, path: Path, params: dict) -> "SpacyTextCatTrainer":
        """Carrega modelo previamente salvo."""
        trainer = cls(params)
        trainer._nlp = spacy.load(path)
        trainer._threshold = trainer._nlp.meta.get(
            "critico_threshold", params["spacy"]["critico_threshold"]
        )
        return trainer


# ── Variante profunda: lê textcat_deep em vez de textcat ─────────────────────

class SpacyDeepTextCatTrainer(SpacyTextCatTrainer):
    """Mesmo que SpacyTextCatTrainer, mas lê params['spacy']['textcat_deep'].

    Permite comparar BOW vs. tok2vec/ensemble no mesmo notebook
    sem duplicar lógica de treino, threshold tuning ou persistência.
    """

    def __init__(self, params: dict, spacy_cfg: dict | None = None):
        # Injeta textcat_deep no lugar de textcat antes de chamar super()
        import copy
        params_deep = copy.deepcopy(params)
        if "textcat_deep" in params_deep.get("spacy", {}):
            params_deep["spacy"]["textcat"] = params_deep["spacy"]["textcat_deep"]
        super().__init__(params_deep, spacy_cfg)


# ── Helpers de métricas (sem sklearn para evitar import circular) ──────────────

def _recall_critico(y_true: Sequence, y_pred: Sequence) -> float:
    true_pos = sum(1 for t, p in zip(y_true, y_pred) if t == "critico" and p == "critico")
    total_critico = sum(1 for t in y_true if t == "critico")
    return true_pos / total_critico if total_critico > 0 else 0.0


def _precision_critico(y_true: Sequence, y_pred: Sequence) -> float:
    true_pos = sum(1 for t, p in zip(y_true, y_pred) if t == "critico" and p == "critico")
    pred_critico = sum(1 for p in y_pred if p == "critico")
    return true_pos / pred_critico if pred_critico > 0 else 0.0


def _f1_critico(y_true: Sequence, y_pred: Sequence) -> float:
    r = _recall_critico(y_true, y_pred)
    p = _precision_critico(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
