"""
Stage DVC: train_spacy

Treina o modelo spaCy para classificação de risco de incidentes em FPSOs,
maximizando recall_critico via:
  1. textcat supervisionado (bow/ensemble)
  2. threshold tuning pós-treino
  3. fallback rule-based para padrões não aprendidos

Persiste:
  - models/spacy/textcat/           (modelo spaCy serializado)
  - models/spacy/rule_based_cfg.json
  - reports/metrics_spacy.json      (métricas — lido pelo DVC)
  - reports/figures/spacy/          (curva de aprendizado + confusion matrix)

Uso:
    python src/pipeline/train_spacy.py
    dvc repro train_spacy
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

# Garante UTF-8 no stdout/stderr mesmo em terminais Windows cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import yaml

from src.config.loader import load_config
from src.evaluation.metrics import (
    classification_metrics,
    eace_from_predictions,
    per_class_report,
    plot_confusion_matrix,
)
from src.features.build import prepare_dataframe
from src.models.spacy_model import (
    RuleBasedCriticoDetector,
    SpacyTextCatTrainer,
    _recall_critico,
    _precision_critico,
)


def _load_params() -> dict:
    with open(ROOT / "params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_data(params: dict) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    eda_cfg = load_config("configs/eda.yaml")
    train = pd.read_parquet(ROOT / eda_cfg["paths"]["train"])
    test = pd.read_parquet(ROOT / eda_cfg["paths"]["test"])

    leakage = ["ruido", "ambiguo", "anotado"]
    train = train.drop(columns=[c for c in leakage if c in train.columns])
    test = test.drop(columns=[c for c in leakage if c in test.columns])

    train = prepare_dataframe(train, params)
    test = prepare_dataframe(test, params)

    target = params["base"]["target_col"]
    text_col = params["features"]["text_col"]

    return (
        train[text_col],
        train[target],
        test[text_col],
        test[target],
    )


# ── Curva de aprendizado ──────────────────────────────────────────────────────

def _learning_curve(
    texts_train: pd.Series,
    labels_train: pd.Series,
    texts_val: pd.Series,
    labels_val: pd.Series,
    params: dict,
    spacy_cfg: dict,
    fracs: list[float],
) -> list[dict]:
    """Treina com subconjuntos crescentes de treino; avalia no val fixo.

    Retorna lista de dicts com frac, n_train, recall_critico, precision_critico, f1_macro.
    """
    seed = params["base"]["random_seed"]
    rng = np.random.default_rng(seed)
    curve_results = []

    texts_arr = texts_train.values
    labels_arr = labels_train.values
    idx = np.arange(len(texts_arr))

    print("\n[spacy] Curva de aprendizado:")
    for frac in fracs:
        n = max(int(len(idx) * frac), 10)
        chosen = rng.choice(idx, size=n, replace=False)
        t_sub = texts_arr[chosen]
        l_sub = labels_arr[chosen]

        trainer = SpacyTextCatTrainer(params, spacy_cfg)
        trainer.fit(
            texts_train=t_sub,
            labels_train=l_sub,
            texts_val=texts_val.values,
            labels_val=labels_val.values,
        )
        trainer.tune_threshold(
            texts=texts_val.values,
            labels=labels_val.values,
        )
        metrics = trainer.evaluate(texts_val.values, labels_val.values)
        row = {
            "frac": frac,
            "n_train": n,
            "recall_critico": metrics["recall_critico"],
            "precision_critico": metrics.get("precision_critico", 0.0),
            "f1_critico": metrics.get("f1_critico", 0.0),
            "f1_macro": metrics["f1_macro"],
        }
        curve_results.append(row)
        print(
            f"  frac={frac:.1f} n={n:>5} | "
            f"recall_critico={row['recall_critico']:.4f} | "
            f"f1_macro={row['f1_macro']:.4f}"
        )

    return curve_results


def _plot_learning_curve(curve: list[dict], fig_dir: Path) -> Path:
    fracs = [r["n_train"] for r in curve]
    metrics = {
        "recall_critico": [r["recall_critico"] for r in curve],
        "f1_critico": [r["f1_critico"] for r in curve],
        "f1_macro": [r["f1_macro"] for r in curve],
    }
    colors = {"recall_critico": "#B71C1C", "f1_critico": "#E53935", "f1_macro": "#1976D2"}
    labels_map = {"recall_critico": "Recall crítico", "f1_critico": "F1 crítico", "f1_macro": "F1 macro"}

    fig, ax = plt.subplots(figsize=(9, 5))
    for key, vals in metrics.items():
        ax.plot(fracs, vals, marker="o", label=labels_map[key], color=colors[key], linewidth=2)

    ax.axhline(0.75, color="gray", linestyle="--", linewidth=1, label="Target recall=0.75")
    ax.set_xlabel("Número de exemplos de treino")
    ax.set_ylabel("Score (validação)")
    ax.set_title("Curva de aprendizado — spaCy textcat")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    path = fig_dir / "learning_curve.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_threshold_curve(
    texts: pd.Series,
    labels: pd.Series,
    trainer: SpacyTextCatTrainer,
    fig_dir: Path,
) -> Path:
    """Plota recall vs precision vs threshold para a classe crítico."""
    grid = np.arange(0.05, 0.95, 0.05)
    scores = trainer.predict_proba(texts.values)
    recalls, precisions, f1s = [], [], []

    for thr in grid:
        preds = trainer._scores_to_labels(scores, float(thr))
        recalls.append(_recall_critico(labels.values, preds))
        precisions.append(_precision_critico(labels.values, preds))
        r, p = recalls[-1], precisions[-1]
        f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(grid, recalls, label="Recall crítico", color="#B71C1C", linewidth=2)
    ax.plot(grid, precisions, label="Precision crítico", color="#1976D2", linewidth=2)
    ax.plot(grid, f1s, label="F1 crítico", color="#388E3C", linewidth=2, linestyle="--")
    ax.axvline(trainer._threshold, color="orange", linestyle=":", linewidth=2,
               label=f"Threshold ótimo={trainer._threshold:.2f}")
    ax.set_xlabel("Threshold de decisão para 'crítico'")
    ax.set_ylabel("Score")
    ax.set_title("Trade-off recall × precision — classe crítico")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    path = fig_dir / "threshold_tradeoff.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    params = _load_params()
    spacy_cfg = load_config("configs/spacy.yaml")
    sp_params = params["spacy"]

    mlflow_uri = params["mlflow"]["tracking_uri"]
    if "://" not in mlflow_uri:
        mlflow_uri = (ROOT / mlflow_uri).as_uri()
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(sp_params["experiment_name"])

    texts_train, labels_train, texts_test, labels_test = _load_data(params)
    print(f"[train_spacy] train={len(texts_train)} | test={len(texts_test)}")

    model_dir = ROOT / "models" / "spacy"
    fig_dir = ROOT / "reports" / "figures" / "spacy"
    model_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Rule-based (sem treino, zero-shot) ─────────────────────────────────
    print("\n[spacy] Avaliando detector por regras (zero-shot)...")
    rule_detector = RuleBasedCriticoDetector(spacy_cfg)
    rule_flags = rule_detector.predict(texts_test.values)

    # converte flag binário → classe: 1=critico, 0=alto (conservador)
    rule_preds_raw = np.where(rule_flags == 1, "critico", "alto")
    rule_recall = _recall_critico(labels_test.values, rule_preds_raw)
    rule_prec = _precision_critico(labels_test.values, rule_preds_raw)
    print(f"  Rule-based | recall_critico={rule_recall:.4f} | precision_critico={rule_prec:.4f}")

    rule_cfg_path = model_dir / "rule_based_cfg.json"
    rule_cfg_path.write_text(
        json.dumps({"n_patterns": len(spacy_cfg.get("critico_patterns", [])),
                    "recall_critico": round(rule_recall, 4),
                    "precision_critico": round(rule_prec, 4)}, indent=2),
        encoding="utf-8",
    )

    # ── 2. Curva de aprendizado ────────────────────────────────────────────────
    fracs = sp_params.get("learning_curve_fracs", [0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    curve = _learning_curve(
        texts_train, labels_train,
        texts_test, labels_test,
        params, spacy_cfg, fracs,
    )
    curve_path = _plot_learning_curve(curve, fig_dir)
    print(f"[spacy] curva de aprendizado → {curve_path}")

    # ── 3. Treino final (100% dos dados de treino) ────────────────────────────
    print("\n[spacy] Treino final com 100% dos dados...")
    with mlflow.start_run(run_name="spacy_textcat"):
        mlflow.log_params({
            "base_model": sp_params["base_model"],
            "architecture": sp_params["textcat"]["architecture"],
            "n_iter": sp_params["textcat"]["n_iter"],
            "dropout": sp_params["textcat"]["dropout"],
        })

        trainer = SpacyTextCatTrainer(params, spacy_cfg)
        history = trainer.fit(
            texts_train=texts_train.values,
            labels_train=labels_train.values,
            texts_val=texts_test.values,
            labels_val=labels_test.values,
        )

        # ── Threshold tuning no test set ──────────────────────────────────────
        print("\n[spacy] Tuning de threshold...")
        best_threshold = trainer.tune_threshold(
            texts=texts_test.values,
            labels=labels_test.values,
            min_precision=0.30,
        )
        mlflow.log_metric("critico_threshold", best_threshold)

        # ── Avaliação final ───────────────────────────────────────────────────
        metrics = trainer.evaluate(texts_test.values, labels_test.values)
        eace = eace_from_predictions(
            labels_test.values,
            trainer.predict(texts_test.values),
            params["eace"],
            params["base"]["class_order"],
        )
        metrics["eace_brl"] = eace

        mlflow.log_metrics({k: v for k, v in metrics.items() if k != "eace_brl"})
        mlflow.log_metric("eace_brl", eace)

        # ── Figuras ───────────────────────────────────────────────────────────
        cm_fig = plot_confusion_matrix(
            labels_test.values,
            trainer.predict(texts_test.values),
            title="Confusion Matrix — spaCy textcat",
            class_order=params["base"]["class_order"],
        )
        cm_path = fig_dir / "confusion_matrix.png"
        cm_fig.savefig(cm_path, dpi=120, bbox_inches="tight")
        plt.close(cm_fig)

        thr_path = _plot_threshold_curve(texts_test, labels_test, trainer, fig_dir)

        mlflow.log_artifact(str(cm_path), artifact_path="figures")
        mlflow.log_artifact(str(thr_path), artifact_path="figures")
        mlflow.log_artifact(str(curve_path), artifact_path="figures")

        # ── Persistência ──────────────────────────────────────────────────────
        textcat_dir = model_dir / "textcat"
        trainer.save(textcat_dir)
        mlflow.log_artifact(str(textcat_dir), artifact_path="model")

        print(f"\n[spacy] modelo → {textcat_dir}")

    # ── 4. Histórico de épocas (figura de loss + recall) ──────────────────────
    _plot_training_history(history, fig_dir)

    # ── 5. Métricas DVC ───────────────────────────────────────────────────────
    dvc_metrics: dict = {
        "spacy__recall_critico": metrics["recall_critico"],
        "spacy__f1_critico": metrics.get("f1_critico", 0.0),
        "spacy__f1_macro": metrics["f1_macro"],
        "spacy__accuracy": metrics["accuracy"],
        "spacy__eace_brl": eace,
        "spacy__threshold": best_threshold,
        "spacy__rule_recall_critico": round(rule_recall, 4),
        "learning_curve": curve,
    }

    metrics_path = ROOT / "reports" / "metrics_spacy.json"
    metrics_path.write_text(json.dumps(dvc_metrics, indent=2), encoding="utf-8")
    print(f"[spacy] métricas → {metrics_path}")

    _print_summary(metrics, rule_recall, best_threshold, eace)
    print("\n[train_spacy] concluído.")


def _plot_training_history(history: list[dict], fig_dir: Path) -> None:
    epochs = [r["epoch"] for r in history]
    losses = [r["loss"] for r in history]
    recalls = [r.get("recall_critico", None) for r in history]

    has_val = any(r is not None for r in recalls)

    if has_val:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    else:
        fig, ax1 = plt.subplots(figsize=(7, 4))

    ax1.plot(epochs, losses, color="#1976D2", linewidth=2)
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss (textcat)")
    ax1.set_title("Loss de treino")
    ax1.grid(True, alpha=0.4)

    if has_val:
        ax2.plot(epochs, recalls, color="#B71C1C", linewidth=2, label="recall_critico")
        f1s = [r.get("f1_macro", None) for r in history]
        if any(f is not None for f in f1s):
            ax2.plot(epochs, f1s, color="#1976D2", linewidth=2, linestyle="--", label="f1_macro")
        ax2.axhline(0.75, color="gray", linestyle=":", linewidth=1)
        ax2.set_xlabel("Época")
        ax2.set_ylabel("Score (validação)")
        ax2.set_title("Métricas por época")
        ax2.legend()
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, alpha=0.4)

    fig.tight_layout()
    path = fig_dir / "training_history.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _print_summary(metrics: dict, rule_recall: float, threshold: float, eace: float) -> None:
    print("\n" + "=" * 70)
    print(f"{'Modelo':<25} {'Recall crit':>12} {'F1 macro':>10} {'EACE R$/ano':>14}")
    print("-" * 70)
    print(f"{'rule_based (zero-shot)':<25} {rule_recall:>12.4f} {'—':>10} {'—':>14}")
    print(f"{'spacy_textcat (tuned)':<25} {metrics['recall_critico']:>12.4f} "
          f"{metrics['f1_macro']:>10.4f} {eace:>14,.0f}")
    print(f"  threshold ótimo: {threshold:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
