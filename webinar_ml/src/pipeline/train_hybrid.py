"""
Stage DVC: train_hybrid

Avaliação conjunta de três abordagens em comparação apples-to-apples:
  - Melhor modelo clássico (best_business.joblib)
  - spaCy textcat (models/spacy/textcat/)
  - Híbrido: ML + spaCy (override / weighted / stack)

Todos avaliados no mesmo test set, com as mesmas métricas:
  recall_critico, f1_macro, accuracy, EACE (R$/ano)

Persiste:
  - models/hybrid/stack_meta.joblib       (meta-modelo para strategy=stack)
  - reports/metrics_hybrid.json           (métricas — lido pelo DVC)
  - reports/figures/hybrid/               (comparison bar, confusion matrices)

Uso:
    python src/pipeline/train_hybrid.py
    dvc repro train_hybrid
"""

from __future__ import annotations

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

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import yaml

from src.config.loader import load_config
from src.evaluation.metrics import (
    classification_metrics,
    compare_models_table,
    eace_from_predictions,
    per_class_report,
    plot_confusion_matrix,
    plot_metrics_comparison,
)
from src.features.build import prepare_dataframe
from src.models.hybrid import HybridClassifier
from src.models.spacy_model import SpacyTextCatTrainer


# ── Carregamento ──────────────────────────────────────────────────────────────

def _load_params() -> dict:
    with open(ROOT / "params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_data(params: dict):
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

    return train, train[text_col], train[target], test, test[text_col], test[target]


def _load_ml_model(params: dict):
    model_path = ROOT / "models" / "classic" / "best_business.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo clássico não encontrado: {model_path}\n"
            "Execute 'dvc repro train_classic' primeiro."
        )
    return joblib.load(model_path)


def _load_spacy_model(params: dict) -> SpacyTextCatTrainer:
    model_path = ROOT / "models" / "spacy" / "textcat"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo spaCy não encontrado: {model_path}\n"
            "Execute 'dvc repro train_spacy' primeiro."
        )
    return SpacyTextCatTrainer.load(model_path, params)


# ── Avaliação comparativa ─────────────────────────────────────────────────────

def _eval_model(name: str, y_true: pd.Series, y_pred: np.ndarray, params: dict) -> dict:
    metrics = classification_metrics(y_true, y_pred, params["base"]["class_order"])
    eace = eace_from_predictions(y_true, y_pred, params["eace"], params["base"]["class_order"])
    metrics["eace_brl"] = eace
    return metrics


def _plot_comparison(results: dict[str, dict], fig_dir: Path) -> Path:
    metrics_to_plot = ["recall_critico", "f1_macro", "accuracy", "f1_critico"]
    # filtra métricas que estão presentes em todos os modelos
    metrics_to_plot = [m for m in metrics_to_plot if all(m in r for r in results.values())]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(4 * len(metrics_to_plot), 5))
    if len(metrics_to_plot) == 1:
        axes = [axes]

    model_names = list(results.keys())
    colors = ["#78909C", "#B71C1C", "#1565C0", "#2E7D32", "#F57F17"]

    for ax, metric in zip(axes, metrics_to_plot):
        vals = [results[m].get(metric, 0.0) for m in model_names]
        bars = ax.bar(range(len(model_names)), vals, color=colors[:len(model_names)], edgecolor="white")
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylim(0, 1.1)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
        ax.axhline(0.75, color="gray", linestyle="--", linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Comparação apples-to-apples — mesmo test set", fontsize=11, y=1.02)
    fig.tight_layout()
    path = fig_dir / "models_comparison.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_eace_comparison(results: dict[str, dict], fig_dir: Path) -> Path:
    """Gráfico de barras horizontal para EACE (R$/ano) — menor é melhor."""
    names = list(results.keys())
    eaces = [results[n].get("eace_brl", 0) for n in names]
    sorted_pairs = sorted(zip(eaces, names))
    eaces_s, names_s = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.7)))
    bars = ax.barh(names_s, eaces_s, color="#B71C1C", edgecolor="white", alpha=0.85)
    ax.set_xlabel("EACE (R$/ano) — menor é melhor")
    ax.set_title("Custo anual esperado de erros por modelo")
    for bar, val in zip(bars, eaces_s):
        ax.text(val + max(eaces_s) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"R${val:,.0f}", va="center", fontsize=8)
    fig.tight_layout()
    path = fig_dir / "eace_comparison.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_confusion_matrices(
    results_preds: dict[str, np.ndarray],
    y_test: pd.Series,
    params: dict,
    fig_dir: Path,
) -> list[Path]:
    paths = []
    for name, preds in results_preds.items():
        fig = plot_confusion_matrix(
            y_test, preds,
            title=f"Confusion Matrix — {name}",
            class_order=params["base"]["class_order"],
        )
        path = fig_dir / f"cm_{name.replace(' ', '_').replace('/', '_')}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    params = _load_params()

    mlflow_uri = params["mlflow"]["tracking_uri"]
    if "://" not in mlflow_uri:
        mlflow_uri = (ROOT / mlflow_uri).as_uri()
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(params["hybrid"]["experiment_name"])

    # ── Carrega dados ─────────────────────────────────────────────────────────
    train_df, texts_train, labels_train, test_df, texts_test, labels_test = _load_data(params)
    print(f"[train_hybrid] train={len(train_df)} | test={len(test_df)}")

    # ── Carrega modelos base ──────────────────────────────────────────────────
    ml_model = _load_ml_model(params)
    spacy_trainer = _load_spacy_model(params)

    model_dir = ROOT / "models" / "hybrid"
    fig_dir = ROOT / "reports" / "figures" / "hybrid"
    model_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Instancia o classificador híbrido e treina o stack ────────────────────
    hybrid = HybridClassifier(params, ml_model, spacy_trainer)

    print("\n[hybrid] Treinando meta-modelo (stack)...")
    hybrid.fit_stack(train_df, texts_train.values, labels_train.values)

    # Persiste o meta-modelo do stack
    stack_path = model_dir / "stack_meta.joblib"
    joblib.dump(hybrid._stack, stack_path)
    print(f"[hybrid] meta-modelo → {stack_path}")

    # ── Predições de todos os sistemas no mesmo test set ──────────────────────
    print("\n[hybrid] Gerando predições no test set...")
    ml_preds = ml_model.predict(test_df)
    spacy_preds = spacy_trainer.predict(texts_test.values)
    hybrid_preds = hybrid.predict_all_strategies(test_df, texts_test.values)

    all_preds: dict[str, np.ndarray] = {
        "ml_classico": ml_preds,
        "spacy_textcat": spacy_preds,
        "hibrido_override": hybrid_preds["override"],
        "hibrido_weighted": hybrid_preds["weighted"],
        "hibrido_stack": hybrid_preds["stack"],
    }

    # ── Avaliação conjunta ────────────────────────────────────────────────────
    print("\n[hybrid] Avaliação comparativa:")
    all_results: dict[str, dict] = {}
    for name, preds in all_preds.items():
        metrics = _eval_model(name, labels_test, preds, params)
        all_results[name] = metrics
        print(
            f"  {name:<22} | recall_critico={metrics['recall_critico']:.4f} "
            f"| f1_macro={metrics['f1_macro']:.4f} "
            f"| EACE=R${metrics.get('eace_brl', 0):>12,.0f}"
        )

    # ── MLflow: log de todos os sistemas num run único ────────────────────────
    with mlflow.start_run(run_name="hybrid_comparison"):
        mlflow.log_params({
            "fusion_strategy": params["hybrid"]["fusion_strategy"],
            "override_threshold": params["hybrid"]["override_threshold"],
            "spacy_weight": params["hybrid"]["spacy_weight"],
        })
        for name, metrics in all_results.items():
            for k, v in metrics.items():
                mlflow.log_metric(f"{name}__{k}", v)

        # ── Figuras ───────────────────────────────────────────────────────────
        comp_path = _plot_comparison(all_results, fig_dir)
        eace_path = _plot_eace_comparison(all_results, fig_dir)
        cm_paths = _save_confusion_matrices(all_preds, labels_test, params, fig_dir)

        mlflow.log_artifact(str(comp_path), artifact_path="figures")
        mlflow.log_artifact(str(eace_path), artifact_path="figures")
        for p in cm_paths:
            mlflow.log_artifact(str(p), artifact_path="figures")

        mlflow.log_artifact(str(stack_path), artifact_path="model")

    # ── Relatório de seleção do híbrido ───────────────────────────────────────
    best_hybrid = _select_best_hybrid(all_results, params)
    selection = {
        "best_hybrid_strategy": best_hybrid,
        "comparison": {
            name: {
                "recall_critico": m["recall_critico"],
                "f1_macro": m["f1_macro"],
                "eace_brl": m.get("eace_brl"),
                "accuracy": m["accuracy"],
            }
            for name, m in all_results.items()
        },
    }
    sel_path = ROOT / "reports" / "hybrid_selection_report.json"
    sel_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"\n[hybrid] relatório → {sel_path}")

    # ── Métricas DVC ─────────────────────────────────────────────────────────
    dvc_metrics: dict = {}
    for name, metrics in all_results.items():
        for k, v in metrics.items():
            if v is not None:
                dvc_metrics[f"{name}__{k}"] = v
    dvc_metrics["best_hybrid_strategy"] = best_hybrid

    metrics_path = ROOT / "reports" / "metrics_hybrid.json"
    metrics_path.write_text(json.dumps(dvc_metrics, indent=2), encoding="utf-8")
    print(f"[hybrid] métricas → {metrics_path}")

    _print_summary(all_results, best_hybrid)
    print("\n[train_hybrid] concluído.")


def _select_best_hybrid(results: dict[str, dict], params: dict) -> str:
    """Seleciona a melhor estratégia híbrida por EACE (menor é melhor).

    Considera apenas as variantes híbridas (não ml_classico / spacy_textcat).
    """
    hybrid_names = [n for n in results if n.startswith("hibrido_")]
    if not hybrid_names:
        return params["hybrid"]["fusion_strategy"]

    best = min(
        hybrid_names,
        key=lambda n: results[n].get("eace_brl", float("inf")),
    )
    eace_val = results[best].get("eace_brl", 0)
    print(f"\n[hybrid] Melhor estratégia híbrida: {best} (EACE=R${eace_val:,.0f}/ano)")
    return best


def _print_summary(results: dict[str, dict], best_hybrid: str) -> None:
    print("\n" + "=" * 100)
    header = f"{'Modelo':<24} {'Recall crit':>12} {'F1 macro':>10} {'Accuracy':>10} {'EACE R$/ano':>16}  Destaque"
    print(header)
    print("-" * 100)

    for name, m in sorted(results.items(), key=lambda x: x[1].get("eace_brl", float("inf"))):
        tag = "★ best hybrid" if name == best_hybrid else ""
        eace = m.get("eace_brl", float("nan"))
        print(
            f"{name:<24} {m['recall_critico']:>12.4f} {m['f1_macro']:>10.4f} "
            f"{m['accuracy']:>10.4f} {eace:>16,.0f}  {tag}"
        )
    print("=" * 100)


if __name__ == "__main__":
    main()
