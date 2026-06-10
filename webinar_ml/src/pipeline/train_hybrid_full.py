"""
Stage DVC: train_hybrid_full

Avaliação global de todos os sistemas num único benchmark apples-to-apples:
  - ml_classico (LogReg best_business)
  - spacy_bow (textcat BOW)
  - spacy_tok2vec (textcat ensemble/tok2vec)
  - hibrido_override / hibrido_weighted / hibrido_stack (ML + BOW)
  - triple_override_deep / triple_weighted_avg / triple_stack (ML + BOW + tok2vec)

Todos avaliados no mesmo test set com as mesmas métricas.

Persiste:
  - models/hybrid_full/triple_stack_meta.joblib
  - reports/metrics_hybrid_full.json          (métricas — lido pelo DVC)
  - reports/hybrid_full_selection_report.json
  - reports/figures/hybrid_full/
      global_comparison.png
      eace_comparison.png
      recall_f1_scatter.png
      cm_*.png

Uso:
    python src/pipeline/train_hybrid_full.py
    dvc repro train_hybrid_full
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mlflow
import numpy as np
import pandas as pd
import yaml

from src.config.loader import load_config
from src.evaluation.metrics import (
    classification_metrics,
    eace_from_predictions,
    plot_confusion_matrix,
)
from src.features.build import prepare_dataframe
from src.models.hybrid import HybridClassifier, TripleHybridClassifier
from src.models.spacy_model import SpacyDeepTextCatTrainer, SpacyTextCatTrainer


LEAKAGE_COLS = ["ruido", "ambiguo", "anotado"]
CLASS_ORDER = ["baixo", "medio", "alto", "critico"]

# Grupos de modelos para organização visual
MODEL_GROUPS = {
    "Tier 1 — ML Clássico": ["ml_classico"],
    "Tier 2 — spaCy": ["spacy_bow", "spacy_tok2vec"],
    "Híbrido Duplo (ML+BOW)": ["hibrido_override", "hibrido_weighted", "hibrido_stack"],
    "Híbrido Triplo (ML+BOW+tok2vec)": ["triple_override_deep", "triple_weighted_avg", "triple_stack"],
}

GROUP_COLORS = {
    "ml_classico":          "#78909C",
    "spacy_bow":            "#546E7A",
    "spacy_tok2vec":        "#1565C0",
    "hibrido_override":     "#E53935",
    "hibrido_weighted":     "#EF9A9A",
    "hibrido_stack":        "#C62828",
    "triple_override_deep": "#2E7D32",
    "triple_weighted_avg":  "#A5D6A7",
    "triple_stack":         "#1B5E20",
}


def _load_params() -> dict:
    with open(ROOT / "params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_data(params: dict):
    eda_cfg = load_config("configs/eda.yaml")
    train = pd.read_parquet(ROOT / eda_cfg["paths"]["train"])
    test = pd.read_parquet(ROOT / eda_cfg["paths"]["test"])

    train = train.drop(columns=[c for c in LEAKAGE_COLS if c in train.columns])
    test = test.drop(columns=[c for c in LEAKAGE_COLS if c in test.columns])

    train = prepare_dataframe(train, params)
    test = prepare_dataframe(test, params)

    target = params["base"]["target_col"]
    text_col = params["features"]["text_col"]

    return train, train[text_col], train[target], test, test[text_col], test[target]


def _eval(y_true, y_pred, params: dict) -> dict:
    m = classification_metrics(y_true, y_pred, params["base"]["class_order"])
    eace = eace_from_predictions(y_true, y_pred, params["eace"], params["base"]["class_order"])
    m["eace_brl"] = eace
    return m


# ── Figuras ───────────────────────────────────────────────────────────────────

def _plot_global_comparison(results: dict[str, dict], fig_dir: Path) -> Path:
    """Bar chart agrupado por tier — recall_critico e f1_macro lado a lado."""
    names = list(results.keys())
    recall = [results[n]["recall_critico"] for n in names]
    f1 = [results[n]["f1_macro"] for n in names]
    colors = [GROUP_COLORS.get(n, "#888") for n in names]

    x = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(12, len(names) * 1.2), 5))
    bars1 = ax.bar(x - width/2, recall, width, label="Recall crítico", color=colors, edgecolor="white", alpha=0.9)
    bars2 = ax.bar(x + width/2, f1, width, label="F1 macro", color=colors, edgecolor="black", linewidth=0.5, alpha=0.55)

    ax.axhline(0.75, color="gray", linestyle="--", linewidth=1, label="Target recall=0.75")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Benchmark global — todos os sistemas\n(barra cheia = recall crítico | barra hachurada = F1 macro)", fontsize=11)
    ax.legend(loc="upper right")

    for bar, val in zip(bars1, recall):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}",
                ha="center", va="bottom", fontsize=7, fontweight="bold")

    # Adiciona separadores de grupo
    group_boundaries = []
    pos = 0
    for group_name, group_models in MODEL_GROUPS.items():
        present = [m for m in group_models if m in names]
        if not present:
            continue
        start = names.index(present[0])
        end = names.index(present[-1])
        if pos > 0:
            ax.axvline(start - 0.5, color="lightgray", linestyle="-", linewidth=1.5)
        ax.text((start + end) / 2, 1.07, group_name, ha="center", fontsize=8, color="dimgray")
        pos += 1

    fig.tight_layout()
    path = fig_dir / "global_comparison.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_eace_comparison(results: dict[str, dict], fig_dir: Path) -> Path:
    names = list(results.keys())
    eaces = [results[n].get("eace_brl", 0) for n in names]
    sorted_pairs = sorted(zip(eaces, names))
    eaces_s, names_s = zip(*sorted_pairs)
    colors_s = [GROUP_COLORS.get(n, "#888") for n in names_s]

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.65)))
    bars = ax.barh(names_s, eaces_s, color=colors_s, edgecolor="white", alpha=0.85)
    ax.set_xlabel("EACE (R$/ano) — menor é melhor")
    ax.set_title("Custo anual esperado de erros — benchmark global")
    for bar, val in zip(bars, eaces_s):
        ax.text(val + max(eaces_s) * 0.005, bar.get_y() + bar.get_height() / 2,
                f"R${val:,.0f}", va="center", fontsize=8)

    # Legenda de grupos
    legend_patches = [
        mpatches.Patch(color="#78909C", label="ML Clássico"),
        mpatches.Patch(color="#1565C0", label="spaCy"),
        mpatches.Patch(color="#C62828", label="Híbrido Duplo"),
        mpatches.Patch(color="#1B5E20", label="Híbrido Triplo"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8)
    fig.tight_layout()
    path = fig_dir / "eace_comparison.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_recall_f1_scatter(results: dict[str, dict], fig_dir: Path) -> Path:
    """Scatter plot recall_critico × f1_macro — posição ideal = canto superior direito."""
    fig, ax = plt.subplots(figsize=(9, 7))

    for name, m in results.items():
        x = m["recall_critico"]
        y = m["f1_macro"]
        color = GROUP_COLORS.get(name, "#888")
        ax.scatter(x, y, s=120, color=color, zorder=5, edgecolors="white", linewidths=1.5)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.axvline(0.75, color="gray", linestyle="--", linewidth=1, label="Target recall=0.75")
    ax.set_xlabel("Recall crítico (↑ melhor — KPI de negócio)")
    ax.set_ylabel("F1 macro (↑ melhor — qualidade geral)")
    ax.set_title("Posicionamento dos sistemas\nRecall crítico × F1 macro", fontsize=11)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    legend_patches = [
        mpatches.Patch(color="#78909C", label="ML Clássico"),
        mpatches.Patch(color="#1565C0", label="spaCy"),
        mpatches.Patch(color="#C62828", label="Híbrido Duplo"),
        mpatches.Patch(color="#1B5E20", label="Híbrido Triplo"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
    fig.tight_layout()
    path = fig_dir / "recall_f1_scatter.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_confusion_matrices(
    all_preds: dict[str, np.ndarray],
    y_test: pd.Series,
    params: dict,
    fig_dir: Path,
) -> list[Path]:
    paths = []
    for name, preds in all_preds.items():
        fig = plot_confusion_matrix(
            y_test, preds,
            title=f"Confusion Matrix — {name}",
            class_order=params["base"]["class_order"],
        )
        path = fig_dir / f"cm_{name}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    params = _load_params()
    spacy_cfg = load_config("configs/spacy.yaml")

    mlflow_uri = params["mlflow"]["tracking_uri"]
    if "://" not in mlflow_uri:
        mlflow_uri = (ROOT / mlflow_uri).as_uri()
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("webinar_ml_hybrid_full")

    train_df, texts_train, labels_train, test_df, texts_test, labels_test = _load_data(params)
    print(f"[train_hybrid_full] train={len(train_df)} | test={len(test_df)}")

    fig_dir = ROOT / "reports" / "figures" / "hybrid_full"
    model_dir = ROOT / "models" / "hybrid_full"
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Carrega modelos base ──────────────────────────────────────────────────
    ml_path = ROOT / "models" / "classic" / "best_business.joblib"
    bow_path = ROOT / "models" / "spacy" / "textcat"
    deep_path = ROOT / "models" / "spacy" / "textcat_deep"

    missing = [str(p) for p in [ml_path, bow_path, deep_path] if not p.exists()]
    if missing:
        print("[train_hybrid_full] modelos faltando:")
        for m in missing:
            print(f"  {m}")
        print("Execute: dvc repro train_classic train_spacy train_spacy_deep")
        sys.exit(1)

    ml_model = joblib.load(ml_path)
    bow_trainer = SpacyTextCatTrainer.load(bow_path, params)
    deep_trainer = SpacyDeepTextCatTrainer.load(deep_path, params)
    print(f"  ML: {type(ml_model.named_steps['classifier']).__name__}")
    print(f"  BOW: carregado de {bow_path}")
    print(f"  tok2vec: carregado de {deep_path}")

    # ── Predições individuais ─────────────────────────────────────────────────
    print("\n[train_hybrid_full] Gerando predições individuais...")
    all_preds: dict[str, np.ndarray] = {
        "ml_classico":    ml_model.predict(test_df),
        "spacy_bow":      bow_trainer.predict(texts_test.values),
        "spacy_tok2vec":  deep_trainer.predict(texts_test.values),
    }
    all_results: dict[str, dict] = {
        name: _eval(labels_test, preds, params)
        for name, preds in all_preds.items()
    }

    # ── Híbrido duplo (ML + BOW) ──────────────────────────────────────────────
    print("\n[train_hybrid_full] Treinando híbrido duplo (ML + BOW)...")
    hybrid_dual = HybridClassifier(params, ml_model, bow_trainer)
    hybrid_dual.fit_stack(train_df, texts_train.values, labels_train.values)
    joblib.dump(hybrid_dual._stack, model_dir / "dual_stack_meta.joblib")

    for strategy, preds in hybrid_dual.predict_all_strategies(test_df, texts_test.values).items():
        name = f"hibrido_{strategy}"
        all_preds[name] = preds
        all_results[name] = _eval(labels_test, preds, params)
        m = all_results[name]
        print(f"  {name:<22} recall={m['recall_critico']:.4f} f1={m['f1_macro']:.4f} EACE=R${m['eace_brl']:,.0f}")

    # ── Híbrido triplo (ML + BOW + tok2vec) ───────────────────────────────────
    print("\n[train_hybrid_full] Treinando híbrido triplo (ML + BOW + tok2vec)...")
    hybrid_triple = TripleHybridClassifier(params, ml_model, bow_trainer, deep_trainer)
    hybrid_triple.fit_stack(train_df, texts_train.values, labels_train.values)
    joblib.dump(hybrid_triple._stack_meta, model_dir / "triple_stack_meta.joblib")

    for strategy, preds in hybrid_triple.predict_all_strategies(test_df, texts_test.values).items():
        name = f"triple_{strategy}"
        all_preds[name] = preds
        all_results[name] = _eval(labels_test, preds, params)
        m = all_results[name]
        print(f"  {name:<22} recall={m['recall_critico']:.4f} f1={m['f1_macro']:.4f} EACE=R${m['eace_brl']:,.0f}")

    # ── Figuras ───────────────────────────────────────────────────────────────
    print("\n[train_hybrid_full] Gerando figuras...")
    with mlflow.start_run(run_name="hybrid_full_benchmark"):
        for name, m in all_results.items():
            for k, v in m.items():
                if v is not None:
                    mlflow.log_metric(f"{name}__{k}", v)

        comp_path = _plot_global_comparison(all_results, fig_dir)
        eace_path = _plot_eace_comparison(all_results, fig_dir)
        scatter_path = _plot_recall_f1_scatter(all_results, fig_dir)
        cm_paths = _save_confusion_matrices(all_preds, labels_test, params, fig_dir)

        for p in [comp_path, eace_path, scatter_path] + cm_paths:
            mlflow.log_artifact(str(p), artifact_path="figures")

    print(f"  global_comparison → {comp_path}")
    print(f"  eace_comparison   → {eace_path}")
    print(f"  scatter           → {scatter_path}")
    print(f"  confusion matrices → {fig_dir}/cm_*.png")

    # ── Relatório de seleção ──────────────────────────────────────────────────
    best_eace = min(all_results, key=lambda n: all_results[n].get("eace_brl", float("inf")))
    best_recall = max(all_results, key=lambda n: all_results[n]["recall_critico"])

    selection = {
        "best_by_eace": best_eace,
        "best_by_recall_critico": best_recall,
        "results": {
            name: {
                "recall_critico": m["recall_critico"],
                "f1_macro": m["f1_macro"],
                "accuracy": m["accuracy"],
                "f1_critico": m.get("f1_critico", 0.0),
                "eace_brl": m.get("eace_brl"),
            }
            for name, m in all_results.items()
        },
    }
    sel_path = ROOT / "reports" / "hybrid_full_selection_report.json"
    sel_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")

    # ── Métricas DVC ─────────────────────────────────────────────────────────
    dvc_metrics: dict = {
        "best_by_eace": best_eace,
        "best_by_recall_critico": best_recall,
    }
    for name, m in all_results.items():
        for k, v in m.items():
            if v is not None:
                dvc_metrics[f"{name}__{k}"] = v

    metrics_path = ROOT / "reports" / "metrics_hybrid_full.json"
    metrics_path.write_text(json.dumps(dvc_metrics, indent=2), encoding="utf-8")

    print(f"\n[train_hybrid_full] seleção → {sel_path}")
    print(f"[train_hybrid_full] métricas → {metrics_path}")

    _print_summary(all_results, best_eace, best_recall)
    print("\n[train_hybrid_full] concluído.")


def _print_summary(results: dict, best_eace: str, best_recall: str) -> None:
    print("\n" + "=" * 105)
    print(f"{'Modelo':<26} {'Recall crit':>12} {'F1 macro':>10} {'Accuracy':>10} {'F1 crítico':>11} {'EACE R$/ano':>16}")
    print("-" * 105)

    # Imprime por grupo
    for group_name, group_models in MODEL_GROUPS.items():
        present = [m for m in group_models if m in results]
        if not present:
            continue
        print(f"\n  ── {group_name} ──")
        for name in present:
            m = results[name]
            tags = []
            if name == best_eace:
                tags.append("★ EACE")
            if name == best_recall:
                tags.append("★ Recall")
            tag_str = "  " + " | ".join(tags) if tags else ""
            eace = m.get("eace_brl", float("nan"))
            print(f"  {name:<24} {m['recall_critico']:>12.4f} {m['f1_macro']:>10.4f} "
                  f"{m['accuracy']:>10.4f} {m.get('f1_critico', 0):>11.4f} {eace:>16,.0f}{tag_str}")

    print("=" * 105)


if __name__ == "__main__":
    main()
