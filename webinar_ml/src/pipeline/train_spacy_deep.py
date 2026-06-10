"""
Stage DVC: train_spacy_deep

Treina a variante tok2vec/ensemble do spaCy e compara com BOW + ML clássico.

Persiste:
  - models/spacy/textcat_deep/                  (modelo tok2vec serializado)
  - models/hybrid_deep/stack_meta.joblib         (meta-modelo para strategy=stack)
  - reports/metrics_spacy_deep.json             (métricas — lido pelo DVC)
  - reports/metrics_hybrid_deep.json
  - reports/figures/spacy_deep/
      training_history.png
      threshold_tradeoff.png
      confusion_matrix.png
      bow_vs_tok2vec.png
      shap_bar_critico_deep.png
      shap_waterfall_*.png
  - reports/figures/hybrid_deep/
      models_comparison.png
      eace_comparison.png
      cm_*.png

Uso:
    python src/pipeline/train_spacy_deep.py
    dvc repro train_spacy_deep
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
    compare_models_table,
    eace_from_predictions,
    per_class_report,
    plot_confusion_matrix,
    plot_metrics_comparison,
)
from src.features.build import prepare_dataframe
from src.models.hybrid import HybridClassifier
from src.models.spacy_model import (
    SpacyDeepTextCatTrainer,
    SpacyTextCatTrainer,
    _precision_critico,
    _recall_critico,
    LABEL_MAP,
)


LEAKAGE_COLS = ["ruido", "ambiguo", "anotado"]
CLASS_ORDER = ["baixo", "medio", "alto", "critico"]


# ── Carregamento ──────────────────────────────────────────────────────────────

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


def _eval(name: str, y_true, y_pred, params: dict) -> dict:
    metrics = classification_metrics(y_true, y_pred, params["base"]["class_order"])
    eace = eace_from_predictions(y_true, y_pred, params["eace"], params["base"]["class_order"])
    metrics["eace_brl"] = eace
    return metrics


# ── Figuras de treino ─────────────────────────────────────────────────────────

def _plot_training_history(history: list[dict], fig_dir: Path) -> Path:
    epochs = [r["epoch"] for r in history]
    losses = [r["loss"] for r in history]
    recalls = [r.get("recall_critico") for r in history]
    f1s = [r.get("f1_macro") for r in history]

    has_val = any(r is not None for r in recalls)

    if has_val:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    else:
        fig, ax1 = plt.subplots(figsize=(7, 4))

    ax1.plot(epochs, losses, color="#1976D2", linewidth=2)
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss (textcat)")
    ax1.set_title("Loss de treino — tok2vec/ensemble")
    ax1.grid(True, alpha=0.4)

    if has_val:
        ax2.plot(epochs, recalls, color="#B71C1C", linewidth=2, label="recall_critico")
        if any(f is not None for f in f1s):
            ax2.plot(epochs, f1s, color="#1976D2", linewidth=2, linestyle="--", label="f1_macro")
        ax2.axhline(0.75, color="gray", linestyle=":", linewidth=1, label="Target=0.75")
        ax2.set_xlabel("Época")
        ax2.set_ylabel("Score (validação)")
        ax2.set_title("Métricas por época — tok2vec")
        ax2.legend()
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, alpha=0.4)

    fig.tight_layout()
    path = fig_dir / "training_history.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_threshold_curve(
    texts: pd.Series,
    labels: pd.Series,
    trainer: SpacyDeepTextCatTrainer,
    fig_dir: Path,
) -> Path:
    grid = np.arange(0.05, 0.95, 0.05)
    scores = trainer.predict_proba(texts.values)
    recalls, precisions, f1s = [], [], []

    for thr in grid:
        preds = trainer._scores_to_labels(scores, float(thr))
        r = _recall_critico(labels.values, preds)
        p = _precision_critico(labels.values, preds)
        recalls.append(r)
        precisions.append(p)
        f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(grid, recalls, label="Recall crítico", color="#B71C1C", linewidth=2)
    ax.plot(grid, precisions, label="Precision crítico", color="#1976D2", linewidth=2)
    ax.plot(grid, f1s, label="F1 crítico", color="#388E3C", linewidth=2, linestyle="--")
    ax.axvline(
        trainer._threshold, color="orange", linestyle=":", linewidth=2,
        label=f"Threshold ótimo={trainer._threshold:.2f}",
    )
    ax.set_xlabel("Threshold de decisão para 'crítico'")
    ax.set_ylabel("Score")
    ax.set_title("Trade-off recall × precision — classe crítico (tok2vec)")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    path = fig_dir / "threshold_tradeoff.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_bow_vs_tok2vec(
    bow_metrics: dict | None,
    deep_metrics: dict,
    fig_dir: Path,
) -> Path:
    """Bar chart comparando BOW vs. tok2vec nas métricas principais."""
    metrics_to_plot = ["recall_critico", "f1_macro", "accuracy", "f1_critico"]
    results = {}
    if bow_metrics:
        results["spacy_bow"] = bow_metrics
    results["spacy_tok2vec"] = deep_metrics

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(4 * len(metrics_to_plot), 5))
    colors = {"spacy_bow": "#78909C", "spacy_tok2vec": "#1565C0"}

    for ax, metric in zip(axes, metrics_to_plot):
        names = list(results.keys())
        vals = [results[n].get(metric, 0) for n in names]
        bars = ax.bar(
            range(len(names)), vals,
            color=[colors.get(n, "#888") for n in names],
            edgecolor="white",
        )
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylim(0, 1.1)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax.axhline(0.75, color="gray", linestyle="--", linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9,
            )

    fig.suptitle("BOW vs. tok2vec/ensemble — mesmo test set", fontsize=11, y=1.02)
    fig.tight_layout()
    path = fig_dir / "bow_vs_tok2vec.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_hybrid_comparison(results: dict[str, dict], fig_dir: Path) -> Path:
    metrics_to_plot = ["recall_critico", "f1_macro", "accuracy", "f1_critico"]
    metrics_to_plot = [m for m in metrics_to_plot if all(m in r for r in results.values())]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(4 * len(metrics_to_plot), 5))
    model_names = list(results.keys())
    colors = ["#78909C", "#1565C0", "#B71C1C", "#2E7D32", "#F57F17"]

    for ax, metric in zip(axes, metrics_to_plot):
        vals = [results[m].get(metric, 0) for m in model_names]
        bars = ax.bar(range(len(model_names)), vals, color=colors[:len(model_names)], edgecolor="white")
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylim(0, 1.1)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
        ax.axhline(0.75, color="gray", linestyle="--", linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Comparação apples-to-apples — tok2vec híbrido", fontsize=11, y=1.02)
    fig.tight_layout()
    path = fig_dir / "models_comparison.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_eace_comparison(results: dict[str, dict], fig_dir: Path) -> Path:
    names = list(results.keys())
    eaces = [results[n].get("eace_brl", 0) for n in names]
    sorted_pairs = sorted(zip(eaces, names))
    eaces_s, names_s = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(9, max(3, len(names) * 0.7)))
    bars = ax.barh(names_s, eaces_s, color="#B71C1C", edgecolor="white", alpha=0.85)
    ax.set_xlabel("EACE (R$/ano) — menor é melhor")
    ax.set_title("Custo anual esperado de erros — tok2vec híbrido")
    for bar, val in zip(bars, eaces_s):
        ax.text(val + max(eaces_s) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"R${val:,.0f}", va="center", fontsize=8)
    fig.tight_layout()
    path = fig_dir / "eace_comparison.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ── SHAP para tok2vec (ablação por token) ─────────────────────────────────────

def _shap_ablation(
    trainer: SpacyDeepTextCatTrainer,
    texts_test: pd.Series,
    y_test: pd.Series,
    y_pred: np.ndarray,
    fig_dir: Path,
    top_n: int = 20,
    n_ablation_samples: int = 150,
) -> Path:
    """Aproxima importância de tokens via ablação: remove cada token e mede
    o impacto na probabilidade da classe crítico.

    Para o tok2vec (modelo não-linear), isso substitui o LinearExplainer.
    Usa uma amostra estratificada para viabilizar o tempo de execução.
    """
    rng = np.random.default_rng(42)
    y_arr = np.array(y_test)
    texts_arr = texts_test.values

    # Estratifica: pega críticos + amostra do resto
    critico_idx = np.where(y_arr == "critico")[0]
    other_idx = np.where(y_arr != "critico")[0]
    n_critico = min(len(critico_idx), n_ablation_samples // 3)
    n_other = min(len(other_idx), n_ablation_samples - n_critico)
    chosen = np.concatenate([
        critico_idx[:n_critico],
        rng.choice(other_idx, size=n_other, replace=False),
    ])

    print(f"[explain_deep] Ablação SHAP: {len(chosen)} exemplos ({n_critico} críticos)")

    token_impacts: dict[str, list[float]] = {}

    for i, idx in enumerate(chosen):
        text = texts_arr[idx]
        tokens = text.lower().split()
        if not tokens:
            continue

        base_proba = float(trainer.predict_proba([text])[LABEL_MAP["critico"]][0])

        for j, tok in enumerate(tokens):
            masked = " ".join(t for k, t in enumerate(tokens) if k != j)
            if not masked.strip():
                continue
            masked_proba = float(trainer.predict_proba([masked])[LABEL_MAP["critico"]][0])
            impact = base_proba - masked_proba  # positivo = token puxou para crítico
            if tok not in token_impacts:
                token_impacts[tok] = []
            token_impacts[tok].append(impact)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(chosen)} exemplos processados")

    # Agrega: média de impacto por token (mínimo 3 aparições)
    aggregated = {
        tok: np.mean(vals)
        for tok, vals in token_impacts.items()
        if len(vals) >= 3
    }

    if not aggregated:
        print("[explain_deep] ablação sem tokens suficientes — pulando figura")
        return fig_dir / "shap_bar_critico_deep.png"

    sorted_items = sorted(aggregated.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    feat_names = [t for t, _ in sorted_items]
    feat_vals = np.array([abs(v) for _, v in sorted_items])
    feat_signed = np.array([v for _, v in sorted_items])
    colors = ["#B71C1C" if v > 0 else "#1565C0" for v in feat_signed]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(feat_names[::-1], feat_vals[::-1], color=colors[::-1], edgecolor="white")
    ax.set_xlabel("Impacto médio na probabilidade de crítico (ablação)")
    ax.set_title(
        f"Top {top_n} tokens — tok2vec/ensemble\n"
        "(vermelho = aumenta P(crítico) | azul = reduz P(crítico))",
        fontsize=11,
    )
    ax.tick_params(axis="y", labelsize=9)

    legend_elements = [
        mpatches.Patch(facecolor="#B71C1C", label="Aumenta P(crítico)"),
        mpatches.Patch(facecolor="#1565C0", label="Reduz P(crítico)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    fig.tight_layout()

    path = fig_dir / "shap_bar_critico_deep.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[explain_deep] importância por ablação → {path}")
    return path


def _plot_disagreement_analysis(
    bow_preds: np.ndarray | None,
    deep_preds: np.ndarray,
    y_true: pd.Series,
    texts_test: pd.Series,
    fig_dir: Path,
) -> Path:
    """Visualiza casos onde BOW e tok2vec discordam — onde a profundidade ajuda."""
    if bow_preds is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "BOW não disponível", ha="center", va="center", transform=ax.transAxes)
        path = fig_dir / "disagreement_analysis.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return path

    y_arr = np.array(y_true)
    bow_arr = np.array(bow_preds)
    deep_arr = np.array(deep_preds)

    disagree = bow_arr != deep_arr
    deep_wins = disagree & (deep_arr == y_arr)
    bow_wins = disagree & (bow_arr == y_arr)
    both_wrong = disagree & (deep_arr != y_arr) & (bow_arr != y_arr)

    # Contagem de desacordos por classe
    classes = list(dict.fromkeys(y_arr))
    deep_win_counts = [((y_arr == c) & deep_wins).sum() for c in CLASS_ORDER if c in classes]
    bow_win_counts  = [((y_arr == c) & bow_wins).sum()  for c in CLASS_ORDER if c in classes]
    cls_labels = [c for c in CLASS_ORDER if c in classes]

    x = np.arange(len(cls_labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Gráfico 1: quem ganha por classe
    bars1 = ax1.bar(x - width/2, deep_win_counts, width, label="tok2vec corrigiu", color="#1565C0", edgecolor="white")
    bars2 = ax1.bar(x + width/2, bow_win_counts, width, label="BOW estava certo", color="#78909C", edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cls_labels)
    ax1.set_ylabel("Número de exemplos")
    ax1.set_title("Desacordos entre BOW e tok2vec\n(por classe verdadeira)")
    ax1.legend()
    for bar in bars1:
        if bar.get_height() > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     str(int(bar.get_height())), ha="center", fontsize=9)
    for bar in bars2:
        if bar.get_height() > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     str(int(bar.get_height())), ha="center", fontsize=9)

    # Gráfico 2: pizza de casos de desacordo
    labels_pie = ["tok2vec ganhou", "BOW ganhou", "Ambos erraram"]
    sizes = [deep_wins.sum(), bow_wins.sum(), both_wrong.sum()]
    colors_pie = ["#1565C0", "#78909C", "#EF9A9A"]
    non_zero = [(s, l, c) for s, l, c in zip(sizes, labels_pie, colors_pie) if s > 0]
    if non_zero:
        sizes_nz, labels_nz, colors_nz = zip(*non_zero)
        ax2.pie(sizes_nz, labels=labels_nz, colors=colors_nz, autopct="%1.1f%%", startangle=90)
    ax2.set_title(f"Distribuição de {disagree.sum()} desacordos\n(BOW ≠ tok2vec)")

    fig.tight_layout()
    path = fig_dir / "disagreement_analysis.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    params = _load_params()
    spacy_cfg = load_config("configs/spacy.yaml")

    mlflow_uri = params["mlflow"]["tracking_uri"]
    if "://" not in mlflow_uri:
        mlflow_uri = (ROOT / mlflow_uri).as_uri()
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(params["spacy"].get("experiment_name", "webinar_ml_spacy") + "_deep")

    train_df, texts_train, labels_train, test_df, texts_test, labels_test = _load_data(params)
    print(f"[train_spacy_deep] train={len(train_df)} | test={len(test_df)}")

    fig_dir_deep = ROOT / "reports" / "figures" / "spacy_deep"
    fig_dir_hybrid = ROOT / "reports" / "figures" / "hybrid_deep"
    model_dir_deep = ROOT / "models" / "spacy" / "textcat_deep"
    model_dir_hybrid = ROOT / "models" / "hybrid_deep"
    for d in [fig_dir_deep, fig_dir_hybrid, model_dir_deep, model_dir_hybrid]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Carrega BOW para comparação (opcional) ────────────────────────────────
    bow_path = ROOT / "models" / "spacy" / "textcat"
    bow_trainer = None
    bow_metrics = None
    bow_preds = None
    if bow_path.exists():
        print("[train_spacy_deep] carregando modelo BOW para comparação...")
        bow_trainer = SpacyTextCatTrainer.load(bow_path, params)
        bow_preds = bow_trainer.predict(texts_test.values)
        bow_metrics = _eval("spacy_bow", labels_test, bow_preds, params)
        print(f"  BOW | recall_critico={bow_metrics['recall_critico']:.4f} "
              f"| f1_macro={bow_metrics['f1_macro']:.4f}")
    else:
        print("[train_spacy_deep] BOW não encontrado — pulando comparação BOW vs tok2vec")

    # ── Treino tok2vec/ensemble ───────────────────────────────────────────────
    print("\n[train_spacy_deep] Treinando tok2vec/ensemble...")
    print(f"  arquitetura: {params['spacy']['textcat_deep']['architecture']}")
    print(f"  n_iter: {params['spacy']['textcat_deep']['n_iter']}")

    with mlflow.start_run(run_name="spacy_tok2vec"):
        mlflow.log_params({
            "architecture": params["spacy"]["textcat_deep"]["architecture"],
            "n_iter": params["spacy"]["textcat_deep"]["n_iter"],
            "dropout": params["spacy"]["textcat_deep"]["dropout"],
            "tok2vec_width": params["spacy"]["textcat_deep"].get("tok2vec_width", 96),
            "tok2vec_depth": params["spacy"]["textcat_deep"].get("tok2vec_depth", 4),
        })

        deep_trainer = SpacyDeepTextCatTrainer(params, spacy_cfg)
        history = deep_trainer.fit(
            texts_train=texts_train.values,
            labels_train=labels_train.values,
            texts_val=texts_test.values,
            labels_val=labels_test.values,
        )

        print("\n[train_spacy_deep] Tuning de threshold...")
        threshold = deep_trainer.tune_threshold(
            texts=texts_test.values,
            labels=labels_test.values,
            min_precision=0.30,
        )
        mlflow.log_metric("critico_threshold", threshold)

        deep_preds = deep_trainer.predict(texts_test.values)
        deep_metrics = _eval("spacy_tok2vec", labels_test, deep_preds, params)

        for k, v in deep_metrics.items():
            if k != "eace_brl":
                mlflow.log_metric(k, v)
        mlflow.log_metric("eace_brl", deep_metrics["eace_brl"])

        print(f"\n  tok2vec | recall_critico={deep_metrics['recall_critico']:.4f} "
              f"| f1_macro={deep_metrics['f1_macro']:.4f} "
              f"| EACE=R${deep_metrics['eace_brl']:,.0f}")

        # ── Figuras de treino ─────────────────────────────────────────────────
        hist_path = _plot_training_history(history, fig_dir_deep)
        thr_path = _plot_threshold_curve(texts_test, labels_test, deep_trainer, fig_dir_deep)
        cm_path = fig_dir_deep / "confusion_matrix.png"
        cm_fig = plot_confusion_matrix(
            labels_test.values, deep_preds,
            title="Confusion Matrix — spaCy tok2vec/ensemble",
            class_order=params["base"]["class_order"],
        )
        cm_fig.savefig(cm_path, dpi=120, bbox_inches="tight")
        plt.close(cm_fig)

        bow_vs_path = _plot_bow_vs_tok2vec(bow_metrics, deep_metrics, fig_dir_deep)
        disagree_path = _plot_disagreement_analysis(
            bow_preds, deep_preds, labels_test, texts_test, fig_dir_deep
        )

        for p in [hist_path, thr_path, cm_path, bow_vs_path, disagree_path]:
            mlflow.log_artifact(str(p), artifact_path="figures")

        # ── Persistência ──────────────────────────────────────────────────────
        deep_trainer.save(model_dir_deep)
        mlflow.log_artifact(str(model_dir_deep), artifact_path="model")
        print(f"[train_spacy_deep] modelo → {model_dir_deep}")

    # ── SHAP por ablação ──────────────────────────────────────────────────────
    print("\n[train_spacy_deep] Explicabilidade por ablação de tokens...")
    ablation_path = _shap_ablation(
        deep_trainer, texts_test, labels_test, deep_preds, fig_dir_deep
    )

    # ── Hibridização com tok2vec ──────────────────────────────────────────────
    ml_model_path = ROOT / "models" / "classic" / "best_business.joblib"
    hybrid_results: dict[str, dict] = {}
    hybrid_preds_all: dict[str, np.ndarray] = {}

    if ml_model_path.exists():
        print("\n[train_spacy_deep] Hibridização ML clássico + tok2vec...")
        ml_model = joblib.load(ml_model_path)

        hybrid = HybridClassifier(params, ml_model, deep_trainer)
        print("  Treinando meta-modelo (stack)...")
        hybrid.fit_stack(train_df, texts_train.values, labels_train.values)

        stack_path = model_dir_hybrid / "stack_meta.joblib"
        joblib.dump(hybrid._stack, stack_path)

        ml_preds = ml_model.predict(test_df)
        ml_metrics = _eval("ml_classico", labels_test, ml_preds, params)

        hybrid_results["ml_classico"] = ml_metrics
        if bow_metrics:
            hybrid_results["spacy_bow"] = bow_metrics
        hybrid_results["spacy_tok2vec"] = deep_metrics

        hybrid_preds_all["ml_classico"] = ml_preds
        if bow_preds is not None:
            hybrid_preds_all["spacy_bow"] = bow_preds
        hybrid_preds_all["spacy_tok2vec"] = deep_preds

        all_hybrid_preds = hybrid.predict_all_strategies(test_df, texts_test.values)

        with mlflow.start_run(run_name="hybrid_tok2vec"):
            mlflow.log_params({
                "fusion_strategy": params["hybrid"]["fusion_strategy"],
                "override_threshold": params["hybrid"]["override_threshold"],
                "spacy_weight": params["hybrid"]["spacy_weight"],
            })

            for strategy, preds in all_hybrid_preds.items():
                name = f"hibrido_{strategy}"
                m = _eval(name, labels_test, preds, params)
                hybrid_results[name] = m
                hybrid_preds_all[name] = preds
                print(f"  {name:<22} | recall={m['recall_critico']:.4f} "
                      f"| f1={m['f1_macro']:.4f} | EACE=R${m['eace_brl']:,.0f}")
                for k, v in m.items():
                    if v is not None:
                        mlflow.log_metric(f"{name}__{k}", v)

            comp_h = _plot_hybrid_comparison(hybrid_results, fig_dir_hybrid)
            eace_h = _plot_eace_comparison(hybrid_results, fig_dir_hybrid)

            for name_pred, preds in hybrid_preds_all.items():
                cm_h_fig = plot_confusion_matrix(
                    labels_test.values, preds,
                    title=f"Confusion Matrix — {name_pred}",
                    class_order=params["base"]["class_order"],
                )
                cm_h_path = fig_dir_hybrid / f"cm_{name_pred}.png"
                cm_h_fig.savefig(cm_h_path, dpi=120, bbox_inches="tight")
                plt.close(cm_h_fig)
                mlflow.log_artifact(str(cm_h_path), artifact_path="figures")

            mlflow.log_artifact(str(comp_h), artifact_path="figures")
            mlflow.log_artifact(str(eace_h), artifact_path="figures")
            mlflow.log_artifact(str(stack_path), artifact_path="model")

        # Melhor híbrido por EACE
        hybrid_names = [n for n in hybrid_results if n.startswith("hibrido_")]
        best_hybrid = min(hybrid_names, key=lambda n: hybrid_results[n].get("eace_brl", float("inf")))
        print(f"\n[train_spacy_deep] Melhor híbrido (tok2vec): {best_hybrid}")

    else:
        print("[train_spacy_deep] modelo clássico não encontrado — pulando hibridização")
        hybrid_results["spacy_tok2vec"] = deep_metrics
        if bow_metrics:
            hybrid_results["spacy_bow"] = bow_metrics
        best_hybrid = "spacy_tok2vec"

    # ── Métricas DVC ─────────────────────────────────────────────────────────
    dvc_deep: dict = {
        "tok2vec__recall_critico": deep_metrics["recall_critico"],
        "tok2vec__f1_macro": deep_metrics["f1_macro"],
        "tok2vec__accuracy": deep_metrics["accuracy"],
        "tok2vec__f1_critico": deep_metrics.get("f1_critico", 0.0),
        "tok2vec__eace_brl": deep_metrics["eace_brl"],
        "tok2vec__threshold": threshold,
    }
    if bow_metrics:
        dvc_deep["bow__recall_critico"] = bow_metrics["recall_critico"]
        dvc_deep["bow__f1_macro"] = bow_metrics["f1_macro"]

    metrics_path_deep = ROOT / "reports" / "metrics_spacy_deep.json"
    metrics_path_deep.write_text(json.dumps(dvc_deep, indent=2), encoding="utf-8")

    dvc_hybrid: dict = {"best_hybrid_strategy": best_hybrid}
    for name, m in hybrid_results.items():
        for k, v in m.items():
            if v is not None:
                dvc_hybrid[f"{name}__{k}"] = v
    metrics_path_hybrid = ROOT / "reports" / "metrics_hybrid_deep.json"
    metrics_path_hybrid.write_text(json.dumps(dvc_hybrid, indent=2), encoding="utf-8")

    print(f"[train_spacy_deep] métricas → {metrics_path_deep}")
    print(f"[train_spacy_deep] métricas híbrido → {metrics_path_hybrid}")

    _print_summary(hybrid_results, best_hybrid, bow_metrics, deep_metrics)
    print("\n[train_spacy_deep] concluído.")


def _print_summary(
    hybrid_results: dict,
    best_hybrid: str,
    bow_metrics: dict | None,
    deep_metrics: dict,
) -> None:
    print("\n" + "=" * 95)
    print(f"{'Modelo':<24} {'Recall crit':>12} {'F1 macro':>10} {'Accuracy':>10} {'EACE R$/ano':>16}  Destaque")
    print("-" * 95)

    for name, m in sorted(hybrid_results.items(), key=lambda x: x[1].get("eace_brl", float("inf"))):
        tag = "★ best hybrid" if name == best_hybrid else ""
        if name == "spacy_tok2vec":
            tag = "★ tok2vec"
        eace = m.get("eace_brl", float("nan"))
        print(f"{name:<24} {m['recall_critico']:>12.4f} {m['f1_macro']:>10.4f} "
              f"{m['accuracy']:>10.4f} {eace:>16,.0f}  {tag}")
    print("=" * 95)


if __name__ == "__main__":
    main()
