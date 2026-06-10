"""
Módulo de avaliação padronizado para todos os tiers (classic, spaCy, LLM).

Todas as funções recebem y_true e y_pred como arrays 1-D e retornam dicts
ou DataFrames — sem side effects. Logging e salvamento ficam nos scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


CLASS_ORDER = ["baixo", "medio", "alto", "critico"]


def classification_metrics(
    y_true: Sequence,
    y_pred: Sequence,
    class_order: list[str] = CLASS_ORDER,
) -> dict:
    """Métricas completas: accuracy, F1 macro, F1/precision/recall por classe.

    O recall de 'critico' é o KPI do negócio — aparece destacado no dict.
    """
    labels = [c for c in class_order if c in set(y_true)]

    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    critico_recall = report.get("critico", {}).get("recall", 0.0)
    critico_f1 = report.get("critico", {}).get("f1-score", 0.0)

    metrics = {
        "accuracy": report["accuracy"],
        "f1_macro": report["macro avg"]["f1-score"],
        "f1_weighted": report["weighted avg"]["f1-score"],
        "recall_critico": critico_recall,
        "f1_critico": critico_f1,
    }

    for cls in labels:
        metrics[f"precision_{cls}"] = report[cls]["precision"]
        metrics[f"recall_{cls}"] = report[cls]["recall"]
        metrics[f"f1_{cls}"] = report[cls]["f1-score"]

    return {k: round(float(v), 4) for k, v in metrics.items()}


def per_class_report(
    y_true: Sequence,
    y_pred: Sequence,
    class_order: list[str] = CLASS_ORDER,
) -> pd.DataFrame:
    """DataFrame com precision/recall/F1/support por classe — fácil de logar."""
    labels = [c for c in class_order if c in set(y_true)]
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    rows = []
    for cls in labels:
        rows.append({
            "classe": cls,
            "precision": round(report[cls]["precision"], 4),
            "recall": round(report[cls]["recall"], 4),
            "f1": round(report[cls]["f1-score"], 4),
            "support": int(report[cls]["support"]),
        })
    return pd.DataFrame(rows).set_index("classe")


def plot_confusion_matrix(
    y_true: Sequence,
    y_pred: Sequence,
    title: str = "Matriz de Confusão",
    class_order: list[str] = CLASS_ORDER,
    figsize: tuple[int, int] = (7, 6),
) -> plt.Figure:
    labels = [c for c in class_order if c in set(y_true)]
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format=".2f")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def eace_from_predictions(
    y_true: Sequence,
    y_pred: Sequence,
    eace_params: dict,
    class_order: list[str] = CLASS_ORDER,
) -> float:
    """Calcula o Expected Annual Cost of Error (EACE) em R$/ano.

    EACE = Σ_{i,j} N × P(true=i) × P(pred=j | true=i) × Cost(i, j)

    O termo dominante é sempre crítico→baixo (R$ 3.2M × volume × miss rate).
    EACE é a métrica de negócio real — menor é melhor.

    Args:
        eace_params: seção 'eace' do params.yaml com cost_matrix,
                     annual_records e class_distribution.
    """
    labels = [c for c in class_order if c in set(y_true)]
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    cost_matrix = eace_params["cost_matrix"]
    dist = eace_params["class_distribution"]
    N = eace_params["annual_records"]

    total = 0.0
    for i, true_cls in enumerate(labels):
        for j, pred_cls in enumerate(labels):
            if true_cls == pred_cls:
                continue
            p_true = dist.get(true_cls, 0.0)
            p_error = float(cm[i, j])
            cost = cost_matrix.get(true_cls, {})
            # cost_matrix[true][pred] pode ser lista ou dict
            if isinstance(cost, list):
                pred_idx = class_order.index(pred_cls)
                c = cost[pred_idx]
            else:
                c = cost.get(pred_cls, 0)
            total += N * p_true * p_error * c

    return round(total, 2)


def eace_breakdown(
    y_true: Sequence,
    y_pred: Sequence,
    eace_params: dict,
    class_order: list[str] = CLASS_ORDER,
) -> pd.DataFrame:
    """EACE decomposto por par (true_class, pred_class) — identifica os erros mais caros."""
    labels = [c for c in class_order if c in set(y_true)]
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    cost_matrix = eace_params["cost_matrix"]
    dist = eace_params["class_distribution"]
    N = eace_params["annual_records"]

    rows = []
    for i, true_cls in enumerate(labels):
        for j, pred_cls in enumerate(labels):
            if true_cls == pred_cls:
                continue
            p_true = dist.get(true_cls, 0.0)
            p_error = float(cm[i, j])
            cost = cost_matrix.get(true_cls, {})
            if isinstance(cost, list):
                pred_idx = class_order.index(pred_cls)
                c = cost[pred_idx]
            else:
                c = cost.get(pred_cls, 0)
            eace_term = N * p_true * p_error * c
            rows.append({
                "true_class": true_cls,
                "pred_class": pred_cls,
                "error_rate": round(p_error, 4),
                "unit_cost_brl": c,
                "eace_contribution_brl": round(eace_term, 2),
            })

    return (
        pd.DataFrame(rows)
        .sort_values("eace_contribution_brl", ascending=False)
        .reset_index(drop=True)
    )


def make_eace_scorer(eace_params: dict, class_order: list[str] = CLASS_ORDER):
    """Retorna um scorer sklearn que minimiza EACE (maior_é_melhor=False → negado).

    Uso: scoring=make_eace_scorer(params['eace'], params['base']['class_order'])
    """
    from sklearn.metrics import make_scorer

    def _eace(y_true, y_pred):
        return eace_from_predictions(y_true, y_pred, eace_params, class_order)

    # greater_is_better=False: sklearn negará o valor internamente
    return make_scorer(_eace, greater_is_better=False)


def compare_models_table(results: dict[str, dict]) -> pd.DataFrame:
    """Cria tabela comparativa dado um dict {model_name: metrics_dict}.

    Inclui EACE se disponível nos resultados.
    """
    rows = []
    key_metrics = ["accuracy", "f1_macro", "recall_critico", "f1_critico", "f1_weighted", "eace_brl"]
    for model_name, metrics in results.items():
        row = {"modelo": model_name}
        for k in key_metrics:
            row[k] = metrics.get(k, None)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("modelo")
    # Remove colunas completamente ausentes
    return df.dropna(axis=1, how="all")


def plot_metrics_comparison(
    results: dict[str, dict],
    metrics_to_plot: list[str] | None = None,
    figsize: tuple[int, int] = (10, 5),
) -> plt.Figure:
    """Bar chart comparando modelos nas métricas principais."""
    if metrics_to_plot is None:
        metrics_to_plot = ["f1_macro", "recall_critico", "accuracy"]

    table = compare_models_table(results)[metrics_to_plot]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(table))
    width = 0.8 / len(metrics_to_plot)
    colors = ["#1976D2", "#B71C1C", "#388E3C"]

    for i, metric in enumerate(metrics_to_plot):
        ax.bar(x + i * width, table[metric], width, label=metric,
               color=colors[i % len(colors)], edgecolor="white")

    ax.set_xticks(x + width * (len(metrics_to_plot) - 1) / 2)
    ax.set_xticklabels(table.index, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Comparação de modelos — métricas principais")
    ax.legend()
    ax.axhline(0.75, color="gray", linestyle="--", linewidth=0.8, label="Target recall@crítico")
    fig.tight_layout()
    return fig
