"""
Loop de treino com busca de hiperparâmetros e tracking MLflow.

Recebe um pipeline sklearn (preprocessador + classificador), executa
RandomizedSearchCV e loga params/métricas/artefatos no MLflow.

Design: sem side effects além do MLflow run e do arquivo salvo em disco.
O script de entrada (train_classic.py) controla quais modelos rodar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.evaluation.metrics import classification_metrics, eace_from_predictions, make_eace_scorer, per_class_report


def train_with_search(
    pipeline: Pipeline,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    params: dict,
    artifact_dir: Path,
) -> tuple[Pipeline, dict]:
    """Treina com RandomizedSearchCV, avalia no test set, loga no MLflow.

    Returns:
        best_pipeline: melhor estimador refitado no conjunto completo de treino.
        metrics: dict de métricas no test set.
    """
    cv_cfg = params["cv"]
    mlflow_cfg = params["mlflow"]

    n_iter = min(cv_cfg["n_iter_random"], _count_combinations(param_grid))

    # recall_critico como scoring do CV: pressiona diretamente a não perder incidentes críticos.
    # EACE fica para comparação final no test set — não como scorer de CV, pois dentro dos
    # folds o volume de crítico é pequeno demais para o sinal dominar sobre baixo/medio.
    scoring_name = cv_cfg["scoring"]
    if scoring_name == "recall_critico":
        from sklearn.metrics import make_scorer, recall_score
        scoring = make_scorer(recall_score, labels=["critico"], average="macro", zero_division=0)
    else:
        scoring = scoring_name

    # n_jobs=1 no SearchCV: evita reutilização de workers loky entre modelos no Windows.
    # O paralelismo real vem de dentro dos estimadores (RF, XGB já têm n_jobs=-1).
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv_cfg["n_splits"],
        scoring=scoring,
        n_jobs=1,
        verbose=cv_cfg["verbose"],
        random_state=params["base"]["random_seed"],
        refit=True,
    )

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({"model": model_name})
        cv_scoring_name = "eace" if "eace" in params else cv_cfg["scoring"]
        mlflow.log_params({"cv_scoring": cv_scoring_name, "cv_folds": cv_cfg["n_splits"]})

        search.fit(X_train, y_train)

        best_params = {k: v for k, v in search.best_params_.items()}
        mlflow.log_params(best_params)
        mlflow.log_metric("cv_best_score", round(search.best_score_, 4))

        y_pred = search.best_estimator_.predict(X_test)
        metrics = classification_metrics(y_test, y_pred, params["base"]["class_order"])

        # EACE: métrica de negócio real (R$/ano) — menor é melhor
        if "eace" in params:
            eace = eace_from_predictions(y_test, y_pred, params["eace"],
                                         params["base"]["class_order"])
            metrics["eace_brl"] = eace
            mlflow.log_metric("eace_brl", eace)

        mlflow.log_metrics({k: v for k, v in metrics.items() if k != "eace_brl"})

        # salva o modelo
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model_path = artifact_dir / f"{model_name}.joblib"
        joblib.dump(search.best_estimator_, model_path)
        mlflow.log_artifact(str(model_path))

        # loga confusion matrix como figura
        _log_confusion_matrix(y_test, y_pred, model_name, params["base"]["class_order"])

        eace_str = f" | EACE=R${metrics.get('eace_brl', 0):,.0f}" if "eace_brl" in metrics else ""
        print(f"[train] {model_name} | cv={search.best_score_:.4f} | "
              f"f1_macro={metrics['f1_macro']:.4f} | recall_critico={metrics['recall_critico']:.4f}"
              f"{eace_str}")

    return search.best_estimator_, metrics


def _log_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    class_order: list[str],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.evaluation.metrics import plot_confusion_matrix

    fig = plot_confusion_matrix(y_true, y_pred, title=f"Confusion Matrix — {model_name}",
                                class_order=class_order)
    tmp_path = Path(f"/tmp/{model_name}_cm.png") if Path("/tmp").exists() \
               else Path(f"{model_name}_cm.png")
    fig.savefig(tmp_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(tmp_path), artifact_path="figures")


def _count_combinations(param_grid: dict) -> int:
    total = 1
    for v in param_grid.values():
        if hasattr(v, "__len__"):
            total *= len(v)
    return max(total, 1)
