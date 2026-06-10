"""
Stage DVC: train_classic

Treina os modelos clássicos de baseline com busca de hiperparâmetros guiada
por params.yaml e tracking via MLflow.

Persiste:
  - models/classic/<model>.joblib          (cada modelo individual)
  - models/classic/best_business.joblib    (melhor por KPI de negócio)
  - models/classic/best_statistical.joblib (melhor por teste de McNemar)
  - reports/metrics_classic.json           (métricas — lido pelo DVC)
  - reports/model_selection_report.json    (comparação negócio vs. estatística)
  - reports/figures/classic/*.png

Uso:
    python src/pipeline/train_classic.py
    dvc repro train_classic
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline

from src.evaluation.metrics import (
    compare_models_table,
    eace_breakdown,
    plot_metrics_comparison,
)
from src.features.build import build_feature_pipeline, prepare_dataframe
from src.models.classic import MODEL_BUILDERS, get_param_grid
from src.models.train import train_with_search


def _load_params() -> dict:
    with open(ROOT / "params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


LEAKAGE_COLS = ["ruido", "ambiguo", "anotado"]


def _load_data(params: dict) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    with open(ROOT / "configs" / "eda.yaml", encoding="utf-8") as f:
        eda_cfg = yaml.safe_load(f)

    train = pd.read_parquet(ROOT / eda_cfg["paths"]["train"])
    test = pd.read_parquet(ROOT / eda_cfg["paths"]["test"])

    # Remove colunas de metadado de anotação — não existem em produção no momento do relato
    train = train.drop(columns=[c for c in LEAKAGE_COLS if c in train.columns])
    test  = test.drop(columns=[c for c in LEAKAGE_COLS if c in test.columns])

    target = params["base"]["target_col"]
    X_train = prepare_dataframe(train, params)
    X_test = prepare_dataframe(test, params)
    y_train = train[target]
    y_test = test[target]

    return X_train, y_train, X_test, y_test


def _build_fresh_pipeline(model_name: str, params: dict) -> Pipeline:
    """Cria um Pipeline novo a cada chamada — preprocessador nunca é compartilhado."""
    feature_pipeline = build_feature_pipeline(params)
    classifier = MODEL_BUILDERS[model_name](params)
    return Pipeline([
        ("preprocessor", feature_pipeline.named_steps["preprocessor"]),
        ("classifier", classifier),
    ])


def _mcnemar_test(y_true, pred_a: np.ndarray, pred_b: np.ndarray) -> tuple[float, float]:
    """Teste de McNemar entre dois classificadores.

    Compara se a diferença de erros entre modelo A e B é estatisticamente
    significativa — complementa a seleção por KPI de negócio.

    Returns: (statistic, p_value)
    """
    from statsmodels.stats.contingency_tables import mcnemar

    correct_a = pred_a == np.array(y_true)
    correct_b = pred_b == np.array(y_true)

    # tabela de contingência: [[ambos certos, só A certo], [só B certo, ambos errados]]
    n00 = int((~correct_a & ~correct_b).sum())
    n01 = int((~correct_a & correct_b).sum())
    n10 = int((correct_a & ~correct_b).sum())
    n11 = int((correct_a & correct_b).sum())

    table = [[n11, n10], [n01, n00]]
    result = mcnemar(table, exact=False, correction=True)
    return float(result.statistic), float(result.pvalue)


def _select_by_mcnemar(
    predictions: dict[str, np.ndarray],
    y_true: pd.Series,
    metrics: dict[str, dict],
    cv_metric: str,
    alpha: float,
) -> str:
    """Seleciona o melhor modelo via McNemar: ranqueia por cv_metric, depois testa
    se o primeiro é significativamente melhor que o segundo.

    Se a diferença não é significativa (p >= alpha), mantém o ranking mas sinaliza.
    Retorna o nome do modelo selecionado.
    """
    ranked = sorted(metrics.keys(), key=lambda n: metrics[n].get(cv_metric, 0), reverse=True)
    best = ranked[0]

    if len(ranked) < 2:
        return best

    second = ranked[1]
    stat, pval = _mcnemar_test(y_true, predictions[best], predictions[second])

    print(f"\n[selection] McNemar: {best} vs {second}")
    print(f"  statistic={stat:.4f}  p-value={pval:.4f}  alpha={alpha}")
    if pval < alpha:
        print(f"  → diferença SIGNIFICATIVA: {best} é estatisticamente melhor")
    else:
        print(f"  → diferença NÃO significativa: modelos equivalentes (escolhendo {best} por {cv_metric})")

    return best


def _build_selection_report(
    metrics: dict[str, dict],
    predictions: dict[str, np.ndarray],
    y_true: pd.Series,
    params: dict,
    best_business: str,
    best_statistical: str,
) -> dict:
    sel_cfg = params["model_selection"]
    alpha = sel_cfg["alpha"]

    pairwise = []
    model_names = list(metrics.keys())
    for i, a in enumerate(model_names):
        for b in model_names[i + 1:]:
            stat, pval = _mcnemar_test(y_true, predictions[a], predictions[b])
            pairwise.append({
                "model_a": a,
                "model_b": b,
                "mcnemar_stat": round(stat, 4),
                "p_value": round(pval, 4),
                "significant": pval < alpha,
                "winner": a if metrics[a].get(sel_cfg["cv_metric"], 0) >=
                               metrics[b].get(sel_cfg["cv_metric"], 0) else b,
            })

    biz_metric = sel_cfg["business_metric"]
    # EACE minimiza; outras métricas maximizam
    if biz_metric == "eace":
        ranking_biz = sorted(
            model_names,
            key=lambda n: metrics[n].get("eace_brl", float("inf")),
        )
        eace_summary = {
            n: {"eace_brl": metrics[n].get("eace_brl"), "recall_critico": metrics[n].get("recall_critico")}
            for n in ranking_biz
        }
    else:
        ranking_biz = sorted(model_names, key=lambda n: metrics[n].get(biz_metric, 0), reverse=True)
        eace_summary = {}

    return {
        "best_by_business_metric": best_business,
        "best_by_statistical_test": best_statistical,
        "business_metric": biz_metric,
        "statistical_metric": sel_cfg["cv_metric"],
        "alpha": alpha,
        "same_winner": best_business == best_statistical,
        "eace_summary": eace_summary,
        "pairwise_mcnemar": pairwise,
        "ranking_by_business": ranking_biz,
        "ranking_by_cv_metric": sorted(
            model_names,
            key=lambda n: metrics[n].get(sel_cfg["cv_metric"], 0),
            reverse=True,
        ),
    }


def main() -> None:
    params = _load_params()
    mlflow_cfg = params["mlflow"]

    tracking_uri = mlflow_cfg["tracking_uri"]
    if "://" not in tracking_uri:
        tracking_uri = (ROOT / tracking_uri).as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    X_train, y_train, X_test, y_test = _load_data(params)
    print(f"[train_classic] train={len(X_train)} | test={len(X_test)}")

    artifact_dir = ROOT / "models" / "classic"
    fig_dir = ROOT / "reports" / "figures" / "classic"
    fig_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}
    all_predictions: dict[str, np.ndarray] = {}

    for model_name in MODEL_BUILDERS:
        print(f"\n[train_classic] === {model_name} ===")
        # _build_fresh_pipeline garante que cada modelo tem seu próprio
        # ColumnTransformer — sem estado compartilhado entre iterações
        pipeline = _build_fresh_pipeline(model_name, params)
        param_grid = get_param_grid(model_name, params)

        best_pipeline, metrics = train_with_search(
            pipeline=pipeline,
            param_grid=param_grid,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_name=model_name,
            params=params,
            artifact_dir=artifact_dir,
        )
        all_results[model_name] = metrics
        all_predictions[model_name] = best_pipeline.predict(X_test)

    # ── Seleção 1: KPI de negócio — EACE (minimizar) ─────────────────────────
    biz_metric = params["model_selection"]["business_metric"]
    # EACE: menor é melhor; qualquer outra métrica: maior é melhor
    if biz_metric == "eace":
        best_business = min(
            all_results,
            key=lambda n: all_results[n].get("eace_brl", float("inf")),
        )
        biz_value = all_results[best_business].get("eace_brl", float("inf"))
        print(f"\n[selection] Melhor por EACE: {best_business} = R${biz_value:,.0f}/ano")
    else:
        best_business = max(all_results, key=lambda n: all_results[n].get(biz_metric, 0))
        biz_value = all_results[best_business].get(biz_metric, 0)
        print(f"\n[selection] Melhor por {biz_metric}: {best_business} = {biz_value:.4f}")

    # ── Seleção 2: McNemar sobre f1_macro (CV metric) ─────────────────────────
    try:
        best_statistical = _select_by_mcnemar(
            predictions=all_predictions,
            y_true=y_test,
            metrics=all_results,
            cv_metric=params["model_selection"]["cv_metric"],
            alpha=params["model_selection"]["alpha"],
        )
    except ImportError:
        print("[selection] statsmodels não instalado — pulando McNemar, usando ranking por f1_macro")
        best_statistical = max(all_results, key=lambda n: all_results[n].get("f1_macro", 0))

    # ── Salvar melhores modelos ────────────────────────────────────────────────
    import shutil
    shutil.copy2(artifact_dir / f"{best_business}.joblib",
                 artifact_dir / "best_business.joblib")
    shutil.copy2(artifact_dir / f"{best_statistical}.joblib",
                 artifact_dir / "best_statistical.joblib")

    # ── Relatório de seleção ──────────────────────────────────────────────────
    selection_report = _build_selection_report(
        metrics=all_results,
        predictions=all_predictions,
        y_true=y_test,
        params=params,
        best_business=best_business,
        best_statistical=best_statistical,
    )
    sel_path = ROOT / "reports" / "model_selection_report.json"
    sel_path.write_text(json.dumps(selection_report, indent=2), encoding="utf-8")
    print(f"[train_classic] relatório de seleção → {sel_path}")

    # ── Métricas DVC (flat dict) ──────────────────────────────────────────────
    dvc_metrics: dict = {
        "best_business_model": best_business,
        "best_statistical_model": best_statistical,
    }
    for model_name, metrics in all_results.items():
        for k, v in metrics.items():
            dvc_metrics[f"{model_name}__{k}"] = v

    metrics_path = ROOT / "reports" / "metrics_classic.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(dvc_metrics, indent=2), encoding="utf-8")
    print(f"[train_classic] métricas → {metrics_path}")

    # ── Figura de comparação ──────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    fig = plot_metrics_comparison(all_results)
    fig_path = fig_dir / "models_comparison.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[train_classic] comparação → {fig_path}")

    print("\n[train_classic] concluído.")
    _print_summary(all_results, best_business, best_statistical, biz_metric)


def _print_summary(
    results: dict,
    best_business: str,
    best_statistical: str,
    biz_metric: str,
) -> None:
    has_eace = any("eace_brl" in m for m in results.values())
    print("\n" + "=" * 90)
    header = f"{'Modelo':<22} {'F1 macro':>10} {'Recall crit':>12} {'Accuracy':>10}"
    if has_eace:
        header += f"  {'EACE (R$/ano)':>16}"
    header += "  Seleção"
    print(header)
    print("-" * 90)
    for name, m in sorted(results.items(), key=lambda x: x[1].get("f1_macro", 0), reverse=True):
        tags = []
        if name == best_business:
            tags.append(f"★ {biz_metric}")
        if name == best_statistical:
            tags.append("★ McNemar")
        tag_str = "  " + " | ".join(tags) if tags else ""
        line = (f"{name:<22} {m['f1_macro']:>10.4f} {m['recall_critico']:>12.4f} "
                f"{m['accuracy']:>10.4f}")
        if has_eace:
            eace = m.get("eace_brl", float("nan"))
            line += f"  {eace:>16,.0f}"
        line += tag_str
        print(line)
    print("=" * 90)


if __name__ == "__main__":
    main()
