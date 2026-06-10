"""
Stage: explain_classic

Gera análise de explicabilidade SHAP para o melhor modelo clássico treinado.

Persiste:
  - reports/figures/classic/shap_beeswarm.png       (importância global)
  - reports/figures/classic/shap_bar_critico.png     (top tokens por classe)
  - reports/figures/classic/shap_waterfall_*.png     (exemplos individuais)
  - reports/shap_classic.json                        (top features por classe)

Uso:
    python src/pipeline/explain_classic.py
    dvc repro explain_classic   (após adicionar o stage no dvc.yaml)

Dependências:
    - models/classic/best_business.joblib  (gerado por train_classic)
    - shap >= 0.42
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import scipy.sparse
import shap
import yaml

from src.config.loader import load_config
from src.features.build import prepare_dataframe


LEAKAGE_COLS = ["ruido", "ambiguo", "anotado"]
CLASS_ORDER = ["baixo", "medio", "alto", "critico"]


def _load_params() -> dict:
    with open(ROOT / "params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_test(params: dict, eda_cfg: dict) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    test_raw = pd.read_parquet(ROOT / eda_cfg["paths"]["test"])
    test_raw = test_raw.drop(columns=[c for c in LEAKAGE_COLS if c in test_raw.columns])
    test = prepare_dataframe(test_raw, params)
    y_test = test[params["base"]["target_col"]]
    return test_raw, y_test, test


def _plot_shap_bar(
    shap_vals: np.ndarray,
    feature_names: np.ndarray,
    mean_signed: np.ndarray,
    top_n: int,
    fig_dir: Path,
) -> Path:
    """Bar chart: top N features por impacto absoluto médio na classe crítico."""
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_n]

    top_feats = feature_names[top_idx]
    top_vals = mean_abs[top_idx]
    top_signed = mean_signed[top_idx]
    colors = ["#B71C1C" if v > 0 else "#1565C0" for v in top_signed]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top_feats[::-1], top_vals[::-1], color=colors[::-1], edgecolor="white")
    ax.set_xlabel("|SHAP value| médio (impacto na predição)")
    ax.set_title(
        f"Top {top_n} features — impacto na classe crítico\n"
        "(vermelho = empurra para crítico | azul = afasta)",
        fontsize=11,
    )
    ax.tick_params(axis="y", labelsize=8)

    legend_elements = [
        mpatches.Patch(facecolor="#B71C1C", label="Empurra → crítico"),
        mpatches.Patch(facecolor="#1565C0", label="Afasta ← crítico"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    fig.tight_layout()

    path = fig_dir / "shap_bar_critico.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_beeswarm(
    shap_explanation,
    fig_dir: Path,
) -> Path:
    """Beeswarm: distribuição dos SHAP values de todos os exemplos."""
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.beeswarm(shap_explanation, max_display=20, show=False, ax=ax)
    ax.set_title("SHAP — importância global (classe: crítico)", pad=14)
    fig.tight_layout()

    path = fig_dir / "shap_beeswarm.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_waterfall(
    shap_explanation,
    idx: int,
    label: str,
    pred: str,
    case_type: str,
    fig_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_explanation[idx], max_display=12, show=False)
    plt.title(
        f"SHAP Waterfall — {case_type} (true={label}, pred={pred})",
        pad=12,
    )
    plt.tight_layout()

    slug = case_type.lower().replace(" ", "_").replace("#", "")
    path = fig_dir / f"shap_waterfall_{slug}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return path


def main() -> None:
    params = _load_params()
    eda_cfg = load_config("configs/eda.yaml")

    model_path = ROOT / "models" / "classic" / "best_business.joblib"
    if not model_path.exists():
        print(f"[explain_classic] modelo não encontrado: {model_path}")
        print("  Execute: dvc repro train_classic")
        sys.exit(1)

    import joblib
    pipeline = joblib.load(model_path)
    clf = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    if not hasattr(clf, "coef_"):
        print(f"[explain_classic] {type(clf).__name__} não tem coef_ — SHAP LinearExplainer não aplicável.")
        sys.exit(0)

    test_raw, y_test, test = _load_test(params, eda_cfg)
    y_pred = pipeline.predict(test)

    fig_dir = ROOT / "reports" / "figures" / "classic"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Transformar para esparso ──────────────────────────────────────────────
    X_transformed = preprocessor.transform(test)
    if not scipy.sparse.issparse(X_transformed):
        X_transformed = scipy.sparse.csr_matrix(X_transformed)

    feature_names = np.array(preprocessor.get_feature_names_out())

    print(f"[explain_classic] {X_transformed.shape[0]} exemplos × {X_transformed.shape[1]} features")
    print(f"[explain_classic] modelo: {type(clf).__name__}")

    # ── LinearExplainer ───────────────────────────────────────────────────────
    print("[explain_classic] calculando SHAP values (LinearExplainer)...")
    explainer = shap.LinearExplainer(
        clf, X_transformed, feature_perturbation="interventional"
    )
    shap_values = explainer(X_transformed)

    # Para modelo multi-classe (LogReg OvR): shape (n, features, n_classes)
    # Para binário: (n, features)
    classes_model = list(clf.classes_) if hasattr(clf, "classes_") else CLASS_ORDER
    is_multiclass = shap_values.values.ndim == 3

    if is_multiclass and "critico" in classes_model:
        critico_idx_model = classes_model.index("critico")
        # Cria Explanation apenas para a classe crítico
        import shap as shap_lib
        shap_critico = shap_lib.Explanation(
            values=shap_values.values[:, :, critico_idx_model],
            base_values=shap_values.base_values[:, critico_idx_model]
            if shap_values.base_values.ndim == 2 else shap_values.base_values,
            data=shap_values.data,
            feature_names=feature_names.tolist(),
        )
    else:
        shap_critico = shap_values

    mean_signed = shap_critico.values.mean(axis=0)

    # ── Figura 1: bar chart ───────────────────────────────────────────────────
    bar_path = _plot_shap_bar(shap_critico.values, feature_names, mean_signed, 25, fig_dir)
    print(f"[explain_classic] bar chart → {bar_path}")

    # ── Figura 2: beeswarm ────────────────────────────────────────────────────
    try:
        bee_path = _plot_beeswarm(shap_critico, fig_dir)
        print(f"[explain_classic] beeswarm → {bee_path}")
    except Exception as e:
        print(f"[explain_classic] beeswarm falhou ({e}) — pulando")

    # ── Figura 3: waterfalls — exemplos selecionados ──────────────────────────
    y_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)
    text_col = params["features"]["text_col"]

    # Caso 1: verdadeiro positivo (true=critico, pred=critico)
    tp_mask = (y_arr == "critico") & (y_pred_arr == "critico")
    tp_indices = np.where(tp_mask)[0]

    # Caso 2: falso negativo (true=critico, pred≠critico)
    fn_mask = (y_arr == "critico") & (y_pred_arr != "critico")
    fn_indices = np.where(fn_mask)[0]

    # Caso 3: falso positivo (pred=critico, true≠critico)
    fp_mask = (y_arr != "critico") & (y_pred_arr == "critico")
    fp_indices = np.where(fp_mask)[0]

    waterfall_paths = []
    cases = [
        (tp_indices, "Verdadeiro_Positivo_1"),
        (fn_indices, "Falso_Negativo_1"),
        (fn_indices[1:], "Falso_Negativo_2"),
        (fp_indices, "Falso_Positivo_1"),
    ]

    for idx_list, case_name in cases:
        if len(idx_list) == 0:
            continue
        idx = idx_list[0]
        relato = test_raw.iloc[idx][text_col]
        print(f"\n  [{case_name}] true={y_arr[idx]} | pred={y_pred_arr[idx]}")
        print(f"  '{relato[:120]}'")
        try:
            p = _plot_waterfall(shap_critico, idx, y_arr[idx], y_pred_arr[idx], case_name, fig_dir)
            waterfall_paths.append(str(p))
            print(f"  → {p}")
        except Exception as e:
            print(f"  waterfall falhou: {e}")

    # ── JSON: top features ────────────────────────────────────────────────────
    top_n_json = 30
    mean_abs = np.abs(shap_critico.values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_n_json]

    top_features_report = {
        "model": type(clf).__name__,
        "top_features_critico": [
            {
                "feature": feature_names[i],
                "mean_abs_shap": round(float(mean_abs[i]), 6),
                "mean_signed_shap": round(float(mean_signed[i]), 6),
                "direction": "empurra_critico" if mean_signed[i] > 0 else "afasta_critico",
            }
            for i in top_idx
        ],
        "false_negatives": int(fn_mask.sum()),
        "false_positives": int(fp_mask.sum()),
        "true_positives": int(tp_mask.sum()),
        "total_critico": int((y_arr == "critico").sum()),
    }

    report_path = ROOT / "reports" / "shap_classic.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(top_features_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[explain_classic] relatório SHAP → {report_path}")

    # ── Sumário ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SHAP — top 10 tokens que mais indicam risco CRÍTICO:")
    for item in top_features_report["top_features_critico"][:10]:
        if item["direction"] == "empurra_critico":
            print(f"  {item['feature']:<30} +{item['mean_abs_shap']:.5f}")
    print("\nSHAP — top 5 tokens que REDUZEM indicação de crítico:")
    afasta = [i for i in top_features_report["top_features_critico"] if i["direction"] == "afasta_critico"]
    for item in afasta[:5]:
        print(f"  {item['feature']:<30}  {item['mean_signed_shap']:.5f}")
    print("=" * 60)
    print("\n[explain_classic] concluído.")


if __name__ == "__main__":
    main()
