"""
Stage DVC: feature_analysis

Executa toda a análise de features do NB02 e persiste:
  - reports/feature_report.json        (métricas serializáveis)
  - reports/figures/features/*.png     (plots)

Uso:
    python src/pipeline/feature_analysis.py
    dvc repro feature_analysis
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from src.features.analysis import (
    feature_importance_table,
    hypothesis_tests_summary,
    pipeline_spec_table,
    plot_cramers_v_ranking,
    plot_temporal_features_vs_class,
    plot_text_length_by_class_boxplot,
    plot_tfidf_projection,
    temporal_signal_table,
)


def _load_config() -> dict:
    with open(ROOT / "configs" / "eda.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_report(df: pd.DataFrame, cfg: dict) -> dict:
    """Serializa as métricas chave da análise de features."""
    importance = feature_importance_table(df, cfg)
    threshold = cfg["feature_analysis"]["cramers_v_threshold"]

    strong_features = importance[importance["sinal_forte"]]["feature"].tolist()
    all_v = importance.set_index("feature")["cramers_v"].to_dict()

    hypothesis = hypothesis_tests_summary(df, cfg)
    significant = hypothesis[hypothesis["significativo"]]["feature"].tolist()

    return {
        "cramers_v_threshold": threshold,
        "strong_signal_features": strong_features,
        "cramers_v_by_feature": {k: float(v) for k, v in all_v.items()},
        "significant_features_chi2_kruskal": significant,
        "n_train": len(df),
        "n_annotated": int(df["anotado"].sum()),
    }


def main() -> None:
    cfg = _load_config()

    import matplotlib.pyplot as plt
    style = cfg["plots"].get("style", "seaborn-v0_8-whitegrid")
    try:
        plt.style.use(style)
    except OSError:
        plt.style.use("seaborn-v0_8-whitegrid")
    import seaborn as sns
    sns.set_context("notebook", font_scale=cfg["plots"].get("font_scale", 1.1))

    train = pd.read_parquet(ROOT / cfg["paths"]["train"])

    out_json = ROOT / "reports" / "feature_report.json"
    out_figs = ROOT / "reports" / "figures" / "features"
    out_figs.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    report = _build_report(train, cfg)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[feature_analysis] report salvo → {out_json}")

    dpi = cfg["plots"]["dpi"]
    fmt = cfg["plots"]["save_format"]

    plots = [
        ("cramers_v_ranking", plot_cramers_v_ranking(train, cfg)),
        ("text_length_boxplot", plot_text_length_by_class_boxplot(train, cfg)),
        ("tfidf_projection", plot_tfidf_projection(train, cfg)),
        ("temporal_features", plot_temporal_features_vs_class(train, cfg)),
    ]

    for name, fig in plots:
        path = out_figs / f"{name}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        fig.clf()
        plt.close("all")
        print(f"[feature_analysis] figura salva → {path}")

    # Pipeline spec como CSV para consulta rápida
    spec = pipeline_spec_table(train, cfg)
    spec_path = ROOT / "reports" / "pipeline_spec.csv"
    spec.to_csv(spec_path, index=False)
    print(f"[feature_analysis] pipeline spec salva → {spec_path}")

    print(f"[feature_analysis] concluído. {len(plots)} figuras, métricas em {out_json}")


if __name__ == "__main__":
    main()
