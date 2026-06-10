"""
Stage DVC: eda

Executa toda a análise exploratória do NB01 e persiste:
  - reports/eda_report.json      (métricas serializáveis)
  - reports/figures/eda/*.png    (plots)

Uso:
    python src/pipeline/eda_report.py
    dvc repro eda
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # sem display — roda em CI/headless

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from src.analysis.eda import (
    apply_plot_style,
    association_table,
    imbalance_summary,
    plot_annotation_breakdown,
    plot_annotation_coverage_by_feature,
    plot_association_heatmap,
    plot_categorical_vs_class,
    plot_class_distribution,
    plot_hour_of_day,
    plot_shift_vs_class_heatmap,
    plot_split_summary,
    plot_temporal_trend,
    plot_text_length_distribution,
    plot_top_tokens_by_class,
    save_figure,
    text_length_stats_by_class,
)


def _load_config() -> dict:
    with open(ROOT / "configs" / "eda.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(ROOT / cfg["paths"]["raw"])
    train = pd.read_parquet(ROOT / cfg["paths"]["train"])
    test = pd.read_parquet(ROOT / cfg["paths"]["test"])
    unannotated = pd.read_parquet(ROOT / cfg["paths"]["unannotated"])
    return raw, train, test, unannotated


def _build_report(raw: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Serializa as métricas chave como dict — salvo em eda_report.json."""
    annotated = raw[raw["anotado"]]
    counts = annotated["classe_risco"].value_counts()
    fracs = annotated["classe_risco"].value_counts(normalize=True)
    majority = int(counts.max())
    minority = int(counts.min())

    return {
        "total_records": len(raw),
        "annotated": int(raw["anotado"].sum()),
        "unannotated": int((~raw["anotado"]).sum()),
        "annotation_coverage": round(raw["anotado"].mean(), 4),
        "mislabeled": int(raw["ruido"].sum()),
        "ambiguous": int(raw["ambiguo"].sum()),
        "class_counts": counts.to_dict(),
        "class_fractions": {k: round(v, 4) for k, v in fracs.items()},
        "imbalance_ratio": round(majority / minority, 2),
        "train_size": len(train),
        "test_size": len(test),
        "text_length_mean": round(raw["relato"].str.split().str.len().mean(), 1),
        "text_length_std": round(raw["relato"].str.split().str.len().std(), 1),
    }


def main() -> None:
    cfg = _load_config()
    apply_plot_style(cfg)

    raw, train, test, unannotated = _load_data(cfg)

    out_json = ROOT / "reports" / "eda_report.json"
    out_figs = ROOT / "reports" / "figures" / "eda"
    out_figs.mkdir(parents=True, exist_ok=True)

    report = _build_report(raw, train, test)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[eda] report salvo → {out_json}")

    plots = [
        ("class_distribution", plot_class_distribution(raw, cfg)),
        ("annotation_breakdown", plot_annotation_breakdown(raw, cfg)),
        ("text_length", plot_text_length_distribution(raw, cfg)),
        ("top_tokens_by_class", plot_top_tokens_by_class(raw, cfg)),
        ("area_fpso_vs_class", plot_categorical_vs_class(raw, "area_fpso", cfg)),
        ("fator_risco_vs_class", plot_categorical_vs_class(raw, "fator_risco", cfg)),
        ("association_heatmap", plot_association_heatmap(raw, cfg)),
        ("temporal_trend", plot_temporal_trend(raw, cfg)),
        ("hour_of_day", plot_hour_of_day(raw, cfg)),
        ("shift_heatmap", plot_shift_vs_class_heatmap(raw, cfg)),
        ("annotation_coverage_area", plot_annotation_coverage_by_feature(raw, "area_fpso", cfg)),
        ("split_summary", plot_split_summary(train, test, unannotated, cfg)),
    ]

    dpi = cfg["plots"]["dpi"]
    fmt = cfg["plots"]["save_format"]
    for name, fig in plots:
        path = out_figs / f"{name}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        fig.clf()
        import matplotlib.pyplot as plt
        plt.close("all")
        print(f"[eda] figura salva → {path}")

    print(f"[eda] concluído. {len(plots)} figuras, métricas em {out_json}")


if __name__ == "__main__":
    main()
