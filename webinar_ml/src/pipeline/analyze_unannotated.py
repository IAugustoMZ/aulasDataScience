"""
Análise de recuperação confiante dos registros não anotados.

Lógica central:
  Para cada registro e cada modelo, calculamos o Custo Esperado de Erro (CEE)
  da predição proposta:

      CEE(i) = sum_j [ P(pred=j | score) * Cost(pred_class, j) ]
              = score_vector · cost_row[pred_class]

  Se CEE(i) <= max_cee_brl  → rotulamos com confiança (recovered)
  Se CEE(i) >  max_cee_brl  → deixamos sem previsão (abstain)

  max_cee_brl é configurado em params.yaml → unannotated.max_cee_brl.
  O default (R$ 50.000) corresponde ao custo de confundir baixo↔alto —
  erros de um degrau toleráveis; erros de dois degraus (critico→baixo) nunca.

Saídas:
  reports/unannotated_recovery.json   — métricas de recuperação por modelo
  reports/figures/unannotated/        — gráficos comparativos
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
import numpy as np
import pandas as pd
import yaml

from src.config.loader import load_config
from src.features.build import prepare_dataframe
from src.models.hybrid import (
    CLASS_ORDER, LABEL_MAP, HybridClassifier, TripleHybridClassifier,
    _ml_proba, _spacy_proba,
)
from src.models.spacy_model import SpacyTextCatTrainer, SpacyDeepTextCatTrainer


# ── Config ────────────────────────────────────────────────────────────────────

def _load_params() -> dict:
    with open(ROOT / "params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_unannotated(params: dict) -> tuple[pd.DataFrame, pd.Series]:
    eda_cfg = load_config("configs/eda.yaml")
    df = pd.read_parquet(ROOT / eda_cfg["paths"]["unannotated"])
    leakage = ["ruido", "ambiguo", "anotado", "classe_risco"]
    df = df.drop(columns=[c for c in leakage if c in df.columns])
    df = prepare_dataframe(df, params)
    text_col = params["features"]["text_col"]
    return df, df[text_col]


def _build_cost_matrix(params: dict) -> np.ndarray:
    """Retorna matriz de custos (4x4) na ordem CLASS_ORDER."""
    raw = params["eace"]["cost_matrix"]
    n = len(CLASS_ORDER)
    mat = np.zeros((n, n))
    for i, true_cls in enumerate(CLASS_ORDER):
        row = raw[true_cls]
        if isinstance(row, list):
            mat[i] = row
        else:
            for j, pred_cls in enumerate(CLASS_ORDER):
                mat[i, j] = row.get(pred_cls, 0)
    return mat


# ── CEE por registro ──────────────────────────────────────────────────────────

def _cee_per_record(proba: np.ndarray, cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Para cada registro, calcula:
      pred_idx  — índice da classe predita (argmax do score)
      cee       — custo esperado de erro dessa predição

    CEE(i) = score_vector[i] · cost_matrix[pred_idx[i]]
             (produto escalar: score de cada classe × custo de errar para ela)
    """
    pred_idx = np.argmax(proba, axis=1)                     # (n,)
    cost_rows = cost_matrix[pred_idx]                       # (n, 4): custo se errar para cada classe
    cee = np.sum(proba * cost_rows, axis=1)                 # (n,): esperança de custo
    return pred_idx, cee


def _recover(
    pred_idx: np.ndarray,
    cee: np.ndarray,
    max_cee: float,
) -> np.ndarray:
    """Retorna array de labels: classe prevista se CEE <= max_cee, None caso contrário."""
    labels = np.where(
        cee <= max_cee,
        np.array([CLASS_ORDER[i] for i in pred_idx]),
        None,
    )
    return labels


# ── Métricas de recuperação ───────────────────────────────────────────────────

def _recovery_metrics(labels: np.ndarray, cee: np.ndarray, model_name: str) -> dict:
    n = len(labels)
    recovered_mask = labels != None  # noqa: E711
    n_recovered = int(recovered_mask.sum())
    n_abstain = n - n_recovered

    dist = {}
    if n_recovered > 0:
        recovered_labels = labels[recovered_mask]
        for cls in CLASS_ORDER:
            count = int((recovered_labels == cls).sum())
            dist[cls] = {"count": count, "pct_recovered": round(count / n_recovered * 100, 1)}

    cee_recovered = cee[recovered_mask] if n_recovered > 0 else np.array([])
    cee_abstained = cee[~recovered_mask] if n_abstain > 0 else np.array([])

    return {
        "model": model_name,
        "n_total": n,
        "n_recovered": n_recovered,
        "n_abstain": n_abstain,
        "recovery_rate_pct": round(n_recovered / n * 100, 1),
        "class_distribution": dist,
        "cee_recovered": {
            "mean": round(float(cee_recovered.mean()), 0) if len(cee_recovered) else None,
            "median": round(float(np.median(cee_recovered)), 0) if len(cee_recovered) else None,
            "max": round(float(cee_recovered.max()), 0) if len(cee_recovered) else None,
        },
        "cee_abstained": {
            "mean": round(float(cee_abstained.mean()), 0) if len(cee_abstained) else None,
            "min": round(float(cee_abstained.min()), 0) if len(cee_abstained) else None,
        },
    }


# ── Paleta de cores por modelo ────────────────────────────────────────────────

MODEL_COLORS = {
    "ml_classico":       "#1565C0",
    "spacy_textcat":     "#1976D2",
    "spacy_tok2vec":     "#42A5F5",
    "hibrido_override":  "#B71C1C",
    "hibrido_weighted":  "#E53935",
    "hibrido_stack":     "#FF5252",
    "triple_override_deep": "#2E7D32",
    "triple_weighted_avg":  "#43A047",
    "triple_stack":         "#66BB6A",
}


# ── Figuras ───────────────────────────────────────────────────────────────────

def _plot_recovery_bars(results: list[dict], fig_dir: Path) -> Path:
    names = [r["model"] for r in results]
    rates = [r["recovery_rate_pct"] for r in results]
    abstain = [r["n_abstain"] for r in results]
    colors = [MODEL_COLORS.get(n, "#78909C") for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(13, len(names) * 1.5), 5))

    bars = ax1.bar(range(len(names)), rates, color=colors, edgecolor="white")
    ax1.set_ylabel("% de registros rotulados")
    ax1.set_title("Taxa de recuperacao por modelo")
    ax1.set_ylim(0, 110)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    for bar, val in zip(bars, rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 1,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    bars2 = ax2.bar(range(len(names)), abstain, color=["#78909C"] * len(names), edgecolor="white")
    ax2.set_ylabel("N de registros sem previsao")
    ax2.set_title("Registros sem previsao (CEE alto demais)")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    for bar, val in zip(bars2, abstain):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 2,
                 str(val), ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    path = fig_dir / "recovery_rates.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_class_distribution(results: list[dict], fig_dir: Path) -> Path:
    """Stacked bar: distribuição de classes nos registros recuperados."""
    cls_colors = {"baixo": "#4CAF50", "medio": "#FFC107", "alto": "#FF5722", "critico": "#B71C1C"}
    names = [r["model"] for r in results]

    fig, ax = plt.subplots(figsize=(max(12, len(names) * 1.5), 5))
    bottom = np.zeros(len(names))

    for cls in CLASS_ORDER:
        vals = []
        for r in results:
            dist = r["class_distribution"]
            vals.append(dist.get(cls, {}).get("pct_recovered", 0.0))
        bars = ax.bar(names, vals, bottom=bottom, label=cls,
                      color=cls_colors[cls], edgecolor="white", alpha=0.9)
        for i, (bar, val) in enumerate(zip(bars, vals)):
            if val >= 5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bottom[i] + val / 2,
                        f"{val:.0f}%", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        bottom += np.array(vals)

    ax.set_ylabel("% dentro dos recuperados")
    ax.set_title("Distribuicao de classes nos registros rotulados com confianca")
    ax.legend(title="Classe", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 110)
    fig.tight_layout()
    path = fig_dir / "class_distribution_recovered.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_cee_histogram(cee_by_model: dict[str, np.ndarray], max_cee: float, fig_dir: Path) -> Path:
    """Histograma do CEE por modelo com linha de corte."""
    n = len(cee_by_model)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    for ax, (name, cee) in zip(axes_flat, cee_by_model.items()):
        color = MODEL_COLORS.get(name, "#78909C")
        ax.hist(cee / 1_000, bins=40, color=color, alpha=0.8, edgecolor="white")
        ax.axvline(max_cee / 1_000, color="red", linestyle="--", linewidth=1.5,
                   label=f"Corte R${max_cee/1000:.0f}k")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("CEE (R$ mil)")
        ax.set_ylabel("N registros")
        ax.legend(fontsize=7)

    for ax in axes_flat[len(cee_by_model):]:
        ax.set_visible(False)

    fig.suptitle("Distribuicao do Custo Esperado de Erro por modelo", fontsize=11)
    fig.tight_layout()
    path = fig_dir / "cee_histogram.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_recovery_vs_cee(results: list[dict], max_cee_values: list[float],
                           cee_by_model: dict[str, np.ndarray], fig_dir: Path) -> Path:
    """Curva: taxa de recuperação em função do limite de CEE."""
    fig, ax = plt.subplots(figsize=(11, 5))

    for name, cee in cee_by_model.items():
        color = MODEL_COLORS.get(name, "#78909C")
        rates = [float((cee <= thr).mean() * 100) for thr in max_cee_values]
        ax.plot([t / 1_000 for t in max_cee_values], rates,
                label=name, color=color, linewidth=2, marker="o", markersize=3)

    ax.set_xlabel("Limite de CEE (R$ mil)")
    ax.set_ylabel("% de registros rotulados")
    ax.set_title("Trade-off: tolerancia ao risco x cobertura de rotulacao")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    path = fig_dir / "recovery_vs_cee_threshold.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_critico_rate_comparison(results: list[dict], fig_dir: Path) -> Path:
    """Barra horizontal: % de crítico dentro dos recuperados por modelo."""
    names = [r["model"] for r in results]
    critico_pcts = [
        r["class_distribution"].get("critico", {}).get("pct_recovered", 0.0)
        for r in results
    ]
    colors = [MODEL_COLORS.get(n, "#78909C") for n in names]

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.55)))
    bars = ax.barh(range(len(names)), critico_pcts, color=colors, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("% critico nos registros recuperados")
    ax.set_title("Deteccao de critico nos nao anotados — por modelo")
    ax.axvline(0, color="black", linewidth=0.5)
    for bar, val in zip(bars, critico_pcts):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    path = fig_dir / "critico_rate_comparison.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Carregamento do triple stack ──────────────────────────────────────────────

def _triple_stack_proba(
    ml_model, bow_trainer, deep_trainer,
    X: pd.DataFrame, texts: np.ndarray,
    stack_path: Path,
) -> np.ndarray:
    """Gera probabilidades do meta-modelo triple a partir do arquivo salvo.

    O meta-modelo é treinado com LabelEncoder que mapeia CLASS_ORDER → [0,1,2,3],
    então as colunas de predict_proba já estão na ordem de CLASS_ORDER.
    """
    saved = joblib.load(stack_path)
    meta = saved["meta"] if isinstance(saved, dict) else saved

    ml_p   = _ml_proba(ml_model, X)
    bow_p  = _spacy_proba(bow_trainer, texts)
    deep_p = _spacy_proba(deep_trainer, texts)
    X_meta = np.hstack([ml_p, bow_p, deep_p])
    # classes_ são inteiros [0,1,2,3] mapeados na ordem de CLASS_ORDER pelo LabelEncoder
    return meta.predict_proba(X_meta)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    params = _load_params()
    cost_matrix = _build_cost_matrix(params)
    max_cee = params.get("unannotated", {}).get("max_cee_brl", 50_000)

    print(f"[unannotated] limite de CEE = R${max_cee:,.0f}")

    df, texts = _load_unannotated(params)
    print(f"[unannotated] {len(df)} registros nao anotados")

    fig_dir = ROOT / "reports" / "figures" / "unannotated"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Caminhos dos modelos ──────────────────────────────────────────────────
    ml_path        = ROOT / "models" / "classic" / "best_business.joblib"
    spacy_path     = ROOT / "models" / "spacy" / "textcat"
    deep_path      = ROOT / "models" / "spacy" / "textcat_deep"
    stack_path     = ROOT / "models" / "hybrid" / "stack_meta.joblib"
    triple_path    = ROOT / "models" / "hybrid_full" / "triple_stack_meta.joblib"

    ml_model    = joblib.load(ml_path)
    bow_trainer = SpacyTextCatTrainer.load(spacy_path, params)
    stack_fusion = joblib.load(stack_path)

    deep_available   = deep_path.exists()
    triple_available = triple_path.exists()

    deep_trainer = None
    if deep_available:
        deep_trainer = SpacyDeepTextCatTrainer.load(deep_path, params)
        print("[unannotated] modelo tok2vec carregado")
    else:
        print("[unannotated] AVISO: textcat_deep nao encontrado — pulando modelos deep e triple")

    # ── Probabilidades brutas ─────────────────────────────────────────────────
    print("\n[unannotated] Gerando scores...")

    ml_proba = _ml_proba(ml_model, df)                           # (n, 4)
    sp_proba = _spacy_proba(bow_trainer, texts.values)            # (n, 4)

    override_thr = params["hybrid"]["override_threshold"]
    spacy_w      = params["hybrid"]["spacy_weight"]
    critico_idx  = CLASS_ORDER.index("critico")

    # override BOW
    override_proba = ml_proba.copy()
    override_mask  = sp_proba[:, critico_idx] >= override_thr
    override_proba[override_mask] = 0.0
    override_proba[override_mask, critico_idx] = 1.0

    # weighted BOW
    weighted_proba = (1 - spacy_w) * ml_proba + spacy_w * sp_proba

    # stack ML+BOW
    stack_raw    = stack_fusion.predict_proba(ml_proba, sp_proba)
    stack_classes = list(stack_fusion._le.classes_)
    stack_idx    = [stack_classes.index(c) for c in CLASS_ORDER if c in stack_classes]
    stack_proba  = stack_raw[:, stack_idx]

    models_proba: dict[str, np.ndarray] = {
        "ml_classico":      ml_proba,
        "spacy_textcat":    sp_proba,
        "hibrido_override": override_proba,
        "hibrido_weighted": weighted_proba,
        "hibrido_stack":    stack_proba,
    }

    # ── Modelos deep e triple ─────────────────────────────────────────────────
    if deep_available and deep_trainer is not None:
        deep_proba = _spacy_proba(deep_trainer, texts.values)     # (n, 4)
        models_proba["spacy_tok2vec"] = deep_proba

        # triple override_deep
        triple_ov = ml_proba.copy()
        triple_ov_mask = deep_proba[:, critico_idx] >= override_thr
        triple_ov[triple_ov_mask] = 0.0
        triple_ov[triple_ov_mask, critico_idx] = 1.0
        models_proba["triple_override_deep"] = triple_ov

        # triple weighted_avg
        sp_avg = 0.4 * sp_proba + 0.6 * deep_proba
        models_proba["triple_weighted_avg"] = (1 - spacy_w) * ml_proba + spacy_w * sp_avg

        # triple stack
        if triple_available:
            try:
                triple_proba = _triple_stack_proba(
                    ml_model, bow_trainer, deep_trainer,
                    df, texts.values, triple_path,
                )
                models_proba["triple_stack"] = triple_proba
                print("[unannotated] triple_stack carregado")
            except Exception as exc:
                print(f"[unannotated] AVISO: triple_stack falhou ({exc}) — pulando")
        else:
            print("[unannotated] AVISO: triple_stack_meta.joblib nao encontrado — pulando triple_stack")

    # ── CEE e recuperação por modelo ──────────────────────────────────────────
    print(f"\n[unannotated] Analise com CEE <= R${max_cee:,.0f}:\n")
    all_results: list[dict] = []
    cee_by_model: dict[str, np.ndarray] = {}

    for name, proba in models_proba.items():
        pred_idx, cee = _cee_per_record(proba, cost_matrix)
        cee_by_model[name] = cee
        labels  = _recover(pred_idx, cee, max_cee)
        metrics = _recovery_metrics(labels, cee, name)
        all_results.append(metrics)

        print(f"  {name:<24} | recuperados={metrics['n_recovered']:>4} "
              f"({metrics['recovery_rate_pct']:>5.1f}%) | "
              f"abstencao={metrics['n_abstain']:>4} | "
              f"CEE medio recuperados=R${metrics['cee_recovered']['mean'] or 0:>10,.0f}")
        for cls in CLASS_ORDER:
            d = metrics["class_distribution"].get(cls, {})
            if d:
                print(f"    {cls:<10}: {d['count']:>4} ({d['pct_recovered']:>5.1f}%)")

    # ── Curva de sensibilidade ao limite de CEE ───────────────────────────────
    cee_grid = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000,
                250_000, 500_000, 1_000_000, 3_200_000]

    # ── Figuras ───────────────────────────────────────────────────────────────
    print("\n[unannotated] Gerando figuras...")
    p1 = _plot_recovery_bars(all_results, fig_dir)
    p2 = _plot_class_distribution(all_results, fig_dir)
    p3 = _plot_cee_histogram(cee_by_model, max_cee, fig_dir)
    p4 = _plot_recovery_vs_cee(all_results, cee_grid, cee_by_model, fig_dir)
    p5 = _plot_critico_rate_comparison(all_results, fig_dir)
    for p in [p1, p2, p3, p4, p5]:
        print(f"  -> {p.name}")

    # ── Salva JSON ────────────────────────────────────────────────────────────
    report = {
        "max_cee_brl": max_cee,
        "n_unannotated": len(df),
        "models": all_results,
    }
    out = ROOT / "reports" / "unannotated_recovery.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[unannotated] relatorio -> {out}")

    _print_summary(all_results, max_cee)


def _print_summary(results: list[dict], max_cee: float) -> None:
    print(f"\n{'='*90}")
    print(f"Limite de CEE: R${max_cee:,.0f}  |  N total: {results[0]['n_total']}")
    print(f"{'='*90}")
    print(f"{'Modelo':<24} {'Recuperados':>11} {'Taxa':>7} {'Abstencao':>10} "
          f"{'critico':>8} {'alto':>7} {'medio':>7} {'baixo':>7}")
    print(f"{'-'*90}")
    for r in sorted(results, key=lambda x: -x["recovery_rate_pct"]):
        dist = r["class_distribution"]

        def pct(cls: str) -> float:
            return dist.get(cls, {}).get("pct_recovered", 0.0)

        print(f"{r['model']:<24} {r['n_recovered']:>11} {r['recovery_rate_pct']:>6.1f}% "
              f"{r['n_abstain']:>10} "
              f"{pct('critico'):>7.1f}% {pct('alto'):>6.1f}% "
              f"{pct('medio'):>6.1f}% {pct('baixo'):>6.1f}%")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
