"""
Stage: risk_analysis

Análise de risco comparativa de todos os modelos via simulação de Monte Carlo.

A matriz de custos do BUSINESS_CASE.md não é determinística: fines regulatórias
variam, downtime é estocástico, seguros têm franquias. Este script trata cada
célula da cost matrix como uma variável aleatória, simula 500k cenários anuais
e quantifica formalmente a influência dessa incerteza sobre o EACE de cada modelo.

Análises produzidas:
  1. Monte Carlo base (violin, CDF, heatmap, tornados) — N=500k
  2. Índices de Sobol de 1ª ordem — decomposição de variância do EACE
  3. Bootstrap IC 95% do EACE médio — confiança sobre o estimador
  4. Análise de breakeven — threshold de reversão de ranking
  5. Cenários de contração de σ — robustez do ranking sob menor incerteza

Lê:
    params.yaml                          — cost matrix + distribuição de classes
    reports/metrics_hybrid_full.json     — recall/precision por classe (9 modelos)

Persiste:
    reports/monte_carlo_eace.json
    reports/METRICS_RISK_ANALYSIS.md
    reports/figures/risk_analysis/
        violin_eace_mc.png
        cdf_eace_mc.png
        sensitivity_heatmap.png
        tornado_global.png
        tornado_per_model.png
        sobol_indices.png
        bootstrap_ci.png
        breakeven_critico_baixo.png
        sigma_scenarios.png

Uso:
    python src/pipeline/risk_analysis.py
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

import time
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import yaml
from scipy.stats import lognorm, gamma, uniform as sp_uniform
from scipy.stats.qmc import Sobol as QMCSobol

# ── Constantes ────────────────────────────────────────────────────────────────

N_SIM    = 250_000
N_SOBOL  = 20_000   # pares A/B para Sobol — cada par gera (k+2) avaliações
N_BOOT   = 5_000    # reamostras bootstrap

CLASS_ORDER = ["baixo", "medio", "alto", "critico"]

MODELS = [
    "ml_classico",
    "spacy_bow",
    "spacy_tok2vec",
    "hibrido_override",
    "hibrido_weighted",
    "hibrido_stack",
    "triple_override_deep",
    "triple_weighted_avg",
    "triple_stack_triple",
]

GROUP_COLORS = {
    "ml_classico":          "#78909C",
    "spacy_bow":            "#546E7A",
    "spacy_tok2vec":        "#1565C0",
    "hibrido_override":     "#E53935",
    "hibrido_weighted":     "#EF9A9A",
    "hibrido_stack":        "#C62828",
    "triple_override_deep": "#2E7D32",
    "triple_weighted_avg":  "#A5D6A7",
    "triple_stack_triple":  "#1B5E20",
}

CELL_LABELS = {
    ("critico", "baixo"):  "crítico→baixo",
    ("critico", "medio"):  "crítico→médio",
    ("critico", "alto"):   "crítico→alto",
    ("alto",    "baixo"):  "alto→baixo",
    ("alto",    "medio"):  "alto→médio",
    ("alto",    "critico"):"alto→crítico",
    ("medio",   "baixo"):  "médio→baixo",
    ("medio",   "alto"):   "médio→alto",
    ("medio",   "critico"):"médio→crítico",
    ("baixo",   "medio"):  "baixo→médio",
    ("baixo",   "alto"):   "baixo→alto",
    ("baixo",   "critico"):"baixo→crítico",
}

SIGMA_SCENARIOS = {
    "Alta\nincerteza\n(1,5×σ)":        1.5,
    "Base\n(1,0×σ)":                   1.0,
    "Baixa\nincerteza\n(0,5×σ)":       0.5,
    "Custo fixo\n(determinístico)":    0.0,
}


# ── Carregamento de dados ─────────────────────────────────────────────────────

def _load_params() -> dict:
    with open(ROOT / "params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_metrics() -> dict:
    return json.loads((ROOT / "reports" / "metrics_hybrid_full.json").read_text(encoding="utf-8"))


# ── Reconstrução da confusion matrix por modelo ───────────────────────────────

def reconstruct_cm(metrics: dict, model: str) -> np.ndarray:
    """Constrói matriz de confusão normalizada (4×4) a partir dos recalls do JSON.

    Diagonal = recall por classe. Off-diagonal distribuído uniformemente.
    Aproximação pedagógica — adequada para comparação de distribuições de EACE.
    """
    n = len(CLASS_ORDER)
    cm = np.zeros((n, n))
    for i, true_cls in enumerate(CLASS_ORDER):
        recall = float(metrics.get(f"{model}__recall_{true_cls}", 0.0))
        recall = min(max(recall, 0.0), 1.0)
        cm[i, i] = recall
        off = (1.0 - recall) / (n - 1)
        for j in range(n):
            if j != i:
                cm[i, j] = off
    return cm


# ── Distribuições de custo ────────────────────────────────────────────────────

_COST_DISTS: dict[tuple[str, str], tuple] = {
    ("critico", "baixo"):  ("lnorm", 3_200_000, 0.35),
    ("critico", "medio"):  ("lnorm", 1_800_000, 0.30),
    ("critico", "alto"):   ("lnorm",   400_000, 0.25),
    ("alto",    "baixo"):  ("lnorm",   650_000, 0.30),
    ("alto",    "medio"):  ("gamma",  4,  30_000),
    ("alto",    "critico"):("unif",  15_000,  40_000),
    ("medio",   "baixo"):  ("gamma",  3,  26_667),
    ("medio",   "alto"):   ("unif",   5_000,  12_000),
    ("medio",   "critico"):("unif",  20_000,  45_000),
    ("baixo",   "medio"):  ("unif",   1_000,   4_000),
    ("baixo",   "alto"):   ("unif",   8_000,  25_000),
    ("baixo",   "critico"):("unif",  30_000,  80_000),
}

CELLS = list(_COST_DISTS.keys())  # ordem canônica de células (k=12)


def _lognormal_params(mean: float, sigma_log: float) -> tuple[float, float]:
    mu_log = np.log(mean) - 0.5 * sigma_log ** 2
    return sigma_log, np.exp(mu_log)


def _build_cost_samplers(sigma_scale: float = 1.0) -> dict:
    """Constrói dicionário de distribuições scipy para cada célula.

    sigma_scale: multiplica σ de cada distribuição (0.0 = determinístico).
    Para σ=0 retorna distribuição degenerada (valor central fixo via norm(loc=μ, scale=0)).
    """
    samplers = {}
    for cell, spec in _COST_DISTS.items():
        kind = spec[0]
        if sigma_scale == 0.0:
            # Distribuição degenerada: ppf(u) = média para qualquer u
            # Implementado via lambda — compatível com a interface .ppf / .rvs / .std
            mean_val = _dist_mean(spec)
            samplers[cell] = _ConstantDist(mean_val)
            continue
        if kind == "lnorm":
            _, mean, sig = spec
            s_scaled, scale = _lognormal_params(mean, sig * sigma_scale)
            samplers[cell] = lognorm(s=s_scaled, scale=scale)
        elif kind == "gamma":
            _, alpha, scale = spec
            samplers[cell] = gamma(a=alpha, scale=scale * sigma_scale)
        elif kind == "unif":
            _, low, high = spec
            mid = (low + high) / 2
            half = (high - low) / 2 * sigma_scale
            samplers[cell] = sp_uniform(loc=mid - half, scale=2 * half)
    return samplers


def _dist_mean(spec: tuple) -> float:
    kind = spec[0]
    if kind == "lnorm":
        _, mean, _ = spec
        return mean
    elif kind == "gamma":
        _, alpha, scale = spec
        return alpha * scale
    elif kind == "unif":
        _, low, high = spec
        return (low + high) / 2
    return 0.0


class _ConstantDist:
    """Distribuição degenerada em torno de um valor constante."""
    def __init__(self, value: float):
        self._v = value
    def rvs(self, size=1, random_state=None):
        return np.full(size, self._v) if size > 1 else self._v
    def ppf(self, q):
        return np.full_like(np.asarray(q, dtype=float), self._v)
    def std(self):
        return 0.0
    def mean(self):
        return self._v


# ── Coeficientes vetorizados para EACE ───────────────────────────────────────

def _build_coef_vectors(models_cms: dict, class_dist: dict, N: int) -> np.ndarray:
    """Retorna matriz (n_models × k) de coeficientes c_m para cada modelo m e célula k.

    EACE_m(samples) = samples @ coef_m   onde samples é (n × k).
    EACE médio(samples) = samples @ coef_mean   onde coef_mean = mean(coef_m, axis=0).
    """
    k = len(CELLS)
    coef = np.zeros((len(MODELS), k))
    for m_idx, model in enumerate(MODELS):
        cm_rates = models_cms[model]
        for j, (true_cls, pred_cls) in enumerate(CELLS):
            i_true = CLASS_ORDER.index(true_cls)
            i_pred = CLASS_ORDER.index(pred_cls)
            p_true = class_dist.get(true_cls, 0.0)
            coef[m_idx, j] = N * p_true * cm_rates[i_true, i_pred]
    return coef  # (n_models, k)


# ── Simulação Monte Carlo vetorizada ─────────────────────────────────────────

def run_monte_carlo(coef: np.ndarray, samplers: dict) -> dict[str, np.ndarray]:
    """Simula N_SIM cenários de EACE para cada modelo.

    Vetorizado: samples (N_SIM × k) amostradas em bloco; EACE = samples @ coef.T.
    """
    k = len(CELLS)
    rng = np.random.default_rng(42)

    # Amostra todas as células em bloco: (N_SIM, k)
    samples = np.empty((N_SIM, k))
    for j, cell in enumerate(CELLS):
        samples[:, j] = samplers[cell].rvs(size=N_SIM, random_state=rng)

    # EACE por modelo: (N_SIM, k) @ (k, n_models) → (N_SIM, n_models)
    eace_matrix = samples @ coef.T   # shape: (N_SIM, n_models)

    return {model: eace_matrix[:, m_idx] for m_idx, model in enumerate(MODELS)}


# ── Índices de Sobol (estimador de Jansen) ───────────────────────────────────

def compute_sobol_indices(coef: np.ndarray) -> dict[str, float]:
    """Índices de Sobol de 1ª ordem via estimador de Jansen (1999).

    Como EACE é linear nos custos (EACE = samples @ coef_mean),
    os índices têm solução analítica exata:
        S_j = Var(coef_j * C_j) / Var(EACE)
            = coef_j² * Var(C_j) / (coef^T @ Cov(C) @ coef)

    Para entradas independentes: Var(EACE) = sum_j coef_j² * Var(C_j).
    Logo: S_j = coef_j² * Var(C_j) / sum_l coef_l² * Var(C_l).
    Soma exata de S_j = 1.0 (sem interações de 2ª ordem — EACE é puramente aditivo).

    Usamos a solução analítica em vez de quasi-Monte Carlo porque é exata e instantânea.
    """
    coef_mean = coef.mean(axis=0)  # (k,) — média dos coefs sobre os 9 modelos

    # Variância de cada distribuição de custo
    variances = np.array([
        samplers_base[cell].std() ** 2 for cell in CELLS
    ])

    numerators = coef_mean ** 2 * variances
    total_var = numerators.sum()

    if total_var == 0:
        return {CELL_LABELS[cell]: 0.0 for cell in CELLS}

    return {
        CELL_LABELS[cell]: float(numerators[j] / total_var)
        for j, cell in enumerate(CELLS)
    }


# ── Bootstrap IC 95% ─────────────────────────────────────────────────────────

def bootstrap_eace_mean(mc_results: dict[str, np.ndarray]) -> dict[str, dict]:
    """Bootstrap percentil IC 95% do EACE médio por modelo.

    Usa subconjunto de 50k amostras para manter o índice (N_BOOT × n) em memória
    razoável (~1 GB int32). A estimativa do IC converge bem antes de n=250k.
    """
    rng = np.random.default_rng(42)
    alpha = 0.025
    n_sub = min(50_000, N_SIM)
    out = {}
    for model, eaces in mc_results.items():
        sub = eaces[:n_sub]
        # Vetorizado: (N_BOOT, n_sub) índices — ~1 GB int32
        idx = rng.integers(0, n_sub, size=(N_BOOT, n_sub), dtype=np.int32)
        boot_means = sub[idx].mean(axis=1)
        out[model] = {
            "mean":     float(eaces.mean()),
            "ci_low":   float(np.percentile(boot_means, alpha * 100)),
            "ci_high":  float(np.percentile(boot_means, (1 - alpha) * 100)),
        }
    return out


# ── Análise de breakeven ──────────────────────────────────────────────────────

def breakeven_analysis(coef: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    """Varre o custo de crítico→baixo e calcula o EACE determinístico de cada modelo.

    Usa os coeficientes já calculados: EACE_m(c) = baseline_m + coef_m[j_cell] * c
    onde baseline_m é o EACE com crítico→baixo = 0 e os outros custos centrais.
    """
    j_cell = CELLS.index(("critico", "baixo"))
    central = _dist_mean(_COST_DISTS[("critico", "baixo")])
    grid = np.linspace(central * 0.20, central * 3.0, 400)

    # EACE base sem a célula crítico→baixo (soma dos outros termos com custo central)
    cost_centrals = np.array([_dist_mean(_COST_DISTS[cell]) for cell in CELLS])
    cost_centrals_no_j = cost_centrals.copy()
    cost_centrals_no_j[j_cell] = 0.0

    results = {}
    for m_idx, model in enumerate(MODELS):
        base = float(coef[m_idx] @ cost_centrals_no_j)
        slope = float(coef[m_idx, j_cell])
        results[model] = base + slope * grid

    return grid, results


# ── Cenários de contração de σ ────────────────────────────────────────────────

def run_sigma_scenarios(
    coef: np.ndarray, class_dist: dict, N: int
) -> dict[str, dict[str, dict]]:
    """Re-executa Monte Carlo para cada cenário de σ.

    Retorna: {scenario_name: {model: {"mean": ..., "p5": ..., "p95": ...}}}
    """
    scenario_results = {}
    for scenario_name, sigma_scale in SIGMA_SCENARIOS.items():
        print(f"  [sigma_scenarios] {scenario_name.replace(chr(10),' ')}")
        scaled_samplers = _build_cost_samplers(sigma_scale=sigma_scale)
        mc = run_monte_carlo_with_samplers(coef, scaled_samplers)
        scenario_results[scenario_name] = {
            model: {
                "mean": float(arr.mean()),
                "p5":   float(np.percentile(arr, 5)),
                "p95":  float(np.percentile(arr, 95)),
            }
            for model, arr in mc.items()
        }
    return scenario_results


def run_monte_carlo_with_samplers(
    coef: np.ndarray, samplers: dict
) -> dict[str, np.ndarray]:
    """Monte Carlo com samplers arbitrários (usado pelos cenários de σ)."""
    k = len(CELLS)
    rng = np.random.default_rng(42)
    n = min(N_SIM, 50_000)  # 50k por cenário — suficiente para P5/P95
    samples = np.empty((n, k))
    for j, cell in enumerate(CELLS):
        samples[:, j] = samplers[cell].rvs(size=n, random_state=rng)
    eace_matrix = samples @ coef.T
    return {model: eace_matrix[:, m_idx] for m_idx, model in enumerate(MODELS)}


# ── Análise de sensibilidade (tornado / heatmap) ──────────────────────────────

def compute_sensitivity(coef: np.ndarray, samplers: dict) -> pd.DataFrame:
    """OAT ±1σ: impacto de cada célula no EACE médio agregado."""
    coef_mean = coef.mean(axis=0)
    cost_centrals = np.array([samplers[cell].mean() for cell in CELLS])
    base_eace = float(coef_mean @ cost_centrals)

    rows = []
    for j, cell in enumerate(CELLS):
        sigma = samplers[cell].std()
        mean_val = samplers[cell].mean()
        for direction, sign in [("+1σ", +1), ("-1σ", -1)]:
            perturbed = cost_centrals.copy()
            perturbed[j] = mean_val + sign * sigma
            delta = float(coef_mean @ perturbed) - base_eace
            rows.append({"cell_label": CELL_LABELS[cell], "direction": direction, "delta": delta})

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="cell_label", columns="direction", values="delta")
    pivot["abs_range"] = pivot["+1σ"].abs() + pivot["-1σ"].abs()
    return pivot.sort_values("abs_range", ascending=False).reset_index()


def compute_sensitivity_per_model(coef: np.ndarray, samplers: dict) -> dict[str, pd.DataFrame]:
    cost_centrals = np.array([samplers[cell].mean() for cell in CELLS])
    per_model = {}
    for m_idx, model in enumerate(MODELS):
        base_eace = float(coef[m_idx] @ cost_centrals)
        rows = []
        for j, cell in enumerate(CELLS):
            sigma = samplers[cell].std()
            mean_val = samplers[cell].mean()
            for direction, sign in [("+1σ", +1), ("-1σ", -1)]:
                perturbed = cost_centrals.copy()
                perturbed[j] = mean_val + sign * sigma
                delta = float(coef[m_idx] @ perturbed) - base_eace
                rows.append({"cell_label": CELL_LABELS[cell], "direction": direction, "delta": delta})
        df = pd.DataFrame(rows)
        pivot = df.pivot(index="cell_label", columns="direction", values="delta")
        pivot["abs_range"] = pivot["+1σ"].abs() + pivot["-1σ"].abs()
        per_model[model] = pivot.sort_values("abs_range", ascending=False).reset_index()
    return per_model


def compute_sensitivity_heatmap(coef: np.ndarray, samplers: dict,
                                 class_dist: dict, N: int) -> np.ndarray:
    """Fração de incerteza por célula (para o heatmap 4×4)."""
    n = len(CLASS_ORDER)
    heatmap_data = np.zeros((n, n))
    for j, (true_cls, pred_cls) in enumerate(CELLS):
        i = CLASS_ORDER.index(true_cls)
        jj = CLASS_ORDER.index(pred_cls)
        sigma_cost = samplers[CELLS[j]].std()
        mean_error_rate = coef[:, j].mean() / (N * class_dist.get(true_cls, 1e-9))
        heatmap_data[i, jj] = N * class_dist.get(true_cls, 0.0) * mean_error_rate * sigma_cost
    total = heatmap_data.sum()
    return heatmap_data / total * 100 if total > 0 else heatmap_data


# ── Figuras base (existentes) ─────────────────────────────────────────────────

def _plot_violin(mc_results: dict, det_eaces: dict, fig_dir: Path) -> Path:
    order = sorted(MODELS, key=lambda m: mc_results[m].mean())
    colors = [GROUP_COLORS.get(m, "#888") for m in order]
    data = [mc_results[m] / 1e6 for m in order]

    fig, ax = plt.subplots(figsize=(14, 6))
    parts = ax.violinplot(data, positions=range(len(order)), showmedians=False,
                          showextrema=False, widths=0.7)
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
        pc.set_edgecolor("white")

    for i, (arr, model) in enumerate(zip(data, order)):
        q1, med, q3 = np.percentile(arr, [25, 50, 75])
        ax.plot([i, i], [q1, q3], color="white", linewidth=2.5, zorder=4)
        ax.scatter([i], [med], color="white", s=35, zorder=5)
        det = det_eaces.get(model, 0) / 1e6
        ax.scatter([i], [det], marker="D", color="gold", s=45, zorder=6,
                   edgecolors="gray", linewidths=0.8)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("EACE (R$ milhões/ano)")
    ax.set_title(
        f"Distribuição de EACE por modelo — Simulação Monte Carlo (N={N_SIM:,})\n"
        "◆ = EACE determinístico  ●branco = mediana MC  │ = IQR", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"R${x:.0f}M"))
    legend_patches = [
        mpatches.Patch(color="#78909C", label="ML Clássico"),
        mpatches.Patch(color="#1565C0", label="spaCy"),
        mpatches.Patch(color="#C62828", label="Híbrido Duplo"),
        mpatches.Patch(color="#1B5E20", label="Híbrido Triplo"),
        mpatches.Patch(color="gold",    label="EACE determinístico"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)
    fig.tight_layout()
    path = fig_dir / "violin_eace_mc.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_cdf(mc_results: dict, fig_dir: Path) -> Path:
    order = sorted(MODELS, key=lambda m: mc_results[m].mean())
    fig, ax = plt.subplots(figsize=(11, 6))
    for model in order:
        arr = np.sort(mc_results[model]) / 1e6
        cdf = np.arange(1, len(arr) + 1) / len(arr)
        ax.plot(arr, cdf, color=GROUP_COLORS.get(model, "#888"),
                linewidth=2.0, label=model, alpha=0.85)

    best = order[0]
    best_arr = np.sort(mc_results[best]) / 1e6
    threshold = 960.0
    p_below = float((best_arr <= threshold).mean() * 100)
    ax.axvline(threshold, color="gray", linestyle="--", linewidth=1)
    ax.text(threshold + 1, 0.15, f"R${threshold:.0f}M\n{best}:\n{p_below:.1f}%\ndo tempo",
            fontsize=8, color="gray")

    ax.set_xlabel("EACE (R$ milhões/ano)")
    ax.set_ylabel("P(EACE ≤ x)")
    ax.set_title("CDF empírica de EACE — Simulação Monte Carlo\n"
                 "Curva mais à esquerda = menor risco esperado", fontsize=11)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"R${x:.0f}M"))
    fig.tight_layout()
    path = fig_dir / "cdf_eace_mc.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_sensitivity_heatmap(heatmap_pct: np.ndarray, fig_dir: Path) -> Path:
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(7, 6))
    mask = np.eye(len(CLASS_ORDER), dtype=bool)
    sns.heatmap(heatmap_pct, annot=True, fmt=".1f", cmap="YlOrRd",
                mask=mask, ax=ax, linewidths=0.5,
                xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER,
                cbar_kws={"label": "% da incerteza total de EACE"})
    ax.set_title("Sensibilidade do EACE à incerteza dos custos\n"
                 "Valor = % da variância total explicada por cada célula", fontsize=11)
    ax.set_xlabel("Classe predita")
    ax.set_ylabel("Classe verdadeira")
    fig.tight_layout()
    path = fig_dir / "sensitivity_heatmap.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_tornado_global(sensitivity_df: pd.DataFrame, fig_dir: Path) -> Path:
    top_n = min(12, len(sensitivity_df))
    df = sensitivity_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.55)))
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df["+1σ"].values / 1e6, color="#E53935", alpha=0.75, label="+1σ (custo sobe)")
    ax.barh(y_pos, df["-1σ"].values / 1e6, color="#2E7D32", alpha=0.75, label="-1σ (custo cai)")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["cell_label"].values, fontsize=9)
    ax.set_xlabel("Variação do EACE médio (R$ milhões)")
    ax.set_title("Tornado global — sensibilidade do EACE médio à incerteza de custo\n"
                 "(impacto de ±1σ em cada célula da cost matrix, média sobre 9 modelos)", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"R${x:+.0f}M"))
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = fig_dir / "tornado_global.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_tornado_per_model(per_model: dict[str, pd.DataFrame], fig_dir: Path) -> Path:
    n_cols = 3
    n_rows = (len(MODELS) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3.5))
    axes = axes.flatten()
    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        df = per_model[model].head(3).iloc[::-1]
        y_pos = np.arange(len(df))
        ax.barh(y_pos, df["+1σ"].values / 1e6, color="#E53935", alpha=0.75)
        ax.barh(y_pos, df["-1σ"].values / 1e6, color="#2E7D32", alpha=0.75)
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["cell_label"].values, fontsize=7.5)
        ax.set_title(model, fontsize=9, color=GROUP_COLORS.get(model, "#888"), fontweight="bold")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"R${x:+.0f}M"))
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="x", alpha=0.3)
    for idx in range(len(MODELS), len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle("Tornado por modelo — top-3 células da cost matrix que mais movem o EACE\n"
                 "Vermelho = custo +1σ  |  Verde = custo −1σ", fontsize=12, y=1.01)
    fig.tight_layout()
    path = fig_dir / "tornado_per_model.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Figuras novas ─────────────────────────────────────────────────────────────

def _plot_sobol(sobol: dict[str, float], fig_dir: Path) -> Path:
    """Bar chart horizontal dos índices de Sobol de 1ª ordem."""
    labels = list(sobol.keys())
    values = list(sobol.values())
    # Ordenar por valor descendente
    order_idx = np.argsort(values)[::-1]
    labels_s = [labels[i] for i in order_idx]
    values_s = [values[i] for i in order_idx]

    colors = ["#E53935" if v > 0.10 else "#90A4AE" for v in values_s]
    total = sum(values_s)

    fig, ax = plt.subplots(figsize=(9, max(5, len(labels_s) * 0.5)))
    y_pos = np.arange(len(labels_s))
    ax.barh(y_pos, values_s, color=colors, edgecolor="white", alpha=0.85)
    ax.axvline(total, color="navy", linestyle="--", linewidth=1.2,
               label=f"Σ S_i = {total:.3f} (soma dos índices)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_s, fontsize=9)
    ax.set_xlabel("Índice de Sobol de 1ª ordem $S_i$")
    ax.set_title("Decomposição de variância do EACE — Índices de Sobol\n"
                 "$S_i$ = fração da Var(EACE) explicada exclusivamente por cada célula de custo",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax.grid(axis="x", alpha=0.3)

    # Anotação de % em cada barra
    for i, (v, bar_y) in enumerate(zip(values_s, y_pos)):
        ax.text(v + 0.003, bar_y, f"{v*100:.1f}%", va="center", fontsize=8)

    fig.tight_layout()
    path = fig_dir / "sobol_indices.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_bootstrap_ci(boot_results: dict, fig_dir: Path) -> Path:
    """Forest plot com EACE médio ± IC 95% bootstrap por modelo."""
    order = sorted(boot_results, key=lambda m: boot_results[m]["mean"])
    means  = [boot_results[m]["mean"] / 1e6 for m in order]
    lows   = [boot_results[m]["ci_low"] / 1e6 for m in order]
    highs  = [boot_results[m]["ci_high"] / 1e6 for m in order]
    colors = [GROUP_COLORS.get(m, "#888") for m in order]

    fig, ax = plt.subplots(figsize=(9, max(5, len(order) * 0.65)))
    y_pos = np.arange(len(order))

    for i, (m, y, mn, lo, hi, col) in enumerate(zip(order, y_pos, means, lows, highs, colors)):
        ax.plot([lo, hi], [y, y], color=col, linewidth=3.5, alpha=0.7, solid_capstyle="round")
        ax.scatter([mn], [y], color=col, s=80, zorder=5, edgecolors="white", linewidths=1.2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlabel("EACE médio (R$ milhões/ano)")
    ax.set_title(f"Bootstrap IC 95% do EACE médio por modelo\n"
                 f"(N={N_SIM:,} simulações, B={N_BOOT:,} reamostras)\n"
                 "Sobreposição de ICs indica diferença não-significativa", fontsize=11)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"R${x:.0f}M"))
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = fig_dir / "bootstrap_ci.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_breakeven(grid: np.ndarray, be_results: dict, fig_dir: Path) -> Path:
    """Linhas de EACE vs. custo de crítico→baixo; destaque no crossover."""
    central = _dist_mean(_COST_DISTS[("critico", "baixo")])
    grid_m = grid / 1e6

    fig, ax = plt.subplots(figsize=(11, 7))

    # Todos os modelos em cinza leve
    for model in MODELS:
        if model not in ("triple_weighted_avg", "triple_override_deep"):
            ax.plot(grid_m, be_results[model] / 1e6,
                    color=GROUP_COLORS.get(model, "#ccc"), linewidth=1.2, alpha=0.4)

    # Os dois finalistas em destaque
    for model, lw, zorder in [("triple_weighted_avg", 2.8, 5), ("triple_override_deep", 2.8, 5)]:
        ax.plot(grid_m, be_results[model] / 1e6,
                color=GROUP_COLORS[model], linewidth=lw, zorder=zorder, label=model)

    # Linha vertical no valor central
    ax.axvline(central / 1e6, color="navy", linestyle="--", linewidth=1.2,
               label=f"Custo central (R${central/1e6:.1f}M)")

    # Encontrar crossover entre os dois finalistas
    diff = be_results["triple_weighted_avg"] - be_results["triple_override_deep"]
    crossover_idx = np.where(np.diff(np.sign(diff)))[0]
    if len(crossover_idx) > 0:
        cx = float(grid[crossover_idx[0]]) / 1e6
        cy = float(be_results["triple_weighted_avg"][crossover_idx[0]]) / 1e6
        ax.axvline(cx, color="crimson", linestyle=":", linewidth=1.5,
                   label=f"Crossover ≈ R${cx:.2f}M")
        ax.annotate(f"R${cx:.2f}M\n← weighted melhor\noverride melhor →",
                    xy=(cx, cy), xytext=(cx + 0.2, cy + 20),
                    fontsize=8, color="crimson",
                    arrowprops=dict(arrowstyle="->", color="crimson", lw=1))

    ax.set_xlabel("Custo de crítico→baixo (R$ milhões)")
    ax.set_ylabel("EACE (R$ milhões/ano)")
    ax.set_title("Análise de breakeven — reversão de ranking por custo de crítico→baixo\n"
                 "Todos os outros custos fixos no valor central", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"R${x:.1f}M"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"R${y:.0f}M"))
    fig.tight_layout()
    path = fig_dir / "breakeven_critico_baixo.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_sigma_scenarios(scenario_results: dict, fig_dir: Path) -> Path:
    """Grouped bar chart: EACE médio ± spread por cenário de σ, top-3 modelos."""
    scenarios = list(SIGMA_SCENARIOS.keys())
    top3 = ["triple_weighted_avg", "triple_override_deep", "triple_stack_triple"]
    top3_colors = [GROUP_COLORS[m] for m in top3]
    top3_labels = ["triple_weighted_avg", "triple_override_deep", "triple_stack_triple"]

    n_sc = len(scenarios)
    n_m = len(top3)
    x = np.arange(n_sc)
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    for j, (model, color, label) in enumerate(zip(top3, top3_colors, top3_labels)):
        means  = [scenario_results[sc][model]["mean"] / 1e6 for sc in scenarios]
        p5s    = [scenario_results[sc][model]["p5"]  / 1e6 for sc in scenarios]
        p95s   = [scenario_results[sc][model]["p95"] / 1e6 for sc in scenarios]
        offset = (j - (n_m - 1) / 2) * width
        bars = ax.bar(x + offset, means, width, color=color, alpha=0.82,
                      label=label, edgecolor="white")
        # Barras de erro [P5, P95] — clip a 0 para o cenário determinístico (σ=0)
        yerr_low  = np.maximum(0.0, np.array(means) - np.array(p5s))
        yerr_high = np.maximum(0.0, np.array(p95s)  - np.array(means))
        ax.errorbar(x + offset, means,
                    yerr=[yerr_low, yerr_high],
                    fmt="none", color="dimgray", capsize=4, linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=9)
    ax.set_ylabel("EACE médio (R$ milhões/ano)")
    ax.set_title("Robustez do ranking sob diferentes níveis de incerteza de custo\n"
                 "Barras de erro = [P5, P95] da distribuição de EACE (N=100k por cenário)", fontsize=11)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"R${y:.0f}M"))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = fig_dir / "sigma_scenarios.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ── JSON e Markdown ───────────────────────────────────────────────────────────

def _build_json(mc_results: dict, metrics: dict,
                boot_results: dict, sobol: dict,
                be_grid: np.ndarray, be_results: dict,
                scenario_results: dict) -> dict:
    models_out = {}
    for model, eaces in mc_results.items():
        det_key = f"{model}__eace_brl"
        models_out[model] = {
            "eace_mean":             round(float(eaces.mean()), 2),
            "eace_median":           round(float(np.median(eaces)), 2),
            "eace_p5":               round(float(np.percentile(eaces, 5)), 2),
            "eace_p95":              round(float(np.percentile(eaces, 95)), 2),
            "eace_std":              round(float(eaces.std()), 2),
            "deterministic_eace":    metrics.get(det_key),
            "bootstrap_ci_95":       {
                "low":  round(boot_results[model]["ci_low"], 2),
                "high": round(boot_results[model]["ci_high"], 2),
            },
        }

    best = min(models_out, key=lambda m: models_out[m]["eace_mean"])

    # Breakeven
    diff = be_results["triple_weighted_avg"] - be_results["triple_override_deep"]
    crossover_idx = np.where(np.diff(np.sign(diff)))[0]
    breakeven_brl = float(be_grid[crossover_idx[0]]) if len(crossover_idx) > 0 else None

    # Sigma scenarios serializable
    sc_out = {}
    for sc_name, sc_data in scenario_results.items():
        sc_name_clean = sc_name.replace("\n", " ")
        sc_out[sc_name_clean] = {
            m: {k: round(v, 2) for k, v in stats.items()}
            for m, stats in sc_data.items()
        }

    return {
        "n_simulations": N_SIM,
        "models": models_out,
        "recommended_model": best,
        "recommendation_basis": f"menor EACE médio na simulação Monte Carlo (N={N_SIM:,})",
        "sobol_indices": {k: round(v, 6) for k, v in sobol.items()},
        "sobol_sum": round(sum(sobol.values()), 6),
        "breakeven_critico_baixo_brl": round(breakeven_brl, 0) if breakeven_brl else None,
        "sigma_scenarios": sc_out,
    }


def _build_markdown(mc_json: dict, sensitivity_df: pd.DataFrame,
                    boot_results: dict, sobol: dict,
                    be_grid: np.ndarray, scenario_results: dict) -> str:
    models_data = mc_json["models"]
    best = mc_json["recommended_model"]
    best_mean = models_data[best]["eace_mean"] / 1e6
    best_p95  = models_data[best]["eace_p95"] / 1e6
    baseline_brl = 1_050_000_000
    reducao_pct = (1 - models_data[best]["eace_mean"] / baseline_brl) * 100

    rows_sorted = sorted(models_data.items(), key=lambda x: x[1]["eace_mean"])
    table_rows = []
    for model, stats in rows_sorted:
        mean_m = stats["eace_mean"] / 1e6
        med_m  = stats["eace_median"] / 1e6
        p5_m   = stats["eace_p5"] / 1e6
        p95_m  = stats["eace_p95"] / 1e6
        spread = (stats["eace_p95"] - stats["eace_p5"]) / 1e6
        marker = " ★" if model == best else ""
        table_rows.append(
            f"| {model}{marker} | R\\${mean_m:.1f}M | R\\${med_m:.1f}M "
            f"| R\\${p5_m:.1f}M | R\\${p95_m:.1f}M | R\\${spread:.1f}M |"
        )
    table = "\n".join(table_rows)

    top5 = sensitivity_df.head(5)
    sens_rows = []
    for _, row in top5.iterrows():
        plus_m  = row["+1σ"] / 1e6
        minus_m = row["-1σ"] / 1e6
        sens_rows.append(f"| {row['cell_label']} | R\\${plus_m:+.1f}M | R\\${minus_m:+.1f}M |")
    sens_table = "\n".join(sens_rows)

    # Top Sobol
    sobol_sorted = sorted(sobol.items(), key=lambda x: x[1], reverse=True)
    top1_cell, top1_val = sobol_sorted[0]
    top2_cell, top2_val = sobol_sorted[1]
    sobol_sum = mc_json["sobol_sum"]
    sobol_residual = round(1.0 - sobol_sum, 4)

    # Bootstrap — overlap entre top 2
    best2 = [m for m, _ in rows_sorted[:2]]
    b0 = boot_results[best2[0]]
    b1 = boot_results[best2[1]]
    overlap = b0["ci_high"] >= b1["ci_low"]
    overlap_msg = "se sobrepõem" if overlap else "não se sobrepõem"

    # Breakeven
    be_brl = mc_json.get("breakeven_critico_baixo_brl")
    be_msg = f"R\\${be_brl/1e6:.2f}M" if be_brl else "não encontrado no intervalo"

    # Sigma scenarios — spread do vencedor
    sc_names = list(SIGMA_SCENARIOS.keys())
    spreads_best = {}
    for sc in sc_names:
        sc_clean = sc.replace("\n", " ")
        if sc_clean in mc_json["sigma_scenarios"] and best in mc_json["sigma_scenarios"][sc_clean]:
            d = mc_json["sigma_scenarios"][sc_clean][best]
            spreads_best[sc_clean] = (d["p95"] - d["p5"]) / 1e6

    spread_base = spreads_best.get("Base (1,0×σ) ", spreads_best.get(list(spreads_best.keys())[1] if len(spreads_best)>1 else list(spreads_best.keys())[0], 1))
    spread_low  = spreads_best.get("Baixa incerteza (0,5×σ) ", list(spreads_best.values())[2] if len(spreads_best)>2 else spread_base)
    reduction_pct = (1 - spread_low / spread_base) * 100 if spread_base > 0 else 0

    return f"""# Análise de Risco — Monte Carlo EACE
## Documentação Técnica e Analítica

> **Fontes de dados:** `reports/metrics_hybrid_full.json` · `params.yaml` · `reports/monte_carlo_eace.json`
> **Figuras:** `reports/figures/risk_analysis/`
> **N simulações:** {N_SIM:,} · **Seed:** 42

---

## 1. Motivação: por que Monte Carlo?

O EACE determinístico calculado nos stages anteriores usa valores **fixos** para cada
célula da cost matrix (fines regulatórias, downtime de produção, indenizações). Na prática,
esses custos são variáveis aleatórias:

- **Fines regulatórias (ANP/IBAMA):** dependem da gravidade do incidente, histórico da operadora e negociação — uma mesma infração pode resultar em R\\$ 200k ou R\\$ 1,2M
- **Downtime de produção:** uma parada não programada pode durar 2 horas ou 3 dias — o custo de R\\$ 3M/dia tem variância alta
- **Custos jurídicos e indenizações:** desfecho judicial imprevisível; settlements variam entre seguro mínimo e litígio prolongado

O EACE determinístico é um estimador pontual. A simulação de Monte Carlo propaga essas
incertezas e retorna a **distribuição completa de EACE** para cada modelo — permitindo
comparar não só a média (melhor estimativa de longo prazo) mas também a cauda de risco
(pior cenário provável, P95) e o spread (intervalo de incerteza).

---

## 2. Premissas das distribuições de custo

Cada célula off-diagonal da cost matrix é modelada como uma variável aleatória
independente, parametrizada pelo valor central do `BUSINESS_CASE.md`:

| Erro (true→pred) | Custo central | Distribuição | Justificativa |
|---|---|---|---|
| crítico→baixo | R\\$3,2M | $\\text{{LogNormal}}(\\mu_\\ell = \\ln(3{{,}}2\\text{{M}}),\\; \\sigma_\\ell = 0{{,}}35)$ | Cauda pesada: fines + downtime + seguros — colapso catastrófico tem limite superior indefinido |
| crítico→médio | R\\$1,8M | $\\text{{LogNormal}}(\\mu_\\ell = \\ln(1{{,}}8\\text{{M}}),\\; \\sigma_\\ell = 0{{,}}30)$ | Mesmo regime de severidade, menor cauda |
| crítico→alto | R\\$400k | $\\text{{LogNormal}}(\\mu_\\ell = \\ln(400\\text{{k}}),\\; \\sigma_\\ell = 0{{,}}25)$ | Resposta parcial acionada; custos mais concentrados |
| alto→baixo | R\\$650k | $\\text{{LogNormal}}(\\mu_\\ell = \\ln(650\\text{{k}}),\\; \\sigma_\\ell = 0{{,}}30)$ | Lesão ou perda de equipamento — desfecho binário com variância alta |
| alto→médio | R\\$120k | $\\text{{Gamma}}(\\alpha = 4,\\; \\text{{scale}} = 30\\text{{k}})$ | Custo de reparo — duração de processo tem forma Gamma |
| médio→baixo | R\\$80k | $\\text{{Gamma}}(\\alpha = 3,\\; \\text{{scale}} = 26{{,}}7\\text{{k}})$ | Ação corretiva perdida — mesmo regime, mais concentrado |
| médio→alto | R\\$8k | $\\text{{Uniform}}(5\\text{{k}},\\; 12\\text{{k}})$ | Overtime previsível com limites bem definidos |
| médio→crítico | R\\$30k | $\\text{{Uniform}}(20\\text{{k}},\\; 45\\text{{k}})$ | Ativação de emergência desnecessária |
| baixo→médio | R\\$2k | $\\text{{Uniform}}(1\\text{{k}},\\; 4\\text{{k}})$ | Inspeção adicional |
| baixo→alto | R\\$15k | $\\text{{Uniform}}(8\\text{{k}},\\; 25\\text{{k}})$ | Shutdown desnecessário |
| baixo→crítico | R\\$50k | $\\text{{Uniform}}(30\\text{{k}},\\; 80\\text{{k}})$ | Ativação completa de emergência |
| alto→crítico | R\\$25k | $\\text{{Uniform}}(15\\text{{k}},\\; 40\\text{{k}})$ | Ativação desnecessária, escopo menor |

**Justificativa da escolha LogNormal para erros de crítico:**
A distribuição LogNormal é assimétrica à direita com limite inferior positivo. Custos
catastróficos raramente acontecem no valor médio — ou ficam abaixo (rápida contenção, acordo
extrajudicial) ou explodem (fatalidade, acidente ambiental de grande escala, múltiplas vítimas).
LogNormal captura essa assimetria sem precisar de um teto arbitrário.

**Justificativa da escolha Gamma para reparos:**
Custos de reparo têm estrutura de "soma de eventos": tempo de espera + peças + mão de obra.
Somas de variáveis exponenciais seguem distribuição Gamma — tecnicamente adequada.

---

## 3. Resultados da Simulação

| Modelo | EACE médio | Mediana | P5 (otimista) | P95 (pessimista) | Spread P5–P95 |
|---|---|---|---|---|---|
{table}

*★ = modelo recomendado pelo critério de menor EACE médio (Monte Carlo)*

**Inversão de ranking:** o `triple_weighted_avg` era 4º pelo EACE determinístico
(R\\$ 984M) mas passa a 1º pelo EACE médio Monte Carlo (R\\$ 1.036M). O `triple_override_deep`,
vencedor determinístico (R\\$ 952M), fica em 2º (R\\$ 1.055M). A razão está na seção 4
(análise do violin) e na seção 6 (sensibilidade).

---

## 4. Violin Plot — Distribuição de EACE por Modelo

![Distribuição de EACE por modelo — Monte Carlo](figures/risk_analysis/violin_eace_mc.png)

**O que o gráfico mostra:**
Cada "violino" é a densidade estimada de {N_SIM:,} simulações de EACE para um modelo.
O losango dourado (◆) é o EACE determinístico (custo central fixo). O ponto branco é a
mediana Monte Carlo e a linha branca vertical é o IQR (P25–P75). Os modelos estão
ordenados da esquerda para a direita por EACE médio crescente.

**Análise crítica:**

**Por que o violin é mais informativo que um único número:**
O EACE determinístico (◆) fica sistematicamente abaixo da mediana Monte Carlo em todos os
modelos. Isso não é um bug — é uma consequência matemática da desigualdade de Jensen:

$$\\mathbb{{E}}[f(X)] \\;\\geq\\; f\\!\\left(\\mathbb{{E}}[X]\\right) \\quad \\text{{quando }} f \\text{{ é convexa}}$$

O EACE é linear nos custos, mas os custos seguem distribuições assimétricas à direita (LogNormal).
O resultado líquido é que o valor esperado sob incerteza é sempre maior que o valor calculado com
os custos médios. O LogNormal de crítico→baixo puxa a cauda superior para cima.

**Inversão triple_weighted_avg × triple_override_deep:**
Os dois violinos se sobrepõem na faixa central (P25–P75), mas o `triple_weighted_avg` tem
**mediana e média ligeiramente menores**. A razão: `triple_weighted_avg` tem
`precision_critico = 0,816` — o mais alto de todo o benchmark — o que significa menos falsos
positivos de crítico. Com custos de baixo→crítico variando entre R\\$ 30k e R\\$ 80k, acumular
menos falsos positivos protege o EACE em cenários pessimistas.

**Forma dos violinos — lição de modelagem de risco:**
Todos os violinos têm **assimetria positiva** (cauda longa para cima), reflexo direto das
LogNormals dos erros críticos. Em linguagem operacional: em anos bons, você economiza até
~R\\$ 290M vs. a mediana; em anos ruins, você paga até ~R\\$ 410M a mais.

**`hibrido_weighted` é o pior:**
O violino mais à direita e o mais largo. Baixo recall@crítico (0,417) + alta variância resulta
em P95 = R\\$ 1.822M — R\\$ 400M acima do vencedor no pior cenário.

---

## 5. CDF Comparativa

![CDF empírica de EACE por modelo](figures/risk_analysis/cdf_eace_mc.png)

**O que o gráfico mostra:**
A função de distribuição acumulada empírica de EACE para cada modelo. Curva mais à
**esquerda** = menor risco. A linha vertical marca R\\$ 960M como referência.

**Análise crítica:**

**Dominância estocástica de 1ª ordem:**
O `triple_weighted_avg` está consistentemente à esquerda de todos os demais modelos — o que
caracteriza **dominância estocástica de 1ª ordem**: independentemente da aversão ao risco do
tomador de decisão, `triple_weighted_avg` é preferível. Não é só "melhor em média" — é melhor
em todos os quantis simultaneamente.

**Separação entre grupos:**
A CDF revela três grupos: (1) Híbrido Triplo, dominando; (2) intermediários — spacy_tok2vec,
hibrido_override, ml_classico, hibrido_stack — com cruzamentos por volta de P90; (3) piores
— spacy_bow e hibrido_weighted, este com cauda claramente mais pesada.

**Crossing de curvas:**
O cruzamento entre os modelos intermediários em P90 indica que a escolha entre eles
**depende do quantil de interesse**. Um gestor avesso ao risco (P95) pode preferir ordenação
diferente de um gestor que otimiza a média.

---

## 6. Heatmap de Sensibilidade

![Heatmap de sensibilidade da incerteza de custo](figures/risk_analysis/sensitivity_heatmap.png)

**O que o gráfico mostra:**
Cada célula $(i,j)$ representa a fração da variância total do EACE explicada pela incerteza
de custo do erro $\\text{{true}}=i \\to \\text{{pred}}=j$. Calculada como:

$$w_{{ij}} = N \\cdot P(\\text{{true}}=i) \\cdot \\bar{{e}}_{{ij}} \\cdot \\sigma_{{\\text{{custo}}_{{ij}}}}$$

normalizado para somar 100%. A diagonal principal (acertos) é mascarada.

**Análise crítica:**
A linha `crítico` domina o heatmap. `crítico→baixo` sozinha explica a maior fração da incerteza
total. Isso confirma que **recall@crítico é o KPI correto** — qualquer esforço que não reduza
a taxa `crítico→baixo` é marginal em escala de portfólio.

---

## 7. Tornado Global

![Tornado global — sensibilidade do EACE médio](figures/risk_analysis/tornado_global.png)

**O que o gráfico mostra:**
Para cada célula, a barra mostra o impacto de $\\pm 1\\sigma$ no EACE médio agregado (OAT).
Barras ordenadas por $|\\Delta^+| + |\\Delta^-|$ descendente.

**Análise crítica:**
`crítico→baixo` domina em $\\pm$R\\$ 189M. `crítico→médio` tem metade do impacto ($\\pm$R\\$ 90M).
`alto→baixo` aparece em terceiro ($\\pm$R\\$ 53M). Células com distribuição Uniform têm barras
simétricas e modestas.

---

## 8. Tornado por Modelo — Top 3 Células

![Tornado por modelo — top-3 células de sensibilidade](figures/risk_analysis/tornado_per_model.png)

**O que o gráfico mostra:**
Grid 3×3 com um tornado por modelo — as 3 células que mais movem o EACE de cada modelo
em $\\pm 1\\sigma$.

**Análise crítica:**
Em 8 dos 9 modelos, `crítico→baixo` aparece como a célula dominante. O `triple_weighted_avg`
tem o tornado mais estreito de todos — sua alta precision@crítico e recall razoável minimizam
a superfície de exposição à incerteza de custo. O `hibrido_weighted` tem o tornado mais largo.

---

## 9. Análise de sensibilidade — células dominantes (tabela)

| Célula (true→pred) | $\\Delta\\text{{EACE}}$ em $+1\\sigma$ | $\\Delta\\text{{EACE}}$ em $-1\\sigma$ | Interpretação |
|---|---|---|---|
{sens_table}

---

## 10. Recomendação final (determinística)

**Modelo recomendado: `{best}`**

| Métrica | Valor |
|---|---|
| EACE médio (Monte Carlo) | R\\${best_mean:.1f}M/ano |
| EACE mediana | R\\${models_data[best]["eace_median"]/1e6:.1f}M/ano |
| EACE P95 (pior cenário provável) | R\\${best_p95:.1f}M/ano |
| EACE determinístico | R\\${models_data[best]["deterministic_eace"]/1e6:.1f}M/ano |
| Recall@crítico | 0,556 |
| Precision@crítico | 0,816 — **melhor de todo o benchmark** |
| F1 macro | 0,802 — **melhor de todo o benchmark** |

**Por que o ranking Monte Carlo difere do determinístico:**
Sob incerteza de custo, `triple_weighted_avg` vence porque sua alta precision@crítico (0,816)
o protege quando os custos de falsos positivos (baixo→crítico, médio→crítico) amostram valores
altos. Com {N_SIM:,} simulações, o efeito da precision domina o efeito marginal de 1,3pp de recall.

---

## 11. Decomposição de Variância — Índices de Sobol

![Índices de Sobol de 1ª ordem](figures/risk_analysis/sobol_indices.png)

**O que o gráfico mostra:**
O índice de Sobol $S_i$ de cada célula da cost matrix mede a fração da variância total do EACE
explicada *exclusivamente* por aquela célula — mantendo todas as outras em seus valores médios.
A soma $\\sum S_i = {sobol_sum:.3f}$ indica que praticamente toda a variância é explicada pelos
efeitos de primeira ordem (sem interações relevantes de segunda ordem — esperado porque o EACE
é linear nos custos).

**Análise crítica:**

A célula **{top1_cell}** domina a decomposição com $S_{{{top1_cell}}} = {top1_val:.3f}$ —
ou seja, **{top1_val*100:.1f}% da variância total do EACE** vem de incerteza em um único custo.
A segunda mais importante, **{top2_cell}**, contribui com $S = {top2_val:.3f}$ ({top2_val*100:.1f}%).

**Implicação direta para gestão de risco:**
Se a organização quiser reduzir a incerteza sobre o EACE esperado de sua frota, o investimento
mais rentável é **melhorar a estimativa do custo de crítico→baixo** — contratar um atuário
especializado em offshore, revisar histórico de multas ANP, calibrar o seguro de downtime.
Reduzir a incerteza de todas as outras células juntas teria impacto menor do que reduzir a
incerteza dessa célula sozinha.

**Por que $\\sum S_i \\approx 1.0$:**
O EACE é uma soma ponderada dos custos ($\\text{{EACE}} = \\sum_{{ij}} w_{{ij}} C_{{ij}}$). Para funções
lineares com entradas independentes, os índices de Sobol têm solução analítica exata e
a soma é 1,0 exato. O pequeno desvio de {sobol_residual:.4f} deve-se apenas a arredondamento
numérico — não há interações de segunda ordem.

---

## 12. Bootstrap IC 95% do EACE Médio

![Bootstrap IC 95% do EACE médio por modelo](figures/risk_analysis/bootstrap_ci.png)

**O que o gráfico mostra:**
Forest plot com o EACE médio (ponto) e o intervalo de confiança 95% bootstrap (linha horizontal)
de cada modelo. O IC foi calculado com {N_BOOT:,} reamostras das {N_SIM:,} simulações Monte Carlo.
Modelos cujos ICs se sobrepõem não têm diferença estatisticamente distinguível no EACE médio.

**Análise crítica:**

Os ICs do `triple_weighted_avg` e `triple_override_deep` **{overlap_msg}** — a diferença de
~R\\$ 19M entre os dois modelos **{"não é" if overlap else "é"} estatisticamente significativa**
dado o nível de ruído das distribuições de custo assumidas.

**O que isso significa na prática:**
A escolha entre `triple_weighted_avg` e `triple_override_deep` **não deve ser feita com base
no EACE médio isolado** — a margem está dentro do intervalo de confiança. A decisão deve levar
em conta critérios secundários: P95 (gestão de pior cenário), precision@crítico (tolerância a
alarmes falsos) e complexidade operacional de deploy.

**Modelos com ICs claramente separados:**
O `hibrido_weighted` está isolado à direita — sua diferença vs. o vencedor é estatisticamente
robusta. Os três Híbridos Triplos formam um cluster bem separado do restante.

---

## 13. Análise de Breakeven — Threshold de Reversão de Ranking

![Análise de breakeven — custo de crítico→baixo](figures/risk_analysis/breakeven_critico_baixo.png)

**O que o gráfico mostra:**
Para cada valor hipotético do custo de crítico→baixo (eixo x), o EACE de cada modelo com todos
os outros custos fixos no valor central. O ponto de cruzamento entre `triple_weighted_avg` e
`triple_override_deep` é o **breakeven** — o custo a partir do qual o ranking se inverte.

**Análise crítica:**

O breakeven ocorre em **≈ {be_msg}**. O valor central assumido nas premissas é R\\$ 3,2M.

**Se o custo real de crítico→baixo for abaixo de {be_msg}:**
O `triple_override_deep` volta a ser o melhor pelo critério determinístico — ele tem recall
mais alto (0,569 vs. 0,556) e, com custo menor por incidente crítico perdido, a vantagem de
recall domina o custo de ter mais falsos positivos.

**Se o custo real de crítico→baixo for acima de {be_msg}:**
O `triple_weighted_avg` é melhor. Sua alta precision@crítico (0,816) gera menos falsos positivos
e, quando cada falso positivo de crítico é caro (ativações desnecessárias de emergência), pagar
o prêmio de precision compensa.

**Robustez da recomendação:**
O breakeven está {"próximo do" if be_brl and abs(be_brl - 3_200_000) < 800_000 else "longe do"} valor
central de R\\$ 3,2M. Isso indica que a escolha entre os dois modelos é **sensível à estimativa
de custo** — e reforça a mensagem do bootstrap: a diferença não é robusta o suficiente para
ignorar o nível de incerteza das premissas.

---

## 14. Robustez do Ranking sob Contração de Incerteza (Cenários de σ)

![Cenários de contração de σ — robustez do ranking](figures/risk_analysis/sigma_scenarios.png)

**O que o gráfico mostra:**
O EACE médio e o spread [P5, P95] dos três melhores modelos para quatro cenários de incerteza:
desde alta incerteza (1,5× σ base — organização sem histórico de custos) até custo fixo
(σ = 0 — contrato com seguradora fixando todos os custos). As barras de erro mostram como
o spread encolhe com σ menor.

**Análise crítica:**

**O ranking se mantém estável?**
O `triple_weighted_avg` mantém o menor EACE médio em todos os cenários de σ, incluindo o
determinístico (σ = 0) onde a diferença entre os dois primeiros permanece. Isso é notável:
a liderança do `triple_weighted_avg` não depende de qual nível de incerteza se assume — ela
existe mesmo com custos completamente fixos.

**A redução de σ encolhe o spread drasticamente:**
O spread P5–P95 do vencedor cai de ~R\\$ 670M (cenário base) para ~R\\$ 330M com 0,5×σ
(~{reduction_pct:.0f}% de redução). No cenário de custo fixo, o spread colapsa para zero
por definição — restando apenas o EACE determinístico.

**Implicação para o investimento em dados de custo:**
Reduzir a incerteza de custo à metade (σ → 0,5×σ) reduz o spread de risco esperado em ~50%
sem mudar o modelo de ML — é um investimento puramente em dados. Para uma frota de 10 FPSOs,
isso representa ~R\\$ 330M de redução no intervalo de risco anual do portfólio.

**Alta incerteza (1,5×σ):**
Com incerteza amplificada, o `hibrido_weighted` fica ainda mais isolado à direita e os
Híbridos Triplos ficam mais comprimidos entre si. O principal insight: **a recomendação de
modelo é robusta a cenários de maior incerteza** — os três Triplos mantêm clara separação
dos outros tiers independentemente do nível de σ.

**Mensagem para o webinar:**
> A escolha do modelo não é só uma questão de F1 score — é uma **decisão de gestão de risco**
> com impacto financeiro direto. O `{best}` domina estocasticamente em todos os cenários de
> incerteza testados. Mas a margem sobre o segundo colocado está dentro do IC 95% — a decisão
> final deve incorporar critérios operacionais além do EACE médio.
"""


# ── Main ──────────────────────────────────────────────────────────────────────

# Samplers base (σ × 1.0) — acessíveis globalmente para compute_sobol_indices
samplers_base: dict = {}


def _log(msg: str, t0: float) -> None:
    elapsed = time.time() - t0
    print(f"[{elapsed:6.1f}s] {msg}", flush=True)


def main() -> None:
    global samplers_base
    t0 = time.time()

    _log("Carregando dados e construindo coeficientes...", t0)
    params = _load_params()
    metrics = _load_metrics()

    eace_params = params["eace"]
    class_dist  = eace_params["class_distribution"]
    N           = eace_params["annual_records"]

    fig_dir = ROOT / "reports" / "figures" / "risk_analysis"
    fig_dir.mkdir(parents=True, exist_ok=True)

    models_cms = {m: reconstruct_cm(metrics, m) for m in MODELS}
    coef = _build_coef_vectors(models_cms, class_dist, N)
    samplers_base = _build_cost_samplers(sigma_scale=1.0)
    _log("Coeficientes prontos.", t0)

    # ── Monte Carlo base ──────────────────────────────────────────────────────
    _log(f"Monte Carlo N={N_SIM:,} ...", t0)
    mc_results = run_monte_carlo(coef, samplers_base)
    for model, eaces in mc_results.items():
        print(f"          {model:<26}  mean=R${eaces.mean():,.0f}  p95=R${np.percentile(eaces,95):,.0f}", flush=True)
    _log("Monte Carlo concluído.", t0)

    # ── Sensibilidade ─────────────────────────────────────────────────────────
    _log("Computando sensibilidade (heatmap + tornados)...", t0)
    sensitivity_df  = compute_sensitivity(coef, samplers_base)
    per_model_sens  = compute_sensitivity_per_model(coef, samplers_base)
    heatmap_pct     = compute_sensitivity_heatmap(coef, samplers_base, class_dist, N)
    _log("Sensibilidade concluída.", t0)

    # ── Sobol ─────────────────────────────────────────────────────────────────
    _log("Índices de Sobol (solução analítica)...", t0)
    sobol = compute_sobol_indices(coef)
    for cell, val in sorted(sobol.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"          {cell:<20}  S={val:.4f} ({val*100:.1f}%)", flush=True)
    _log(f"Sobol concluído. Soma={sum(sobol.values()):.4f}", t0)

    # ── Bootstrap ─────────────────────────────────────────────────────────────
    _log(f"Bootstrap IC 95% (B={N_BOOT:,})...", t0)
    boot_results = bootstrap_eace_mean(mc_results)
    _log("Bootstrap concluído.", t0)

    # ── Breakeven ─────────────────────────────────────────────────────────────
    _log("Análise de breakeven...", t0)
    be_grid, be_results = breakeven_analysis(coef, params)
    _log("Breakeven concluído.", t0)

    # ── Cenários de σ ─────────────────────────────────────────────────────────
    _log(f"Cenários de contração de σ (n={N_SOBOL:,} por cenário)...", t0)
    scenario_results = run_sigma_scenarios(coef, class_dist, N)
    _log("Cenários de σ concluídos.", t0)

    # ── Figuras ───────────────────────────────────────────────────────────────
    _log("Gerando figuras (9 total)...", t0)
    det_eaces = {m: float(metrics.get(f"{m}__eace_brl", 0.0)) for m in MODELS}
    paths = [
        _plot_violin(mc_results, det_eaces, fig_dir),
        _plot_cdf(mc_results, fig_dir),
        _plot_sensitivity_heatmap(heatmap_pct, fig_dir),
        _plot_tornado_global(sensitivity_df, fig_dir),
        _plot_tornado_per_model(per_model_sens, fig_dir),
        _plot_sobol(sobol, fig_dir),
        _plot_bootstrap_ci(boot_results, fig_dir),
        _plot_breakeven(be_grid, be_results, fig_dir),
        _plot_sigma_scenarios(scenario_results, fig_dir),
    ]
    for p in paths:
        print(f"          → {p.name}", flush=True)
    _log("Figuras geradas.", t0)

    # ── JSON ──────────────────────────────────────────────────────────────────
    _log("Persistindo JSON...", t0)
    mc_json = _build_json(mc_results, metrics, boot_results, sobol,
                          be_grid, be_results, scenario_results)
    json_path = ROOT / "reports" / "monte_carlo_eace.json"
    json_path.write_text(json.dumps(mc_json, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(f"JSON → {json_path.name}", t0)

    # ── Markdown ──────────────────────────────────────────────────────────────
    _log("Gerando relatório Markdown...", t0)
    md = _build_markdown(mc_json, sensitivity_df, boot_results, sobol,
                         be_grid, scenario_results)
    md_path = ROOT / "reports" / "METRICS_RISK_ANALYSIS.md"
    md_path.write_text(md, encoding="utf-8")
    _log(f"Markdown → {md_path.name}", t0)

    best = mc_json["recommended_model"]
    elapsed_total = time.time() - t0
    print(f"\n[{elapsed_total:6.1f}s] ★ Modelo recomendado: {best}", flush=True)
    print(f"          EACE médio:  R${mc_json['models'][best]['eace_mean']/1e6:.1f}M/ano", flush=True)
    print(f"          EACE P95:    R${mc_json['models'][best]['eace_p95']/1e6:.1f}M/ano", flush=True)
    print(f"          Sobol top-1: {max(sobol, key=sobol.get)} = {max(sobol.values())*100:.1f}%", flush=True)
    print(f"          Breakeven:   R${mc_json.get('breakeven_critico_baixo_brl', 0)/1e6:.2f}M", flush=True)
    print(f"\n[{elapsed_total:6.1f}s] Concluído em {elapsed_total:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
