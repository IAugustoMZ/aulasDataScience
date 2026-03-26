"""
run_eda.py — EDA Pipeline Orchestrator

Executes all EDA modules in sequence (or a specified subset), collects
timing and output metadata, and writes a final JSON run summary.

Usage
-----
# From the aula02 directory:
    python eda/run_eda.py
    python eda/run_eda.py --steps descriptive pivot_tables
    python eda/run_eda.py --config config/ --base-dir .
    python eda/run_eda.py --steps all --force

Modules executed (in order)
----------------------------
1. descriptive          — stats, missing values, correlations
2. visualizations       — all plots (histograms, maps, heatmaps)
3. pivot_tables         — pivot and contingency tables
4. statistical_tests    — ANOVA, Kruskal-Wallis, post-hoc tests
5. interactions         — 2-way and 3-way interaction effects
6. feature_engineering  — ratio/log/distance/polynomial features
7. clustering           — K-Means (geo + feature-space), DBSCAN, PCA
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# ── Path bootstrap ─────────────────────────────────────────────────────────────
# Allow running from: aula02/, eda/, or anywhere via absolute path
_THIS_FILE = Path(__file__).resolve()
_EDA_DIR   = _THIS_FILE.parent
_BASE_DIR  = _EDA_DIR.parent
if str(_BASE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))
if str(_EDA_DIR) not in sys.path:
    sys.path.insert(0, str(_EDA_DIR))

from src.utils.config_loader import load_config  # noqa: E402
from src.utils.logger import get_logger           # noqa: E402

# Default logging config — used before pipeline.yaml is loaded
_DEFAULT_LOG_CFG: dict = {
    "level": "INFO",
    "format": "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    "datefmt": "%Y-%m-%d %H:%M:%S",
    "log_to_file": False,
}
logger = get_logger("eda.orchestrator", _DEFAULT_LOG_CFG)

# ── Module registry ────────────────────────────────────────────────────────────
# Ordered list of (step_name, module_path, import_name)
_STEPS: list[tuple[str, str]] = [
    ("descriptive",         "eda.descriptive"),
    ("visualizations",      "eda.visualizations"),
    ("pivot_tables",        "eda.pivot_tables"),
    ("statistical_tests",   "eda.statistical_tests"),
    ("interactions",        "eda.interactions"),
    ("feature_engineering", "eda.feature_engineering"),
    ("clustering",          "eda.clustering"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="California Housing EDA Pipeline Orchestrator",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config",
        help="Path to config directory (default: config/)",
    )
    parser.add_argument(
        "--base-dir",
        default=str(_BASE_DIR),
        help=f"Base directory for relative paths (default: {_BASE_DIR})",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["all"],
        help=(
            "EDA steps to run (default: all).\n"
            "Valid values: all | descriptive | visualizations | pivot_tables |\n"
            "              statistical_tests | interactions | feature_engineering | clustering"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-run steps even if output files already exist",
    )
    return parser.parse_args()


def _resolve_steps(requested: list[str]) -> list[tuple[str, str]]:
    """Return ordered list of (step_name, module) to execute."""
    if "all" in requested:
        return _STEPS
    valid_names = {name for name, _ in _STEPS}
    unknown = set(requested) - valid_names
    if unknown:
        raise ValueError(
            f"Unknown EDA step(s): {unknown}\n"
            f"Valid steps: {sorted(valid_names)}"
        )
    step_map = dict(_STEPS)
    return [(name, step_map[name]) for name in requested if name in step_map]


def _import_step(module_path: str) -> Any:
    """Dynamically import an EDA module and return it."""
    import importlib
    return importlib.import_module(module_path)


def _ensure_output_dirs(config: dict, base_dir: Path) -> None:
    """Create output directories if they don't exist."""
    dirs = [
        config.get("paths", {}).get("figures_dir", "outputs/figures"),
        config.get("paths", {}).get("tables_dir",  "outputs/tables"),
        config.get("paths", {}).get("stats_dir",   "outputs/stats"),
        config.get("paths", {}).get("docs_dir",    "docs"),
    ]
    for d in dirs:
        (base_dir / d).mkdir(parents=True, exist_ok=True)
    logger.info("Output directories verified/created.")


def _write_run_summary(
    results: dict[str, Any],
    base_dir: Path,
    config: dict,
) -> Path:
    """Write a JSON run summary with timing and step outcomes."""
    summary_path = base_dir / config.get("paths", {}).get("stats_dir", "outputs/stats") / "00_run_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info(f"Run summary written to {summary_path}")
    return summary_path


# ── Main orchestrator ──────────────────────────────────────────────────────────

def run_eda(
    config_dir: str | Path = "config",
    base_dir: str | Path | None = None,
    steps: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run the full EDA pipeline.

    Args:
        config_dir: Directory containing pipeline.yaml, data.yaml, and eda.yaml.
        base_dir:   Root directory of the aula02 project.
        steps:      List of step names to run (default: all).

    Returns:
        dict with per-step results and overall run metadata.
    """
    pipeline_start = time.time()

    # ── Resolve paths ────────────────────────────────────────────────────────
    base_dir = Path(base_dir) if base_dir else _BASE_DIR
    config_dir = base_dir / config_dir if not Path(config_dir).is_absolute() else Path(config_dir)

    # ── Load configuration ───────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("California Housing EDA Pipeline — Starting")
    logger.info(f"Base directory : {base_dir}")
    logger.info(f"Config directory: {config_dir}")
    logger.info("=" * 70)

    try:
        config = load_config(config_dir)
        logger.info(
            f"Configuration loaded — EDA version: "
            f"{config.get('eda', {}).get('version', 'unknown')}"
        )
    except Exception as exc:
        logger.error(f"Failed to load configuration: {exc}")
        raise

    # Load eda-specific config (eda.yaml may need separate load since load_config merges pipeline+data)
    eda_yaml_path = config_dir / "eda.yaml"
    if eda_yaml_path.exists():
        import yaml
        with eda_yaml_path.open("r", encoding="utf-8") as fh:
            eda_cfg = yaml.safe_load(fh) or {}
        # Merge EDA config on top
        for k, v in eda_cfg.items():
            if k in config and isinstance(config[k], dict) and isinstance(v, dict):
                config[k] = {**config[k], **v}
            else:
                config[k] = v
        logger.info("EDA-specific configuration merged.")

    # ── Ensure output directories ────────────────────────────────────────────
    _ensure_output_dirs(config, base_dir)

    # ── Resolve which steps to run ───────────────────────────────────────────
    steps_to_run = _resolve_steps(steps or ["all"])
    logger.info(f"Steps to execute: {[s for s, _ in steps_to_run]}")

    # ── Execute steps ────────────────────────────────────────────────────────
    run_results: dict[str, Any] = {
        "pipeline": config.get("eda", {}).get("name", "california-housing-eda"),
        "version":  config.get("eda", {}).get("version", "1.0.0"),
        "base_dir": str(base_dir),
        "steps":    {},
    }

    for step_name, module_path in steps_to_run:
        logger.info("-" * 60)
        logger.info(f">> Running step: {step_name}")
        step_start = time.time()
        step_result: dict[str, Any] = {
            "status":   "pending",
            "elapsed_s": None,
            "error":    None,
            "output":   None,
        }
        try:
            module = _import_step(module_path)
            if not hasattr(module, "run"):
                raise AttributeError(
                    f"Module '{module_path}' has no 'run(config, base_dir)' function."
                )
            output = module.run(config, base_dir)
            step_result["status"]  = "success"
            step_result["output"]  = str(output)[:500] if output is not None else None
            logger.info(f"  [OK] Step '{step_name}' completed in {time.time() - step_start:.1f}s")
        except Exception as exc:
            step_result["status"] = "failed"
            step_result["error"]  = str(exc)
            logger.error(f"  [FAIL] Step '{step_name}' FAILED: {exc}")
            logger.debug(traceback.format_exc())
        finally:
            step_result["elapsed_s"] = round(time.time() - step_start, 2)
            run_results["steps"][step_name] = step_result

    # ── Summary ──────────────────────────────────────────────────────────────
    total_elapsed = round(time.time() - pipeline_start, 2)
    run_results["total_elapsed_s"] = total_elapsed
    run_results["success_count"] = sum(
        1 for s in run_results["steps"].values() if s["status"] == "success"
    )
    run_results["failed_count"] = sum(
        1 for s in run_results["steps"].values() if s["status"] == "failed"
    )

    logger.info("=" * 70)
    logger.info(
        f"EDA Pipeline complete — "
        f"{run_results['success_count']} succeeded, "
        f"{run_results['failed_count']} failed — "
        f"{total_elapsed:.1f}s total"
    )
    logger.info("=" * 70)

    # ── Write run summary ────────────────────────────────────────────────────
    try:
        _write_run_summary(run_results, base_dir, config)
    except Exception as exc:
        logger.warning(f"Could not write run summary: {exc}")

    return run_results


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> int:
    args = _parse_args()

    try:
        results = run_eda(
            config_dir=args.config,
            base_dir=args.base_dir,
            steps=args.steps,
        )
    except Exception as exc:
        logger.error(f"Pipeline aborted: {exc}")
        logger.debug(traceback.format_exc())
        return 1

    failed = results.get("failed_count", 0)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
