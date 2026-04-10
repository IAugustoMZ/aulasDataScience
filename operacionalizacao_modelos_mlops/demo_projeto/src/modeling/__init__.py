"""
modeling/ — Módulo de Experimentação e Modelagem MLOps.

API pública:
  • ModelingStep       — etapa PipelineStep completa (orquestra toda a modelagem)
  • FeatureReducer     — redutor de features sklearn-compatível (none|rfe|pca|kpca)

Submódulos internos:
  base           — ABCs: BaseOptimizer, BaseEvaluator
  metrics        — compute_metrics(), aggregate_fold_metrics()
  model_factory  — build_model(), build_pipeline()
  cross_validation — CVRunner (KFold leak-free)
  optimizer      — OptimizerFactory + OptunaOptimizer, GridSearchOptimizer, RandomizedSearchOptimizer
  ensemble       — EnsembleBuilder (Stacking + Voting)
  evaluator      — HoldoutEvaluator
  artifacts      — ArtifactGenerator (plots diagnósticos)
  tracker        — MLflowTracker (setup, log, registry)
  reducer        — FeatureReducer
  step           — ModelingStep
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from src.modeling.reducer import FeatureReducer

if TYPE_CHECKING:
    from src.modeling.step import ModelingStep

__all__ = ["FeatureReducer", "ModelingStep"]


def __getattr__(name: str):  # noqa: D401
    """Importação lazy de ModelingStep para evitar importações circulares."""
    if name == "ModelingStep":
        from src.modeling.step import ModelingStep as _MS
        return _MS
    raise AttributeError(f"module 'src.modeling' has no attribute {name!r}")
