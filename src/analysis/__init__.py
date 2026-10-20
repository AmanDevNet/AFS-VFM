"""
AFS-VFM Statistical Analysis & Failure Mapping Pipeline
======================================================

Analytical tools for evaluating Vision Foundation Model empirical failure boundaries,
anticipatory internal signal dynamics, and stratified diagnostic dataset selection.

Public API
----------
- ``extract_failure_boundaries`` — identifies first-failure frames across (image, degradation, model)
- ``compute_failure_statistics``  — aggregates survival and failure rates per model-degradation group
- ``merge_boundaries_and_signals`` — aligns internal feature trajectories with empirical failure events
- ``select_strategic_subset``      — stratifies benchmark samples into fragile, robust, and disagreement sets
- ``cross_model_comparison``       — multi-model comparative failure and self-awareness synthesis
- ``compute_robustness_rankings``  — rankings across models and degradations
- ``compute_failure_correlation``  — Jaccard similarity and failure overlap across architectures
- ``compute_complementarity``      — multi-model ensemble survival benefits
"""

import importlib
from typing import Any


def __getattr__(name: str) -> Any:
    if name in ("extract_failure_boundaries", "compute_failure_statistics"):
        mod = importlib.import_module(f"{__name__}.failure_boundaries")
        return getattr(mod, name)
    elif name == "merge_boundaries_and_signals":
        mod = importlib.import_module(f"{__name__}.anticipatory_signals")
        return getattr(mod, name)
    elif name == "select_strategic_subset":
        mod = importlib.import_module(f"{__name__}.select_subset")
        return getattr(mod, name)
    elif name in (
        "cross_model_comparison",
        "discover_models",
        "compute_robustness_rankings",
        "compute_failure_correlation",
        "compute_complementarity",
        "analyze_architecture_effects",
        "rank_signal_awareness",
        "generate_synthesis",
    ):
        mod = importlib.import_module(f"{__name__}.cross_model_comparison")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "extract_failure_boundaries",
    "compute_failure_statistics",
    "merge_boundaries_and_signals",
    "select_strategic_subset",
    "cross_model_comparison",
    "discover_models",
    "compute_robustness_rankings",
    "compute_failure_correlation",
    "compute_complementarity",
    "analyze_architecture_effects",
    "rank_signal_awareness",
    "generate_synthesis",
]
