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
"""

from typing import Any


def __getattr__(name: str) -> Any:
    if name in ("extract_failure_boundaries", "compute_failure_statistics"):
        from .failure_boundaries import extract_failure_boundaries, compute_failure_statistics
        if name == "extract_failure_boundaries":
            return extract_failure_boundaries
        return compute_failure_statistics
    elif name == "merge_boundaries_and_signals":
        from .anticipatory_signals import merge_boundaries_and_signals
        return merge_boundaries_and_signals
    elif name == "select_strategic_subset":
        from .select_subset import select_strategic_subset
        return select_strategic_subset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "extract_failure_boundaries",
    "compute_failure_statistics",
    "merge_boundaries_and_signals",
    "select_strategic_subset",
]
