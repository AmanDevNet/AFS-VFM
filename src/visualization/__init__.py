"""
AFS-VFM Visualization & Publication Figure Suite
================================================

Generates publication-ready scientific visualizations, robustness heatmaps,
signal trajectories, and multi-architecture radar comparisons for Vision Foundation Models.

Public API
----------
- ``set_publication_style`` — sets journal ('paper') or dashboard ('dark') matplotlib style
- ``setup_style``          — backward-compatible dark theme configuration
- ``MODEL_COLORS``         — standardized scientific color palette per model
- ``MODEL_LABELS``         — publication-formatted display names
- ``DEG_ORDER``            — canonical degradation ordering
- ``DEG_LABELS``           — clean display labels for corruptions
- ``generate_all_figures`` — generates complete 9-figure paper suite
"""

from typing import Any

from .generate_figures import (
    MODEL_COLORS,
    MODEL_LABELS,
    DEG_ORDER,
    DEG_LABELS,
    set_publication_style,
    setup_style,
)


def __getattr__(name: str) -> Any:
    if name == "generate_all_figures":
        from .generate_figures import generate_all_figures
        return generate_all_figures
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "set_publication_style",
    "setup_style",
    "MODEL_COLORS",
    "MODEL_LABELS",
    "DEG_ORDER",
    "DEG_LABELS",
    "generate_all_figures",
]
