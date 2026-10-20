"""
AFS-VFM Early Warning Failure Anticipation System
=================================================

Monitors intermediate VFM representations during progressive corruption,
evaluates anticipatory signal drift against empirical failure boundaries,
and issues early-warning alerts before catastrophic prediction breakdown.

Public API
----------
- ``FailurePredictor``   — real-time inference health and failure risk evaluator
- ``learn_thresholds``   — optimal threshold optimization maximizing F1 score
- ``MODEL_SIGNALS``      — canonical signal monitoring mappings per model
- ``LEARNED_THRESHOLDS`` — cached empirical optimal thresholds dictionary
"""

import os
import json
from typing import Dict, Any

from .failure_predictor import FailurePredictor

# Canonical signal mappings (defined here to avoid heavy dependencies on import)
MODEL_SIGNALS: Dict[str, list] = {
    "dinov2": ["top1_confidence", "logit_margin", "cosine_sim"],
    "dinov2_large": ["top1_confidence", "logit_margin", "cosine_sim"],
    "clip":   ["top1_confidence", "logit_margin", "cosine_sim"],
    "siglip": ["top1_confidence", "logit_margin", "cosine_sim"],
    "detr":   ["top_detection_confidence", "cosine_sim"],
}

# Expose cached learned thresholds if present
_THRESHOLDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thresholds.json")
if os.path.exists(_THRESHOLDS_PATH):
    try:
        with open(_THRESHOLDS_PATH, "r", encoding="utf-8") as _f:
            LEARNED_THRESHOLDS: Dict[str, Any] = json.load(_f)
    except Exception:
        LEARNED_THRESHOLDS = {}
else:
    LEARNED_THRESHOLDS = {}


def __getattr__(name: str) -> Any:
    if name == "learn_thresholds":
        from .learn_thresholds import learn_thresholds
        return learn_thresholds
    elif name == "LearnedFailurePredictor":
        from .learned_predictor import LearnedFailurePredictor
        return LearnedFailurePredictor
    elif name in ("build_classifier", "prepare_training_dataset", "evaluate_cross_validated_models"):
        from . import learned_predictor
        return getattr(learned_predictor, name)
    elif name in ("TemporalFeatureExtractor", "CORE_FEATURE_NAMES"):
        from . import feature_extractor
        return getattr(feature_extractor, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FailurePredictor",
    "LearnedFailurePredictor",
    "TemporalFeatureExtractor",
    "CORE_FEATURE_NAMES",
    "build_classifier",
    "prepare_training_dataset",
    "evaluate_cross_validated_models",
    "learn_thresholds",
    "MODEL_SIGNALS",
    "LEARNED_THRESHOLDS",
]

