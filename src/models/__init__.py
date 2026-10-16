"""
AFS-VFM Vision Model Inference Pipeline
========================================

Provides loaders and evaluation wrappers for the three foundation models
under test: DINOv2, CLIP, and DETR.

Public API
----------
- ``VisionModelLoader`` — loads any of the three models onto the best device
- ``SignalExtractor``   — extracts latent representations, embeddings, and drift metrics
"""

from .model_loader import VisionModelLoader
from .signal_extractor import SignalExtractor

__all__ = [
    "VisionModelLoader",
    "SignalExtractor",
]
