"""
AFS-VFM Vision Model Inference Pipeline
========================================

Provides loaders and evaluation wrappers for the five foundation models
under test: DINOv2-Base, DINOv2-Large, CLIP, SigLIP, and DETR.

Public API
----------
- ``VisionModelLoader`` — loads models across multiple scales and paradigms onto compute device
- ``SignalExtractor``   — extracts latent representations, embeddings, and drift metrics
"""

from .model_loader import VisionModelLoader
from .signal_extractor import SignalExtractor

__all__ = [
    "VisionModelLoader",
    "SignalExtractor",
]
