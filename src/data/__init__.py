"""
AFS-VFM Dataset Curation and Ingestion Pipeline
===============================================

Utilities for curating, resizing, and generating ground-truth annotations
for ImageNet-1k and COCO-val2017 benchmark subsets.

Public API
----------
- ``curate_imagenet`` — streams and processes ImageNet-1k validation images
- ``curate_coco``     — downloads and preprocesses COCO val2017 detection images
"""

from typing import Any

def __getattr__(name: str) -> Any:
    if name == "curate_imagenet":
        from .curate_datasets import curate_imagenet
        return curate_imagenet
    elif name == "curate_coco":
        from .curate_coco import curate_coco
        return curate_coco
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "curate_imagenet",
    "curate_coco",
]
