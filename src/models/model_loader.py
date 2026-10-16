"""
Vision Foundation Model Loader for AFS-VFM.

Loads and prepares DINOv2, CLIP, and DETR for evaluation mode inference.
Each model is placed on the best available device (CUDA > MPS > CPU).

Usage
-----
>>> from src.models.model_loader import VisionModelLoader
>>> loader = VisionModelLoader()
>>> model, processor = loader.load_dinov2()
"""

from typing import Tuple
import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    CLIPProcessor,
    CLIPModel,
    DetrImageProcessor,
    DetrForObjectDetection,
)


def get_device() -> torch.device:
    """Return the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class VisionModelLoader:
    """Central loader for all three vision foundation models.

    Attributes
    ----------
    device : torch.device
        The device models are loaded onto.
    """

    def __init__(self) -> None:
        self.device = get_device()
        print(f"[ModelLoader] Using device: {self.device}")

    # ── DINOv2 (Image Classification) ──────────────────────────────────

    def load_dinov2(self) -> Tuple[AutoModelForImageClassification, AutoImageProcessor]:
        """Load Meta DINOv2 for ImageNet-1k classification."""
        print("  Loading DINOv2 (facebook/dinov2-base-imagenet1k-1-layer) ...")
        model_id = "facebook/dinov2-base-imagenet1k-1-layer"

        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id)

        model.to(self.device)
        model.eval()
        return model, processor

    # ── CLIP (Zero-Shot Classification) ────────────────────────────────

    def load_clip(self) -> Tuple[CLIPModel, CLIPProcessor]:
        """Load OpenAI CLIP ViT-B/32 for zero-shot classification."""
        print("  Loading CLIP (openai/clip-vit-base-patch32) ...")
        model_id = "openai/clip-vit-base-patch32"

        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id)

        model.to(self.device)
        model.eval()
        return model, processor

    # ── DETR (Object Detection) ────────────────────────────────────────

    def load_detr(self) -> Tuple[DetrForObjectDetection, DetrImageProcessor]:
        """Load Meta DETR ResNet-50 for object detection."""
        print("  Loading DETR (facebook/detr-resnet-50) ...")
        model_id = "facebook/detr-resnet-50"

        processor = DetrImageProcessor.from_pretrained(model_id)
        model = DetrForObjectDetection.from_pretrained(model_id)

        model.to(self.device)
        model.eval()
        return model, processor


if __name__ == "__main__":
    loader = VisionModelLoader()

    dino_model, _ = loader.load_dinov2()
    print("  DINOv2 loaded successfully.\n")

    clip_model, _ = loader.load_clip()
    print("  CLIP loaded successfully.\n")

    detr_model, _ = loader.load_detr()
    print("  DETR loaded successfully.\n")
