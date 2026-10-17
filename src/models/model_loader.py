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

from typing import Tuple, Any
import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoProcessor,
    AutoModel,
    CLIPProcessor,
    CLIPModel,
    DetrImageProcessor,
    DetrForObjectDetection,
    SiglipProcessor,
    SiglipModel,
)


def get_device() -> torch.device:
    """Return the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class VisionModelLoader:
    """Central loader for vision foundation models across multiple scales and paradigms.

    Supported Models:
    -----------------
    - ``dinov2`` / ``dinov2_base``: Meta DINOv2-Base (ViT-B/14, 86M params, 768-dim)
    - ``dinov2_large``: Meta DINOv2-Large (ViT-L/14, 304M params, 1024-dim)
    - ``clip``: OpenAI CLIP ViT-B/32 (Softmax contrastive pretraining)
    - ``siglip``: Google SigLIP ViT-B/16 (Pairwise sigmoid BCE loss)
    - ``detr``: Meta DETR ResNet-50 (End-to-end transformer object detection)

    Attributes
    ----------
    device : torch.device
        The device models are loaded onto.
    """

    def __init__(self) -> None:
        self.device = get_device()
        print(f"[ModelLoader] Using device: {self.device}")

    # ── DINOv2-Base (Image Classification) ──────────────────────────────

    def load_dinov2(self) -> Tuple[AutoModelForImageClassification, AutoImageProcessor]:
        """Load Meta DINOv2-Base (ViT-B/14, 86M params) for ImageNet-1k classification."""
        print("  Loading DINOv2-Base (facebook/dinov2-base-imagenet1k-1-layer) ...")
        model_id = "facebook/dinov2-base-imagenet1k-1-layer"

        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id)

        model.to(self.device)
        model.eval()
        return model, processor

    # ── DINOv2-Large (Image Classification - Scale Comparison) ───────────

    def load_dinov2_large(self) -> Tuple[AutoModelForImageClassification, AutoImageProcessor]:
        """Load Meta DINOv2-Large (ViT-L/14, 304M params, 1024-dim) for scale comparison."""
        print("  Loading DINOv2-Large (facebook/dinov2-large-imagenet1k-1-layer) ...")
        model_id = "facebook/dinov2-large-imagenet1k-1-layer"

        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id)

        model.to(self.device)
        model.eval()
        return model, processor

    # ── CLIP (Zero-Shot Classification - Softmax Loss) ──────────────────

    def load_clip(self) -> Tuple[CLIPModel, CLIPProcessor]:
        """Load OpenAI CLIP ViT-B/32 for zero-shot classification."""
        print("  Loading CLIP (openai/clip-vit-base-patch32) ...")
        model_id = "openai/clip-vit-base-patch32"

        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id)

        model.to(self.device)
        model.eval()
        return model, processor

    # ── SigLIP (Zero-Shot Classification - Sigmoid BCE Loss) ─────────────

    def load_siglip(self) -> Tuple[Any, Any]:
        """Load Google SigLIP (google/siglip-base-patch16-224) with sigmoid contrastive loss."""
        print("  Loading SigLIP (google/siglip-base-patch16-224) ...")
        model_id = "google/siglip-base-patch16-224"

        try:
            processor = SiglipProcessor.from_pretrained(model_id)
            model = SiglipModel.from_pretrained(model_id)
        except Exception:
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModel.from_pretrained(model_id)

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

    # ── Unified Model Dispatcher ───────────────────────────────────────

    def load_model(self, model_name: str) -> Tuple[Any, Any]:
        """Unified loader dispatcher across all supported vision foundation models.

        Parameters
        ----------
        model_name : str
            One of: 'dinov2', 'dinov2_base', 'dinov2_large', 'clip', 'siglip', 'detr'.
        """
        key = model_name.strip().lower().replace("-", "_")
        if key in ("dinov2", "dinov2_base"):
            return self.load_dinov2()
        elif key in ("dinov2_large", "dinov2_l"):
            return self.load_dinov2_large()
        elif key == "clip":
            return self.load_clip()
        elif key == "siglip":
            return self.load_siglip()
        elif key == "detr":
            return self.load_detr()
        else:
            raise ValueError(
                f"Unknown model_name '{model_name}'. "
                f"Supported: 'dinov2', 'dinov2_large', 'clip', 'siglip', 'detr'."
            )


if __name__ == "__main__":
    loader = VisionModelLoader()

    dino_model, _ = loader.load_dinov2()
    print("  DINOv2-Base loaded successfully.\n")

    clip_model, _ = loader.load_clip()
    print("  CLIP loaded successfully.\n")

    detr_model, _ = loader.load_detr()
    print("  DETR loaded successfully.\n")

