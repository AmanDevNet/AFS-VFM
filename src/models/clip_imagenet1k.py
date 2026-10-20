"""
ImageNet-1K Zero-Shot Evaluator & Text Embedding Cache for CLIP and SigLIP.
==========================================================================

Resolves Limitation 10:
Expands CLIP and SigLIP zero-shot candidate set from toy 20-label lists to the
canonical 1,000-class ImageNet-1K hierarchy.

Key Features:
-------------
1. Prompt Ensembling:
   Averages normalized text embeddings across canonical templates:
       w_c = Normalize( (1/T) * sum_t Normalize(E_text(template_t(class_c))) )
   significantly improving zero-shot classification robustness and reducing prompt bias.

2. Precomputed Text Feature Caching:
   Computes W_text in R^{1000 x d} once and caches it to disk (results/cache/*.pt).
   Per-frame zero-shot evaluation is reduced to a single GEMM (image_features @ W_text.T),
   executing in < 0.2 ms per frame instead of re-running the text transformer on 1,000
   tokens per frame.

3. Canonical Alignment:
   Class index 0..999 strictly matches DINOv2 (facebook/dinov2-base-imagenet1k-1-layer),
   enabling fair, direct 1:1 cross-model comparison on the exact same class hierarchy.
"""

import os
from typing import List, Optional, Union, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from data.imagenet_classes import (
    IMAGENET_1K_CLASSES,
    IMAGENET_PROMPT_TEMPLATES,
    get_clean_label,
    get_imagenet_classes,
)

# Standard default cache directory
DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results",
    "cache",
)

# Top 10 high-impact prompt templates for balanced ensembling speed & accuracy
CANONICAL_ENSEMBLE_TEMPLATES = [
    "a photo of a {}.",
    "a photo of the {}.",
    "a clean photo of a {}.",
    "a close-up photo of a {}.",
    "a bright photo of a {}.",
    "a low resolution photo of the {}.",
    "a blurry photo of the {}.",
    "a cropped photo of a {}.",
    "a good photo of a {}.",
    "a photo of a small {}.",
]


def extract_embedding_tensor(features: Any) -> torch.Tensor:
    """Extract a 2D embedding tensor (batch_size, embed_dim) from model outputs.

    Ensures seamless compatibility across transformers v4 and v5+:
    - In transformers v4, model.get_text_features() and get_image_features() returned a bare torch.Tensor.
    - In transformers v5+, these methods return a ModelOutput dataclass
      (e.g., BaseModelOutputWithPooling, CLIPTextModelOutput, CLIPVisionModelOutput)
      where the projected pooled embedding resides in .pooler_output, .text_embeds, or .image_embeds.
    - In standard forward passes (model(**inputs)), output contains .text_embeds or .image_embeds.
    """
    if isinstance(features, torch.Tensor):
        return features

    # Check for direct attribute fields on ModelOutput / BaseModelOutputWithPooling
    if hasattr(features, "text_embeds") and features.text_embeds is not None:
        return features.text_embeds
    if hasattr(features, "image_embeds") and features.image_embeds is not None:
        return features.image_embeds
    if hasattr(features, "pooler_output") and features.pooler_output is not None:
        return features.pooler_output
    if hasattr(features, "last_hidden_state") and features.last_hidden_state is not None:
        lhs = features.last_hidden_state
        return lhs[:, 0] if lhs.dim() == 3 else lhs

    # Sequence / tuple output fallback
    if isinstance(features, (tuple, list)) and len(features) > 0:
        if len(features) > 1 and isinstance(features[1], torch.Tensor):
            return features[1]
        if isinstance(features[0], torch.Tensor):
            return features[0]

    raise TypeError(
        f"Unable to extract embedding tensor from object of type '{type(features).__name__}'. "
        f"Available attributes: {dir(features)}"
    )


class ImageNet1KZeroShotEvaluator:
    """Fast zero-shot evaluator over the 1,000-class ImageNet hierarchy.

    Parameters
    ----------
    model_type : str
        'clip' (softmax contrastive) or 'siglip' (pairwise sigmoid BCE).
    text_features : torch.Tensor
        Pre-normalized text embedding matrix of shape (1000, embed_dim).
    class_names : List[str]
        Canonical 1,000 class names.
    logit_scale : float or torch.Tensor
        Temperature scaling factor (e.g. exp(model.logit_scale)).
    logit_bias : float or torch.Tensor, optional
        Bias term for SigLIP (model.logit_bias).
    device : torch.device, optional
        Computation device (CUDA / CPU).
    """

    def __init__(
        self,
        model_type: str,
        text_features: torch.Tensor,
        class_names: List[str],
        logit_scale: Union[float, torch.Tensor],
        logit_bias: Optional[Union[float, torch.Tensor]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model_type = model_type.lower().strip()
        self.class_names = list(class_names)
        self.num_classes = len(self.class_names)
        self.device = device or (
            text_features.device if isinstance(text_features, torch.Tensor) else torch.device("cpu")
        )

        # Ensure text features are float tensor on target device and L2 normalized
        if not isinstance(text_features, torch.Tensor):
            text_features = torch.tensor(text_features, dtype=torch.float32)
        self.text_features = F.normalize(text_features.to(self.device, dtype=torch.float32), p=2, dim=-1)

        # Scale factor
        if isinstance(logit_scale, torch.Tensor):
            self.logit_scale = float(logit_scale.detach().cpu().item())
        else:
            self.logit_scale = float(logit_scale)

        # Bias factor (SigLIP only)
        if logit_bias is not None:
            if isinstance(logit_bias, torch.Tensor):
                self.logit_bias = float(logit_bias.detach().cpu().item())
            else:
                self.logit_bias = float(logit_bias)
        else:
            self.logit_bias = 0.0 if "siglip" in self.model_type else None

    def to(self, device: torch.device) -> "ImageNet1KZeroShotEvaluator":
        """Move evaluator tensors to specified device."""
        self.device = device
        self.text_features = self.text_features.to(device)
        return self

    def evaluate(
        self,
        image_embedding: Union[torch.Tensor, np.ndarray],
    ) -> Dict[str, Any]:
        """Compute zero-shot classification over 1,000 ImageNet classes.

        Parameters
        ----------
        image_embedding : torch.Tensor or np.ndarray
            Visual feature embedding of shape (embed_dim,) or (1, embed_dim).

        Returns
        -------
        dict
            Dictionary containing prediction, top1_confidence, logit_margin,
            prediction_entropy, top5_labels, top5_confs, full_probs, and full_logits.
        """
        # Format input tensor
        if not isinstance(image_embedding, (torch.Tensor, np.ndarray)):
            image_embedding = extract_embedding_tensor(image_embedding)

        if isinstance(image_embedding, np.ndarray):
            img_tensor = torch.from_numpy(image_embedding).to(self.device, dtype=torch.float32)
        else:
            img_tensor = image_embedding.to(self.device, dtype=torch.float32)

        if img_tensor.dim() == 1:
            img_tensor = img_tensor.unsqueeze(0)  # (1, d)

        # L2 normalize image feature
        img_norm = F.normalize(img_tensor, p=2, dim=-1)

        with torch.no_grad():
            # Raw cosine similarity dot product: (1, 1000)
            similarity = torch.matmul(img_norm, self.text_features.T)

            # Apply scale & bias
            if "siglip" in self.model_type:
                bias = self.logit_bias if self.logit_bias is not None else 0.0
                logits = similarity * self.logit_scale + bias
                probs = torch.sigmoid(logits)
            else:
                # CLIP softmax contrastive
                logits = similarity * self.logit_scale
                probs = F.softmax(logits, dim=-1)

            # Top-5 extraction
            top5_probs_t, top5_indices_t = probs[0].topk(5)
            top5_probs = top5_probs_t.cpu().numpy()
            top5_indices = top5_indices_t.cpu().numpy()
            top5_labels = [self.class_names[idx] for idx in top5_indices]

            top1_conf = float(top5_probs[0])
            top2_conf = float(top5_probs[1]) if len(top5_probs) > 1 else 0.0
            logit_margin = float(top1_conf - top2_conf)
            prediction = top5_labels[0]

            # Entropy calculation across the 1,000-class distribution
            p_vec = probs[0]
            if "siglip" in self.model_type:
                # For SigLIP, normalize the sigmoid activations to compute distribution entropy
                p_dist = p_vec / (torch.sum(p_vec) + 1e-12)
                p_safe = torch.clamp(p_dist, min=1e-12)
                entropy = float(-torch.sum(p_safe * torch.log(p_safe)).item())
            else:
                p_safe = torch.clamp(p_vec, min=1e-12)
                entropy = float(-torch.sum(p_safe * torch.log(p_safe)).item())

        return {
            "prediction": prediction,
            "top1_confidence": top1_conf,
            "logit_margin": logit_margin,
            "prediction_entropy": entropy,
            "top5_labels": top5_labels,
            "top5_logits": top5_probs,
            "top5_confs": [float(p) for p in top5_probs],
            "full_probs": probs[0].cpu().numpy(),
            "full_logits": logits[0].cpu().numpy(),
        }


def build_imagenet1k_text_features(
    model: nn.Module,
    processor: Any,
    model_type: str = "clip",
    templates: Optional[List[str]] = None,
    classes: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
    verbose: bool = True,
) -> Tuple[torch.Tensor, float, Optional[float]]:
    """Compute prompt-ensembled text feature matrix for 1,000 ImageNet classes.

    Parameters
    ----------
    model : nn.Module
        HF CLIPModel or SiglipModel.
    processor : Any
        Corresponding tokenizer/processor.
    model_type : str
        'clip' or 'siglip'.
    templates : List[str], optional
        Prompt templates (defaults to CANONICAL_ENSEMBLE_TEMPLATES).
    classes : List[str], optional
        Class names (defaults to canonical IMAGENET_1K_CLASSES).
    device : torch.device, optional
        Device to run text encoding on.
    batch_size : int
        Batch size for text tokenization and forward pass.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    Tuple[torch.Tensor, float, Optional[float]]
        (text_features of shape (1000, embed_dim), logit_scale, logit_bias)
    """
    device = device or next(model.parameters()).device
    classes = classes or IMAGENET_1K_CLASSES
    templates = templates or CANONICAL_ENSEMBLE_TEMPLATES

    model.eval()
    clean_classes = [get_clean_label(c) for c in classes]

    if verbose:
        print(f"[TextCache] Ensembling {len(classes)} classes x {len(templates)} templates ({len(classes)*len(templates)} prompts)...")

    ensembled_embeddings = []

    with torch.no_grad():
        for i, class_name in enumerate(classes):
            clean_name = clean_classes[i]
            # Generate prompts for this single class
            class_prompts = [tmpl.format(clean_name) for tmpl in templates]

            # Tokenize & encode
            if "siglip" in model_type.lower():
                inputs = processor(
                    text=class_prompts,
                    return_tensors="pt",
                    padding="max_length",
                ).to(device)
            else:
                inputs = processor(
                    text=class_prompts,
                    return_tensors="pt",
                    padding=True,
                ).to(device)

            if hasattr(model, "get_text_features"):
                features = model.get_text_features(**inputs)
            elif hasattr(model, "text_model"):
                out = model.text_model(**inputs)
                features = out[1] if len(out) > 1 else out[0][:, 0]
                if hasattr(model, "text_projection"):
                    features = model.text_projection(features)
            else:
                out = model(**inputs)
                features = out.text_embeds if hasattr(out, "text_embeds") else out[0]

            # Safely unwrap ModelOutput/BaseModelOutputWithPooling to torch.Tensor
            features = extract_embedding_tensor(features)
            if features.dim() == 1:
                features = features.unsqueeze(0)

            # Normalize embeddings per template
            features = F.normalize(features, p=2, dim=-1)

            # Average across templates (prompt ensembling)
            mean_embedding = features.mean(dim=0, keepdim=True)

            # Re-normalize average
            mean_embedding = F.normalize(mean_embedding, p=2, dim=-1)
            ensembled_embeddings.append(mean_embedding.cpu())

            if verbose and (i + 1) % 200 == 0:
                print(f"  - Processed {i + 1}/{len(classes)} classes...")

    text_features = torch.cat(ensembled_embeddings, dim=0)  # (num_classes, embed_dim)

    # Extract logit scale & bias
    if hasattr(model, "logit_scale"):
        scale_val = model.logit_scale.exp().detach().cpu().item()
    else:
        scale_val = 100.0

    bias_val = None
    if hasattr(model, "logit_bias") and model.logit_bias is not None:
        bias_val = model.logit_bias.detach().cpu().item()

    if verbose:
        print(f"[TextCache] Done. Features shape: {tuple(text_features.shape)}, Scale: {scale_val:.2f}, Bias: {bias_val}")

    return text_features, scale_val, bias_val


def get_or_build_evaluator(
    model: nn.Module,
    processor: Any,
    model_type: str = "clip",
    cache_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
    force_rebuild: bool = False,
    templates: Optional[List[str]] = None,
    classes: Optional[List[str]] = None,
    verbose: bool = True,
) -> ImageNet1KZeroShotEvaluator:
    """Get or build the 1,000-class zero-shot evaluator with disk caching.

    Parameters
    ----------
    model : nn.Module
        Loaded vision-language model (CLIP or SigLIP).
    processor : Any
        Loaded processor/tokenizer.
    model_type : str
        'clip' or 'siglip'.
    cache_dir : str, optional
        Directory where cached embeddings are stored.
    device : torch.device, optional
        Target device for evaluator.
    force_rebuild : bool
        If True, ignores existing cache and recomputes text features.
    templates : List[str], optional
        Prompt templates for ensembling.
    classes : List[str], optional
        Target class list (defaults to IMAGENET_1K_CLASSES).
    verbose : bool
        Whether to print status messages.

    Returns
    -------
    ImageNet1KZeroShotEvaluator
        Ready-to-use evaluator loaded in memory.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    target_device = device or (
        next(model.parameters()).device if (model is not None and hasattr(model, "parameters") and list(model.parameters())) else torch.device("cpu")
    )

    classes = classes or IMAGENET_1K_CLASSES
    model_key = "siglip" if "siglip" in model_type.lower() else "clip"
    cache_path = os.path.join(cache_dir, f"{model_key}_imagenet1k_text_features.pt")

    if not force_rebuild and os.path.isfile(cache_path):
        try:
            if verbose:
                print(f"[TextCache] Loading cached text embeddings from: {cache_path}")
            data = torch.load(cache_path, map_location=target_device, weights_only=False)
            text_features = data["text_features"]
            logit_scale = data["logit_scale"]
            logit_bias = data.get("logit_bias", None)
            cached_classes = data.get("class_names", classes)

            evaluator = ImageNet1KZeroShotEvaluator(
                model_type=model_key,
                text_features=text_features,
                class_names=cached_classes,
                logit_scale=logit_scale,
                logit_bias=logit_bias,
                device=target_device,
            )
            if verbose:
                print(f"  [PASS] Cached embeddings loaded ({text_features.shape[0]} classes, {text_features.shape[1]}-dim)")
            return evaluator
        except Exception as e:
            if verbose:
                print(f"[TextCache] WARNING: Failed to load cache ({e}). Rebuilding...")

    # Build features
    text_features, scale_val, bias_val = build_imagenet1k_text_features(
        model=model,
        processor=processor,
        model_type=model_key,
        templates=templates,
        classes=classes,
        device=target_device,
        verbose=verbose,
    )

    # Save to disk
    os.makedirs(cache_dir, exist_ok=True)
    payload = {
        "text_features": text_features,
        "logit_scale": scale_val,
        "logit_bias": bias_val,
        "class_names": classes,
        "templates": templates or CANONICAL_ENSEMBLE_TEMPLATES,
    }
    torch.save(payload, cache_path)
    if verbose:
        print(f"[TextCache] Cached {model_key.upper()} text embeddings to: {cache_path}")

    return ImageNet1KZeroShotEvaluator(
        model_type=model_key,
        text_features=text_features,
        class_names=classes,
        logit_scale=scale_val,
        logit_bias=bias_val,
        device=target_device,
    )
