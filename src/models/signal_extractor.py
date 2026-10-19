"""
AFS-VFM Phase 2B - Internal Signal Extractor (v2 - fixed)
==========================================================

Extracts intermediate representations from DINOv2, CLIP, and DETR.

Changes from v1:
  - Fixed DINOv2: use model() with output_hidden_states, not model.dinov2()
  - Fixed CLIP: single forward pass, extract embeddings from outputs directly
  - Added error printing so failures are visible, not silently caught
"""

from typing import Optional, Union, Any, List
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class SignalExtractor:
    """Extracts internal signals from vision foundation models."""

    def __init__(self, device: torch.device):
        self.device = device

    # ================================================================
    # DINOv2
    # ================================================================

    def extract_dinov2(self, model, processor, frame_pil: Image.Image) -> dict:
        """Extract internal signals from DINOv2 for a single frame."""
        inputs = processor(images=frame_pil, return_tensors="pt").to(self.device)

        with torch.no_grad():
            # Single forward pass with hidden states
            outputs = model(**inputs, output_hidden_states=True)
            logits = outputs.logits[0]  # (num_classes,)

            # CLS embedding from last hidden state
            # hidden_states is a tuple of (embedding_output, layer1, layer2, ...)
            last_hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
            cls_embedding = last_hidden[:, 0, :].cpu().numpy().flatten()

        # Top-5 predictions
        probs = F.softmax(logits, dim=-1)
        top5_vals, top5_idx = probs.topk(5)
        top5_logits = top5_vals.cpu().numpy()
        top5_labels = [model.config.id2label[idx.item()] for idx in top5_idx]

        # Confidence metrics
        top1_conf = float(top5_logits[0])
        logit_margin = float(top5_logits[0] - top5_logits[1])

        # Attention entropy: skip if model doesn't support it
        avg_entropy = 0.0
        try:
            attn_outputs = model(**inputs, output_attentions=True)
            last_attn = attn_outputs.attentions[-1].cpu().numpy()[0]  # (heads, seq, seq)
            cls_attn = last_attn[:, 0, :]  # CLS row
            cls_attn_safe = np.clip(cls_attn, 1e-10, None)
            entropy_per_head = -np.sum(cls_attn_safe * np.log(cls_attn_safe), axis=-1)
            avg_entropy = float(np.mean(entropy_per_head))
        except Exception:
            # Some model configs don't support output_attentions
            avg_entropy = 0.0

        prediction = top5_labels[0]

        return {
            "cls_embedding": cls_embedding,
            "top5_logits": top5_logits,
            "top5_labels": top5_labels,
            "top1_confidence": top1_conf,
            "logit_margin": logit_margin,
            "attention_entropy": avg_entropy,
            "prediction": prediction,
        }

    # ================================================================
    # ================================================================
    # Helper: Visual Feature Extraction for Contrastive Models
    # ================================================================

    def _extract_vision_embedding(self, model, processor, frame_pil: Image.Image) -> torch.Tensor:
        """Helper to extract visual embedding tensor from CLIP or SigLIP."""
        inputs = processor(images=frame_pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            if hasattr(model, "get_image_features"):
                img_feat = model.get_image_features(**inputs)
            elif hasattr(model, "vision_model"):
                out = model.vision_model(**inputs)
                img_feat = out.pooler_output if hasattr(out, "pooler_output") else out[0][:, 0]
                if hasattr(model, "visual_projection"):
                    img_feat = model.visual_projection(img_feat)
            else:
                outputs = model(**inputs)
                if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
                    img_feat = outputs.image_embeds
                elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                    img_feat = outputs.last_hidden_state[:, 0]
                else:
                    img_feat = outputs[0]
        return img_feat

    # ================================================================
    # CLIP
    # ================================================================

    def extract_clip(self, model, processor, frame_pil: Image.Image,
                     candidate_labels: Optional[Union[list, str]] = None,
                     evaluator: Any = None) -> dict:
        """Extract internal signals from CLIP for a single frame.

        Supports two evaluation modes:
        1. 1,000-class ImageNet-1K hierarchy via precomputed text cache (Limitation 10).
           Activated when evaluator is provided, or candidate_labels is None, 'imagenet1k', or empty.
        2. Arbitrary custom candidate labels list (legacy / specialized subsets).
        """
        # Mode 1: 1,000-class ImageNet evaluation
        if evaluator is not None or candidate_labels in (None, "imagenet1k", "1000", []):
            if evaluator is None:
                from .clip_imagenet1k import get_or_build_evaluator
                evaluator = get_or_build_evaluator(
                    model=model, processor=processor, model_type="clip", device=self.device
                )
            img_feat = self._extract_vision_embedding(model, processor, frame_pil)
            result = evaluator.evaluate(img_feat)
            result["image_embedding"] = img_feat[0].detach().cpu().numpy()
            return result

        # Mode 2: Custom candidate labels
        inputs = processor(
            text=candidate_labels, images=frame_pil,
            return_tensors="pt", padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = model(**inputs)
            image_embedding = outputs.image_embeds[0].cpu().numpy()
            logits_per_image = outputs.logits_per_image[0]  # (n_labels,)

        probs = F.softmax(logits_per_image, dim=-1).cpu().numpy()
        raw_logits = logits_per_image.cpu().numpy()

        sorted_idx = np.argsort(probs)[::-1]
        top1_conf = float(probs[sorted_idx[0]])
        top2_conf = float(probs[sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0
        logit_margin = float(top1_conf - top2_conf)
        prediction = candidate_labels[sorted_idx[0]]

        p_safe = np.clip(probs, 1e-12, None)
        entropy = float(-np.sum(p_safe * np.log(p_safe)))

        top_k = min(5, len(candidate_labels))
        top5_labels = [candidate_labels[idx] for idx in sorted_idx[:top_k]]
        top5_logits = [float(probs[idx]) for idx in sorted_idx[:top_k]]

        return {
            "image_embedding": image_embedding,
            "full_logits": raw_logits,
            "full_probs": probs,
            "top1_confidence": top1_conf,
            "logit_margin": logit_margin,
            "prediction_entropy": entropy,
            "top5_labels": top5_labels,
            "top5_logits": np.array(top5_logits),
            "top5_confs": top5_logits,
            "prediction": prediction,
        }

    # ================================================================
    # SigLIP (Pairwise Sigmoid BCE Contrastive Learning)
    # ================================================================

    def extract_siglip(self, model, processor, frame_pil: Image.Image,
                       candidate_labels: Optional[Union[list, str]] = None,
                       evaluator: Any = None) -> dict:
        """Extract internal signals from Google SigLIP for a single frame.

        Supports two evaluation modes:
        1. 1,000-class ImageNet-1K hierarchy via precomputed text cache (Limitation 10).
           Activated when evaluator is provided, or candidate_labels is None, 'imagenet1k', or empty.
        2. Arbitrary custom candidate labels list (legacy / specialized subsets).
        """
        # Mode 1: 1,000-class ImageNet evaluation
        if evaluator is not None or candidate_labels in (None, "imagenet1k", "1000", []):
            if evaluator is None:
                from .clip_imagenet1k import get_or_build_evaluator
                evaluator = get_or_build_evaluator(
                    model=model, processor=processor, model_type="siglip", device=self.device
                )
            img_feat = self._extract_vision_embedding(model, processor, frame_pil)
            result = evaluator.evaluate(img_feat)
            result["image_embedding"] = img_feat[0].detach().cpu().numpy()
            return result

        # Mode 2: Custom candidate labels
        inputs = processor(
            text=candidate_labels, images=frame_pil,
            return_tensors="pt", padding="max_length"
        ).to(self.device)

        with torch.no_grad():
            outputs = model(**inputs)
            if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
                image_embedding = outputs.image_embeds[0].cpu().numpy()
            elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                image_embedding = outputs.last_hidden_state[0, 0].cpu().numpy()
            else:
                image_embedding = outputs[0][0].cpu().numpy()

            logits_per_image = outputs.logits_per_image[0]  # (n_labels,)

        probs = torch.sigmoid(logits_per_image).cpu().numpy()
        raw_logits = logits_per_image.cpu().numpy()

        sorted_idx = np.argsort(probs)[::-1]
        top1_conf = float(probs[sorted_idx[0]])
        top2_conf = float(probs[sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0
        logit_margin = float(top1_conf - top2_conf)
        prediction = candidate_labels[sorted_idx[0]]

        p_norm = probs / (np.sum(probs) + 1e-12)
        p_safe = np.clip(p_norm, 1e-12, None)
        entropy = float(-np.sum(p_safe * np.log(p_safe)))

        top_k = min(5, len(candidate_labels))
        top5_labels = [candidate_labels[idx] for idx in sorted_idx[:top_k]]
        top5_logits = [float(probs[idx]) for idx in sorted_idx[:top_k]]

        return {
            "image_embedding": image_embedding,
            "full_logits": raw_logits,
            "full_probs": probs,
            "top1_confidence": top1_conf,
            "logit_margin": logit_margin,
            "prediction_entropy": entropy,
            "top5_labels": top5_labels,
            "top5_logits": np.array(top5_logits),
            "top5_confs": top5_logits,
            "prediction": prediction,
        }

    # ================================================================
    # DETR
    # ================================================================

    def extract_detr(self, model, processor, frame_pil: Image.Image) -> dict:
        """Extract internal signals from DETR for a single frame."""
        inputs = processor(images=frame_pil, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            logits = outputs.logits[0]  # (num_queries=100, num_classes+1)

            # Global spatial representation from Transformer Encoder (Image Embedding)
            # encoder_last_hidden_state: (batch, seq_len=49, hidden_dim=256)
            if hasattr(outputs, "encoder_last_hidden_state") and outputs.encoder_last_hidden_state is not None:
                encoder_emb = outputs.encoder_last_hidden_state[0].mean(dim=0).cpu().numpy()
            elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                encoder_emb = outputs.last_hidden_state[0].mean(dim=0).cpu().numpy()
            else:
                encoder_emb = np.zeros(256, dtype=np.float32)

        # Drop the 'no-object' class (last column)
        class_logits = logits[:, :-1]
        probs = F.softmax(class_logits, dim=-1)

        # Best detection per query
        max_prob_per_query, max_class_per_query = probs.max(dim=-1)

        # Overall best detection
        best_query_idx = max_prob_per_query.argmax().item()
        top_det_conf = float(max_prob_per_query[best_query_idx].cpu())
        top_class_id = max_class_per_query[best_query_idx].item()
        prediction = model.config.id2label[top_class_id]

        # Top query embedding (Decoder object query embedding)
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            query_emb = outputs.last_hidden_state[0, best_query_idx].cpu().numpy()
        else:
            query_emb = encoder_emb

        # Objectness
        objectness = float(max_prob_per_query.max().cpu())

        # Number of detections above threshold
        n_above_thresh = int((max_prob_per_query > 0.5).sum().cpu())

        # Top-5 detections
        top5_vals, top5_query_idx = max_prob_per_query.topk(min(5, len(max_prob_per_query)))
        top5_logits = top5_vals.cpu().numpy()
        top5_labels = [model.config.id2label[max_class_per_query[qi].item()]
                       for qi in top5_query_idx]

        return {
            "image_embedding": encoder_emb,
            "encoder_embedding": encoder_emb,
            "query_embedding": query_emb,
            "top_detection_confidence": top_det_conf,
            "objectness_score": objectness,
            "top5_detection_logits": top5_logits,
            "top5_detection_labels": top5_labels,
            "num_detections_above_threshold": n_above_thresh,
            "prediction": prediction,
        }

    # ================================================================
    # Unified Extractor Dispatcher
    # ================================================================

    def extract(self, model_name: str, model, processor, frame_pil: Image.Image, **kwargs) -> dict:
        """Unified signal extraction dispatcher across all supported vision foundation models.

        Parameters
        ----------
        model_name : str
            One of: 'dinov2', 'dinov2_base', 'dinov2_large', 'clip', 'siglip', 'detr'.
        model :
            Loaded vision model in evaluation mode.
        processor :
            Corresponding feature processor/tokenizer.
        frame_pil : Image.Image
            RGB image frame.
        **kwargs :
            Additional parameters (e.g. candidate_labels for CLIP / SigLIP).
        """
        key = model_name.strip().lower().replace("-", "_")
        if "dinov2" in key:
            return self.extract_dinov2(model, processor, frame_pil)
        elif "siglip" in key:
            candidate_labels = kwargs.get("candidate_labels", None)
            evaluator = kwargs.get("evaluator", None)
            return self.extract_siglip(model, processor, frame_pil, candidate_labels=candidate_labels, evaluator=evaluator)
        elif "clip" in key:
            candidate_labels = kwargs.get("candidate_labels", None)
            evaluator = kwargs.get("evaluator", None)
            return self.extract_clip(model, processor, frame_pil, candidate_labels=candidate_labels, evaluator=evaluator)
        elif "detr" in key:
            return self.extract_detr(model, processor, frame_pil)
        else:
            raise ValueError(
                f"Unsupported model_name '{model_name}'. "
                f"Expected one of: 'dinov2', 'dinov2_large', 'clip', 'siglip', 'detr'."
            )

    # ================================================================
    # Computed Metrics
    # ================================================================

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine similarity with strict numerical stability guards against NaN."""
        if vec_a is None or vec_b is None:
            return 0.0
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0.0 or norm_b == 0.0 or np.isnan(norm_a) or np.isnan(norm_b):
            return 0.0
        dot = np.dot(vec_a, vec_b)
        if np.isnan(dot):
            return 0.0
        sim = float(dot / (norm_a * norm_b + 1e-8))
        if np.isnan(sim):
            return 0.0
        return float(np.clip(sim, -1.0, 1.0))

    @staticmethod
    def l2_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute L2 distance with numerical sanity checks."""
        if vec_a is None or vec_b is None:
            return 0.0
        diff = vec_a - vec_b
        if np.isnan(diff).any():
            return 0.0
        return float(np.linalg.norm(diff))

    @staticmethod
    def top5_overlap(labels_a: list, labels_b: list) -> float:
        set_a = set(labels_a)
        set_b = set(labels_b)
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / 5.0
