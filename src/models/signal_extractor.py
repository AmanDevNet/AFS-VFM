"""
AFS-VFM Phase 2B - Internal Signal Extractor
=============================================

Extracts intermediate representations from DINOv2, CLIP, and DETR
during inference on degraded images. Unlike Phase 1 (which only saved
the final prediction + boolean), this captures:

  DINOv2: CLS embedding (768-d), top-5 logits + labels, attention weights
  CLIP:   Image embedding (512-d), full logit vector (20 labels), logit margin
  DETR:   Top detection confidence, objectness score, detection logits

From these raw signals, we compute per-frame metrics:
  - Cosine similarity drift (vs frame 0 embedding)
  - Attention entropy
  - Logit margin (top-1 minus top-2 confidence)
  - Embedding L2 distance from baseline
  - Top-5 prediction stability

Usage:
    This module is imported by the Phase 2B Kaggle notebook.
    >>> from src.models.signal_extractor import SignalExtractor
    >>> extractor = SignalExtractor(device)
    >>> signals = extractor.extract_dinov2(model, processor, frames)
"""

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
        """Extract internal signals from DINOv2 for a single frame.

        Returns dict with:
            - cls_embedding: numpy array (768,)
            - top5_logits: numpy array (5,)
            - top5_labels: list of 5 label strings
            - top1_confidence: float (softmax probability of top prediction)
            - logit_margin: float (top1 - top2 softmax probability)
            - prediction: str (top-1 label)
        """
        inputs = processor(images=frame_pil, return_tensors="pt").to(self.device)

        with torch.no_grad():
            # Get the full model output including hidden states
            outputs = model.dinov2(pixel_values=inputs["pixel_values"],
                                   output_hidden_states=True,
                                   output_attentions=True)

            # CLS token embedding from last hidden state
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

            # Attention weights from last layer
            # Shape: (batch, num_heads, seq_len, seq_len)
            last_attn = outputs.attentions[-1].cpu().numpy()[0]  # (num_heads, seq_len, seq_len)

            # Run through the classifier head to get logits
            classifier_output = model(pixel_values=inputs["pixel_values"])
            logits = classifier_output.logits[0]  # (num_classes,)

        # Top-5 predictions
        probs = F.softmax(logits, dim=-1)
        top5_vals, top5_idx = probs.topk(5)
        top5_logits = top5_vals.cpu().numpy()
        top5_labels = [model.config.id2label[idx.item()] for idx in top5_idx]

        # Confidence metrics
        top1_conf = top5_logits[0]
        logit_margin = float(top5_logits[0] - top5_logits[1])

        # Attention entropy: average over all heads, CLS token row
        # CLS attends to all patches - how spread out is its attention?
        cls_attn = last_attn[:, 0, :]  # (num_heads, seq_len)
        # Add small epsilon to avoid log(0)
        cls_attn_safe = np.clip(cls_attn, 1e-10, None)
        entropy_per_head = -np.sum(cls_attn_safe * np.log(cls_attn_safe), axis=-1)
        avg_entropy = float(np.mean(entropy_per_head))

        prediction = top5_labels[0]

        return {
            "cls_embedding": cls_embedding,
            "top5_logits": top5_logits,
            "top5_labels": top5_labels,
            "top1_confidence": float(top1_conf),
            "logit_margin": logit_margin,
            "attention_entropy": avg_entropy,
            "prediction": prediction,
        }

    # ================================================================
    # CLIP
    # ================================================================

    def extract_clip(self, model, processor, frame_pil: Image.Image,
                     candidate_labels: list) -> dict:
        """Extract internal signals from CLIP for a single frame.

        Returns dict with:
            - image_embedding: numpy array (512,)
            - full_logits: numpy array (n_labels,) raw logits
            - full_probs: numpy array (n_labels,) softmax probs
            - top1_confidence: float
            - logit_margin: float (top1 - top2)
            - prediction: str
        """
        inputs = processor(
            text=candidate_labels, images=frame_pil,
            return_tensors="pt", padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = model(**inputs)

            # Image embedding (normalized)
            image_embedding = model.get_image_features(
                pixel_values=inputs["pixel_values"]
            )
            image_embedding = image_embedding.cpu().numpy().flatten()

            # Logits per image
            logits_per_image = outputs.logits_per_image[0]  # (n_labels,)

        probs = F.softmax(logits_per_image, dim=-1).cpu().numpy()
        raw_logits = logits_per_image.cpu().numpy()

        # Sort for top predictions
        sorted_idx = np.argsort(probs)[::-1]
        top1_conf = float(probs[sorted_idx[0]])
        logit_margin = float(probs[sorted_idx[0]] - probs[sorted_idx[1]])
        prediction = candidate_labels[sorted_idx[0]]

        return {
            "image_embedding": image_embedding,
            "full_logits": raw_logits,
            "full_probs": probs,
            "top1_confidence": top1_conf,
            "logit_margin": logit_margin,
            "prediction": prediction,
        }

    # ================================================================
    # DETR
    # ================================================================

    def extract_detr(self, model, processor, frame_pil: Image.Image) -> dict:
        """Extract internal signals from DETR for a single frame.

        Returns dict with:
            - top_detection_confidence: float (max softmax prob of best detection)
            - objectness_score: float (max confidence that ANY object exists)
            - top5_detection_logits: numpy array (5,) top class probabilities
            - top5_detection_labels: list of 5 label strings
            - num_detections_above_threshold: int (detections with conf > 0.5)
            - prediction: str (top detection class)
        """
        inputs = processor(images=frame_pil, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0]  # (num_queries=100, num_classes+1)

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

        # Objectness: how confident is DETR that it found anything?
        # Use the max probability across all queries
        objectness = float(max_prob_per_query.max().cpu())

        # Number of detections above threshold
        n_above_thresh = int((max_prob_per_query > 0.5).sum().cpu())

        # Top-5 detections (by confidence)
        top5_vals, top5_query_idx = max_prob_per_query.topk(min(5, len(max_prob_per_query)))
        top5_logits = top5_vals.cpu().numpy()
        top5_labels = [model.config.id2label[max_class_per_query[qi].item()]
                       for qi in top5_query_idx]

        return {
            "top_detection_confidence": top_det_conf,
            "objectness_score": objectness,
            "top5_detection_logits": top5_logits,
            "top5_detection_labels": top5_labels,
            "num_detections_above_threshold": n_above_thresh,
            "prediction": prediction,
        }

    # ================================================================
    # Computed Metrics (post-hoc, comparing frame N to frame 0)
    # ================================================================

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    @staticmethod
    def l2_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute L2 (Euclidean) distance between two vectors."""
        return float(np.linalg.norm(vec_a - vec_b))

    @staticmethod
    def top5_overlap(labels_a: list, labels_b: list) -> float:
        """Fraction of top-5 labels in common (0.0 to 1.0)."""
        set_a = set(labels_a)
        set_b = set(labels_b)
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / 5.0
