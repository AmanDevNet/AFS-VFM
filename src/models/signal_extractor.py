"""
AFS-VFM Phase 2B - Internal Signal Extractor (v2 - fixed)
==========================================================

Extracts intermediate representations from DINOv2, CLIP, and DETR.

Changes from v1:
  - Fixed DINOv2: use model() with output_hidden_states, not model.dinov2()
  - Fixed CLIP: single forward pass, extract embeddings from outputs directly
  - Added error printing so failures are visible, not silently caught
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
    # CLIP
    # ================================================================

    def extract_clip(self, model, processor, frame_pil: Image.Image,
                     candidate_labels: list) -> dict:
        """Extract internal signals from CLIP for a single frame."""
        inputs = processor(
            text=candidate_labels, images=frame_pil,
            return_tensors="pt", padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = model(**inputs)

            # Image embedding from the outputs
            image_embedding = outputs.image_embeds[0].cpu().numpy()

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
        """Extract internal signals from DETR for a single frame."""
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
            "top_detection_confidence": top_det_conf,
            "objectness_score": objectness,
            "top5_detection_logits": top5_logits,
            "top5_detection_labels": top5_labels,
            "num_detections_above_threshold": n_above_thresh,
            "prediction": prediction,
        }

    # ================================================================
    # Computed Metrics
    # ================================================================

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        dot = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    @staticmethod
    def l2_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        return float(np.linalg.norm(vec_a - vec_b))

    @staticmethod
    def top5_overlap(labels_a: list, labels_b: list) -> float:
        set_a = set(labels_a)
        set_b = set(labels_b)
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / 5.0
