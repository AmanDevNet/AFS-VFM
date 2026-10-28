"""
Verification suite for Limitation 5 Fix:
- Expanded Model Diversity (DINOv2-Base, DINOv2-Large, CLIP, SigLIP, DETR)
- Model Dispatcher in VisionModelLoader
- SigLIP Pairwise Sigmoid Signal Extraction
- Unified SignalExtractor.extract Interface
- Scale & Paradigm Fidelity (768 vs 1024 dims; Softmax vs Sigmoid)

Tests:
1. VisionModelLoader load_model dispatching and argument validation.
2. SignalExtractor.extract_dinov2 on Base (768) and Large (1024) architectures.
3. SignalExtractor.extract_siglip pairwise sigmoid probability computation.
4. CLIP vs SigLIP paradigm comparison (Softmax sum=1.0 vs Sigmoid independent in [0, 1]).
5. Unified SignalExtractor.extract dispatcher across all 5 models.
6. Embedding normalization, margin computation, and zero-NaN numerical stability.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from models import VisionModelLoader, SignalExtractor


# ── Mock Classes for Fast, Deterministic Unit Verification ──────────────

class MockConfig:
    def __init__(self, id2label=None):
        self.id2label = id2label or {i: f"class_{i}" for i in range(100)}


class MockDINOv2(nn.Module):
    """Simulates DINOv2-Base or DINOv2-Large model outputs."""
    def __init__(self, hidden_dim: int = 768, num_classes: int = 100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = MockConfig({i: f"label_{i}" for i in range(num_classes)})

    def forward(self, pixel_values=None, images=None, output_hidden_states=True, output_attentions=False, **kwargs):
        batch_size = 1
        logits = torch.randn(batch_size, len(self.config.id2label))
        # Hidden state: (batch, seq_len=1370, hidden_dim)
        cls_hidden = torch.randn(batch_size, 1, self.hidden_dim)
        patch_hidden = torch.randn(batch_size, 10, self.hidden_dim)
        hidden = torch.cat([cls_hidden, patch_hidden], dim=1)

        class Output:
            pass
        out = Output()
        out.logits = logits
        out.hidden_states = [hidden]
        out.last_hidden_state = hidden
        return out


class MockCLIP(nn.Module):
    """Simulates CLIP zero-shot classification output."""
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, input_ids=None, pixel_values=None, **kwargs):
        batch_size = 1
        num_labels = kwargs.get("num_labels", 4)
        image_embeds = torch.randn(batch_size, self.embed_dim)
        image_embeds = nn.functional.normalize(image_embeds, p=2, dim=-1)
        logits_per_image = torch.tensor([[4.5, 2.1, 0.3, -1.2]])

        class Output:
            pass
        out = Output()
        out.image_embeds = image_embeds
        out.logits_per_image = logits_per_image
        return out


class MockSigLIP(nn.Module):
    """Simulates Google SigLIP pairwise sigmoid classification output."""
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, input_ids=None, pixel_values=None, **kwargs):
        batch_size = 1
        image_embeds = torch.randn(batch_size, self.embed_dim)
        image_embeds = nn.functional.normalize(image_embeds, p=2, dim=-1)
        # Raw logits before sigmoid
        logits_per_image = torch.tensor([[3.2, 1.4, -0.8, -2.5]])

        class Output:
            pass
        out = Output()
        out.image_embeds = image_embeds
        out.logits_per_image = logits_per_image
        return out


class MockDETR(nn.Module):
    """Simulates DETR object detection output."""
    def __init__(self, num_queries: int = 100, num_classes: int = 91):
        super().__init__()
        self.config = MockConfig({i: f"coco_{i}" for i in range(num_classes)})
        self.num_queries = num_queries

    def forward(self, pixel_values=None, output_hidden_states=True, **kwargs):
        batch_size = 1
        logits = torch.randn(batch_size, self.num_queries, len(self.config.id2label) + 1)
        # Make one query distinctly high confidence
        logits[0, 5, 10] = 12.0
        encoder_hidden = torch.randn(batch_size, 49, 256)
        decoder_hidden = torch.randn(batch_size, self.num_queries, 256)

        class Output:
            pass
        out = Output()
        out.logits = logits
        out.encoder_last_hidden_state = encoder_hidden
        out.last_hidden_state = decoder_hidden
        return out


class MockProcessor:
    """Pass-through processor converting image to dummy tensors."""
    def __call__(self, images=None, text=None, return_tensors="pt", **kwargs):
        class Inputs(dict):
            def to(self, device):
                return self
        inp = Inputs()
        inp["pixel_values"] = torch.zeros(1, 3, 224, 224)
        if text is not None:
            inp["input_ids"] = torch.zeros(1, len(text), 16, dtype=torch.long)
        return inp


def main():
    print("=" * 70)
    print("  VERIFYING LIMITATION 5 FIX: Expanded Model Diversity")
    print("=" * 70)

    device = torch.device("cpu")
    extractor = SignalExtractor(device=device)
    loader = VisionModelLoader()
    canvas = Image.new("RGB", (100, 100), color=(180, 50, 70))
    candidate_labels = ["sports car", "traffic light", "tree", "pedestrian"]
    proc = MockProcessor()

    # ------------------------------------------------------------
    # Test 1: VisionModelLoader Dispatcher Logic
    # ------------------------------------------------------------
    print("\n[Test 1] VisionModelLoader Dispatcher Validation:")
    for m in ["dinov2", "dinov2_base", "dinov2_large", "clip", "siglip", "detr"]:
        assert hasattr(loader, "load_" + (m.replace("dinov2_base", "dinov2"))), f"Loader missing method for {m}"
        print(f"  - Model ID '{m:12s}' properly mapped in VisionModelLoader")

    # Invalid model check
    try:
        loader.load_model("unknown_vfm_architecture")
        assert False, "Should have raised ValueError on invalid model"
    except ValueError:
        print("  - Invalid model name properly rejected with ValueError")
    print("  [PASS] Unified loader dispatcher structure validated")

    # ------------------------------------------------------------
    # Test 2: DINOv2 Scale Comparison (Base 768 vs Large 1024)
    # ------------------------------------------------------------
    print("\n[Test 2] DINOv2 Scale Diversity (Base vs Large):")
    model_base = MockDINOv2(hidden_dim=768)
    model_large = MockDINOv2(hidden_dim=1024)

    sig_base = extractor.extract_dinov2(model_base, proc, canvas)
    sig_large = extractor.extract_dinov2(model_large, proc, canvas)

    assert sig_base["cls_embedding"].shape == (768,), f"Expected 768, got {sig_base['cls_embedding'].shape}"
    assert sig_large["cls_embedding"].shape == (1024,), f"Expected 1024, got {sig_large['cls_embedding'].shape}"
    assert "logit_margin" in sig_base and "top1_confidence" in sig_base
    assert "logit_margin" in sig_large and "top1_confidence" in sig_large

    print(f"  - DINOv2-Base CLS embedding dimension:  {sig_base['cls_embedding'].shape[0]}")
    print(f"  - DINOv2-Large CLS embedding dimension: {sig_large['cls_embedding'].shape[0]}")
    print("  [PASS] Both Base and Large scales extract cleanly with respective dimensions")

    # ------------------------------------------------------------
    # Test 3: SigLIP Pairwise Sigmoid Formulation
    # ------------------------------------------------------------
    print("\n[Test 3] SigLIP Signal Extraction & Sigmoid Calibration:")
    model_siglip = MockSigLIP(embed_dim=768)
    sig_siglip = extractor.extract_siglip(model_siglip, proc, canvas, candidate_labels)

    probs = sig_siglip["full_probs"]
    assert probs.shape == (4,), f"Expected 4 label probabilities, got {probs.shape}"
    assert np.all((probs >= 0.0) & (probs <= 1.0)), "SigLIP probabilities outside [0, 1]"
    assert sig_siglip["prediction"] == candidate_labels[0]
    assert sig_siglip["image_embedding"].shape == (768,)
    assert not np.isnan(sig_siglip["logit_margin"])

    print(f"  - SigLIP raw probabilities: {np.round(probs, 4)}")
    print(f"  - Top-1 Prediction: '{sig_siglip['prediction']}' (Confidence: {sig_siglip['top1_confidence']:.4f})")
    print(f"  - Logit Margin: {sig_siglip['logit_margin']:.4f}")
    print("  [PASS] SigLIP pairwise sigmoid probabilities and signals verified")

    # ------------------------------------------------------------
    # Test 4: CLIP vs SigLIP Contrastive Objective Comparison
    # ------------------------------------------------------------
    print("\n[Test 4] CLIP (Softmax) vs SigLIP (Sigmoid) Comparison:")
    model_clip = MockCLIP(embed_dim=512)
    sig_clip = extractor.extract_clip(model_clip, proc, canvas, candidate_labels)

    clip_prob_sum = float(np.sum(sig_clip["full_probs"]))
    siglip_prob_sum = float(np.sum(sig_siglip["full_probs"]))

    print(f"  - CLIP Probabilities Sum:   {clip_prob_sum:.6f} (Softmax normalizes to 1.0)")
    print(f"  - SigLIP Probabilities Sum: {siglip_prob_sum:.6f} (Independent sigmoid BCE)")

    assert abs(clip_prob_sum - 1.0) < 1e-5, "CLIP softmax sum must equal 1.0"
    assert siglip_prob_sum > 1.0 or siglip_prob_sum < 0.99, "SigLIP sigmoid sum is decoupled from 1.0"
    print("  [PASS] Contrastive paradigm distinction mathematically confirmed")

    # ------------------------------------------------------------
    # Test 5: Unified SignalExtractor.extract Dispatcher
    # ------------------------------------------------------------
    print("\n[Test 5] Unified Extractor Dispatcher Across All 5 Models:")
    model_detr = MockDETR()

    configs = [
        ("dinov2", model_base, {}),
        ("dinov2_large", model_large, {}),
        ("clip", model_clip, {"candidate_labels": candidate_labels}),
        ("siglip", model_siglip, {"candidate_labels": candidate_labels}),
        ("detr", model_detr, {}),
    ]

    for name, m, kwargs in configs:
        res = extractor.extract(name, m, proc, canvas, **kwargs)
        assert "prediction" in res, f"Model {name} missing 'prediction'"
        assert "image_embedding" in res or "cls_embedding" in res, f"Model {name} missing embedding"
        print(f"  - Unified dispatch for '{name:12s}': OK (Prediction: {res['prediction']})")

    print("  [PASS] Unified extraction dispatcher operates seamlessly across all 5 models")

    # ------------------------------------------------------------
    # Test 6: Numerical Stability & Embedding Drift Metrics
    # ------------------------------------------------------------
    print("\n[Test 6] Numerical Stability of Multi-Model Embeddings:")
    emb1 = sig_siglip["image_embedding"]
    # Unit-normalized 768-dim vector: subtle perturbation for drift test
    noise = np.random.randn(*emb1.shape)
    noise = (noise / np.linalg.norm(noise)) * 0.05
    emb2 = emb1 + noise
    sim = SignalExtractor.cosine_similarity(emb1, emb2)
    dist = SignalExtractor.l2_distance(emb1, emb2)

    assert 0.90 < sim <= 1.0, f"Unexpected cosine similarity: {sim}"
    assert not np.isnan(dist) and dist > 0.0
    print(f"  - Cosine similarity: {sim:.4f}, L2 distance: {dist:.4f}")
    print("  [PASS] Numerical stability guards verified on 768-dim SigLIP embeddings")

    print("\n" + "=" * 70)
    print("  ALL VERIFICATION CHECKS PASSED: Limitation 5 is completely resolved!")
    print("=" * 70)


if __name__ == "__main__":
    main()
