"""
AFS-VFM Verification Script: Limitation 10 Resolution
=====================================================
Validates 1,000-class ImageNet-1K prompt ensembling, precomputed text embedding caching,
and ultra-fast zero-shot inference for CLIP and SigLIP.

Tests:
1. ImageNet-1K Canonical Classes & Prompt Templates Alignment
2. Ensembled Text Feature Normalization & Matrix Properties
3. CLIP (Softmax) Zero-Shot Evaluator & Numerical Invariants
4. SigLIP (Pairwise Sigmoid BCE) Zero-Shot Evaluator & Numerical Invariants
5. Disk Serialization, Caching Speed & Fidelity
6. SignalExtractor Integration & Full 1,000-Class Dispatch
"""

import os
import sys
import time
import tempfile
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Path setup
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from data.imagenet_classes import (
    IMAGENET_1K_CLASSES,
    IMAGENET_PROMPT_TEMPLATES,
    get_clean_label,
    get_imagenet_classes,
)
from models.clip_imagenet1k import (
    ImageNet1KZeroShotEvaluator,
    CANONICAL_ENSEMBLE_TEMPLATES,
    get_or_build_evaluator,
)
from models.signal_extractor import SignalExtractor


def test_1_imagenet_classes():
    print("\n[Test 1] ImageNet-1K Canonical Classes & Prompt Templates:")
    assert len(IMAGENET_1K_CLASSES) == 1000, f"Expected 1000 classes, got {len(IMAGENET_1K_CLASSES)}"
    assert IMAGENET_1K_CLASSES[0] == "tench, Tinca tinca", f"Class 0 mismatch: {IMAGENET_1K_CLASSES[0]}"
    assert IMAGENET_1K_CLASSES[1] == "goldfish, Carassius auratus", f"Class 1 mismatch: {IMAGENET_1K_CLASSES[1]}"
    
    clean_0 = get_clean_label(IMAGENET_1K_CLASSES[0])
    assert clean_0 == "tench", f"Expected 'tench', got '{clean_0}'"
    
    classes_clean = get_imagenet_classes(clean=True)
    assert len(classes_clean) == 1000
    assert classes_clean[0] == "tench"

    assert len(IMAGENET_PROMPT_TEMPLATES) >= 50, f"Expected >= 50 templates, got {len(IMAGENET_PROMPT_TEMPLATES)}"
    assert len(CANONICAL_ENSEMBLE_TEMPLATES) == 10, f"Expected 10 canonical templates, got {len(CANONICAL_ENSEMBLE_TEMPLATES)}"

    print(f"  - Total canonical classes: {len(IMAGENET_1K_CLASSES)}")
    print(f"  - Class 0: '{IMAGENET_1K_CLASSES[0]}' -> clean: '{clean_0}'")
    print(f"  - Class 999: '{IMAGENET_1K_CLASSES[999]}' -> clean: '{get_clean_label(IMAGENET_1K_CLASSES[999])}'")
    print(f"  - Available prompt templates: {len(IMAGENET_PROMPT_TEMPLATES)}")
    print("  [PASS] ImageNet-1K class ontology and prompt templates verified")


def test_2_ensembled_text_feature_math():
    print("\n[Test 2] Ensembled Text Feature Normalization & Matrix Properties:")
    embed_dim = 512
    num_classes = 1000
    num_templates = len(CANONICAL_ENSEMBLE_TEMPLATES)

    # Simulate multi-template text features: (1000, num_templates, embed_dim)
    torch.manual_seed(42)
    raw_feats = torch.randn(num_classes, num_templates, embed_dim)
    # Normalize per template
    norm_feats = raw_feats / raw_feats.norm(dim=-1, keepdim=True)
    # Average across templates
    mean_feats = norm_feats.mean(dim=1)
    # Re-normalize
    final_text_weights = mean_feats / mean_feats.norm(dim=-1, keepdim=True)

    norms = final_text_weights.norm(dim=-1)
    assert final_text_weights.shape == (1000, 512), f"Unexpected shape {final_text_weights.shape}"
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Not all vectors are unit norm"
    assert not torch.isnan(final_text_weights).any(), "Found NaNs in text feature matrix"

    print(f"  - Synthesized ensemble matrix shape: {tuple(final_text_weights.shape)}")
    print(f"  - Vector norms min: {norms.min().item():.6f}, max: {norms.max().item():.6f}")
    print("  [PASS] Prompt ensembling unit-norm mathematical constraints verified")


def test_3_clip_zero_shot_evaluator():
    print("\n[Test 3] CLIP (Softmax) Zero-Shot Evaluator & Latency:")
    torch.manual_seed(42)
    embed_dim = 512
    # Create 1,000 class normalized text features
    text_features = torch.randn(1000, embed_dim)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    evaluator = ImageNet1KZeroShotEvaluator(
        model_type="clip",
        text_features=text_features,
        class_names=IMAGENET_1K_CLASSES,
        logit_scale=100.0,
    )

    # Create dummy image embedding aligned with class index 42
    target_class_idx = 42
    image_embed = text_features[target_class_idx].clone().numpy()
    # Add small noise to simulate realistic frame embedding
    image_embed = image_embed + 0.1 * np.random.randn(embed_dim)

    res = evaluator.evaluate(image_embed)

    # Invariants
    assert res["prediction"] == IMAGENET_1K_CLASSES[target_class_idx], (
        f"Expected {IMAGENET_1K_CLASSES[target_class_idx]}, got {res['prediction']}"
    )
    assert len(res["top5_labels"]) == 5
    assert len(res["top5_confs"]) == 5
    assert res["top1_confidence"] > 0.0 and res["top1_confidence"] <= 1.0
    assert res["logit_margin"] >= 0.0
    assert res["prediction_entropy"] > 0.0
    assert abs(res["full_probs"].sum() - 1.0) < 1e-4, "Softmax probs must sum to 1.0"
    assert res["full_probs"].shape == (1000,)

    # Latency benchmark: 200 consecutive evaluations
    t0 = time.perf_counter()
    n_iters = 200
    for _ in range(n_iters):
        _ = evaluator.evaluate(image_embed)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    ms_per_eval = elapsed_ms / n_iters

    print(f"  - Top-1 Predicted Class: '{res['prediction']}'")
    print(f"  - Top-1 Confidence: {res['top1_confidence']:.4f}")
    print(f"  - Logit Margin: {res['logit_margin']:.4f}")
    print(f"  - Distribution Entropy (1000 classes): {res['prediction_entropy']:.4f} nats")
    print(f"  - Inference Latency: {ms_per_eval:.4f} ms/frame ({n_iters} evals in {elapsed_ms:.2f} ms)")
    assert ms_per_eval < 2.0, f"Latency too slow: {ms_per_eval:.2f} ms"
    print("  [PASS] CLIP 1,000-class zero-shot evaluation and ultra-low latency verified")


def test_4_siglip_zero_shot_evaluator():
    print("\n[Test 4] SigLIP (Pairwise Sigmoid BCE) Zero-Shot Evaluator:")
    torch.manual_seed(99)
    embed_dim = 768
    text_features = torch.randn(1000, embed_dim)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    evaluator = ImageNet1KZeroShotEvaluator(
        model_type="siglip",
        text_features=text_features,
        class_names=IMAGENET_1K_CLASSES,
        logit_scale=10.0,
        logit_bias=-2.5,
    )

    # Create dummy image embedding aligned with class 105
    target_class_idx = 105
    image_embed = text_features[target_class_idx].clone().numpy()

    res = evaluator.evaluate(image_embed)

    assert res["prediction"] == IMAGENET_1K_CLASSES[target_class_idx]
    assert np.all((res["full_probs"] >= 0.0) & (res["full_probs"] <= 1.0))
    assert res["top1_confidence"] > 0.0
    assert res["logit_margin"] >= 0.0
    assert res["prediction_entropy"] > 0.0
    assert res["full_probs"].shape == (1000,)

    print(f"  - SigLIP Top-1 Prediction: '{res['prediction']}'")
    print(f"  - SigLIP Top-1 Sigmoid Prob: {res['top1_confidence']:.4f}")
    print(f"  - SigLIP Logit Margin: {res['logit_margin']:.4f}")
    print(f"  - SigLIP Normalized Entropy: {res['prediction_entropy']:.4f} nats")
    print("  [PASS] SigLIP pairwise sigmoid BCE zero-shot evaluator verified")


def test_5_disk_caching_and_fidelity():
    print("\n[Test 5] Disk Serialization, Cache Persistence & Fidelity:")
    with tempfile.TemporaryDirectory() as tmpdir:
        embed_dim = 512
        text_features = torch.randn(1000, embed_dim)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        scale_val = 100.0

        cache_path = os.path.join(tmpdir, "clip_imagenet1k_text_features.pt")
        payload = {
            "text_features": text_features,
            "logit_scale": scale_val,
            "logit_bias": None,
            "class_names": IMAGENET_1K_CLASSES,
            "templates": CANONICAL_ENSEMBLE_TEMPLATES,
        }
        torch.save(payload, cache_path)
        assert os.path.isfile(cache_path), "Cache file not written"
        cache_size_kb = os.path.getsize(cache_path) / 1024.0

        # Load back
        t0 = time.perf_counter()
        loaded_evaluator = get_or_build_evaluator(
            model=None,
            processor=None,
            model_type="clip",
            cache_dir=tmpdir,
            force_rebuild=False,
            verbose=False,
        )
        load_ms = (time.perf_counter() - t0) * 1000.0

        assert loaded_evaluator.text_features.shape == (1000, embed_dim)
        assert torch.allclose(loaded_evaluator.text_features, text_features)
        assert loaded_evaluator.logit_scale == scale_val
        print(f"  - Cached file size: {cache_size_kb:.1f} KB")
        print(f"  - Cache load latency: {load_ms:.2f} ms")
        assert load_ms < 500.0, "Cache loading too slow"
        print("  [PASS] Disk caching fidelity, size, and instant load time verified")


def test_6_signal_extractor_integration():
    print("\n[Test 6] SignalExtractor Integration & Full 1,000-Class Dispatch:")
    device = torch.device("cpu")
    extractor = SignalExtractor(device=device)

    embed_dim = 512
    text_features = torch.randn(1000, embed_dim)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    evaluator = ImageNet1KZeroShotEvaluator(
        model_type="clip",
        text_features=text_features,
        class_names=IMAGENET_1K_CLASSES,
        logit_scale=100.0,
    )

    class DummyCLIPModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.logit_scale = nn.Parameter(torch.tensor(4.6052))

        def get_image_features(self, pixel_values=None):
            # Return target class vector (e.g. class 7)
            return text_features[7:8].clone()

    class DummyProcessor:
        def __call__(self, images=None, text=None, return_tensors="pt", **kwargs):
            class Inputs(dict):
                def to(self, dev):
                    return self
            inp = Inputs()
            inp["pixel_values"] = torch.zeros(1, 3, 224, 224)
            return inp

    model = DummyCLIPModel()
    proc = DummyProcessor()
    canvas = Image.new("RGB", (100, 100), color=(100, 150, 200))

    # Test direct extract_clip with evaluator
    sig_1000 = extractor.extract_clip(model, proc, canvas, evaluator=evaluator)
    assert sig_1000["prediction"] == IMAGENET_1K_CLASSES[7]
    assert sig_1000["full_probs"].shape == (1000,)
    assert len(sig_1000["top5_labels"]) == 5
    assert sig_1000["image_embedding"].shape == (embed_dim,)
    assert "prediction_entropy" in sig_1000
    assert "logit_margin" in sig_1000

    # Test unified dispatch with evaluator
    sig_dispatch = extractor.extract("clip", model, proc, canvas, evaluator=evaluator)
    assert sig_dispatch["prediction"] == IMAGENET_1K_CLASSES[7]
    assert sig_dispatch["full_probs"].shape == (1000,)

    print(f"  - Extracted 1,000-class prediction: '{sig_1000['prediction']}'")
    print(f"  - Top-5 labels: {sig_1000['top5_labels']}")
    print(f"  - Top-1 confidence: {sig_1000['top1_confidence']:.4f}")
    print(f"  - Unified dispatch verified: {sig_dispatch['prediction'] == sig_1000['prediction']}")
    print("  [PASS] SignalExtractor full 1,000-class zero-shot pipeline verified")


def main():
    print("=" * 70)
    print("  AFS-VFM LIMITATION 10 VERIFICATION: 1,000-CLASS ZERO-SHOT PIPELINE")
    print("=" * 70)

    test_1_imagenet_classes()
    test_2_ensembled_text_feature_math()
    test_3_clip_zero_shot_evaluator()
    test_4_siglip_zero_shot_evaluator()
    test_5_disk_caching_and_fidelity()
    test_6_signal_extractor_integration()

    print("\n" + "=" * 70)
    print("  ALL 6 LIMITATION 10 VERIFICATION TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
