"""
Verification script for Limitation 7 Fix: DETR Feature Drift and Cosine Similarity NaN.
Tests:
1. Extraction of encoder-pooled global embedding from DETR.
2. Extraction of top-confidence query embedding.
3. Numerical stability of cosine similarity under normal, degraded, edge-case, and zero-norm inputs.
"""

import sys
import os
import numpy as np
from PIL import Image, ImageFilter
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from models.model_loader import VisionModelLoader
from models.signal_extractor import SignalExtractor


def main():
    print("=" * 60)
    print("  VERIFYING LIMITATION 7 FIX: DETR Feature Drift & Cosine Sim")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Test numerical stability of cosine_similarity directly
    print("\n[Test 1] Numerical stability edge cases:")
    extractor = SignalExtractor(device=device)

    # Identical vectors
    v1 = np.random.randn(256).astype(np.float32)
    sim_self = extractor.cosine_similarity(v1, v1)
    print(f"  Self-similarity: {sim_self:.6f} (Expected: 1.000000)")
    assert abs(sim_self - 1.0) < 1e-5, f"Self similarity failed: {sim_self}"

    # Zero norm vectors
    v_zero = np.zeros(256, dtype=np.float32)
    sim_zero = extractor.cosine_similarity(v_zero, v1)
    print(f"  Zero-vector similarity: {sim_zero} (Expected: 0.0, No NaN)")
    assert sim_zero == 0.0 and not np.isnan(sim_zero), "Zero vector produced NaN!"

    # Both zero vectors
    sim_both_zero = extractor.cosine_similarity(v_zero, v_zero)
    print(f"  Both zero similarity: {sim_both_zero} (Expected: 0.0, No NaN)")
    assert sim_both_zero == 0.0 and not np.isnan(sim_both_zero), "Both zero produced NaN!"

    # NaN input vector
    v_nan = np.full(256, np.nan, dtype=np.float32)
    sim_nan = extractor.cosine_similarity(v_nan, v1)
    print(f"  NaN-vector input: {sim_nan} (Expected: 0.0, No NaN crash)")
    assert sim_nan == 0.0 and not np.isnan(sim_nan), "NaN vector produced NaN!"

    # 2. Test live DETR model extraction and feature drift
    print("\n[Test 2] Live DETR extraction & drift tracking:")
    loader = VisionModelLoader()
    detr_model, detr_proc = loader.load_detr()

    # Generate synthetic clean test image with patterns
    from PIL import ImageDraw
    clean_img = Image.new("RGB", (224, 224), color=(240, 240, 240))
    draw = ImageDraw.Draw(clean_img)
    draw.rectangle([20, 20, 80, 80], fill=(220, 30, 30))
    draw.ellipse([100, 100, 180, 180], fill=(30, 120, 220))
    draw.polygon([(50, 120), (120, 200), (30, 180)], fill=(30, 200, 60))

    # Degraded image (heavy blur)
    degraded_img = clean_img.filter(ImageFilter.GaussianBlur(radius=25))

    clean_sig = extractor.extract_detr(detr_model, detr_proc, clean_img)
    degraded_sig = extractor.extract_detr(detr_model, detr_proc, degraded_img)

    print("\nDETR Signal Output Keys:", list(clean_sig.keys()))
    assert "image_embedding" in clean_sig, "image_embedding missing from DETR signals!"
    assert "query_embedding" in clean_sig, "query_embedding missing from DETR signals!"

    emb_clean = clean_sig["image_embedding"]
    emb_deg = degraded_sig["image_embedding"]

    print(f"  Clean embedding shape: {emb_clean.shape}, dtype: {emb_clean.dtype}")
    print(f"  Any NaN in clean embedding: {np.isnan(emb_clean).any()}")
    print(f"  Any NaN in degraded embedding: {np.isnan(emb_deg).any()}")

    assert not np.isnan(emb_clean).any(), "Clean embedding contains NaN!"
    assert not np.isnan(emb_deg).any(), "Degraded embedding contains NaN!"

    # Compute cosine drift
    cos_self = extractor.cosine_similarity(emb_clean, emb_clean)
    cos_drift = extractor.cosine_similarity(emb_clean, emb_deg)
    l2_dist = extractor.l2_distance(emb_clean, emb_deg)

    print(f"  DETR Baseline Cosine Similarity (Frame 0 to Frame 0): {cos_self:.6f}")
    print(f"  DETR Degraded Cosine Similarity (Frame 0 to Degraded): {cos_drift:.6f}")
    print(f"  DETR Feature L2 Distance: {l2_dist:.6f}")

    assert abs(cos_self - 1.0) < 1e-5, f"Baseline cosine similarity should be 1.0, got {cos_self}"
    assert not np.isnan(cos_drift), "Degraded cosine similarity is NaN!"
    assert cos_drift < 0.999, f"Feature drift not detected: {cos_drift}"
    assert l2_dist > 0.0, f"L2 distance should be positive, got {l2_dist}"

    print("\n" + "=" * 60)
    print("  ALL VERIFICATION CHECKS PASSED: Limitation 7 is completely resolved!")
    print("=" * 60)


if __name__ == "__main__":
    main()
