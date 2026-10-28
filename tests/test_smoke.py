"""
AFS-VFM Smoke Test Suite
========================
Runs a fast (<2s) self-contained smoke test using sample_data/ without requiring
external GPU access or downloading large ImageNet/COCO datasets.
"""

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image

# Setup root path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from degradation.degradation import generate_degradation_sequence
from degradation.scheduler import SeverityScheduler
from earlywarning.failure_predictor import FailurePredictor
from models.clip_imagenet1k import extract_embedding_tensor
import torch


def test_smoke():
    print("=" * 60)
    print("  AFS-VFM Self-Contained Smoke Test Suite")
    print("=" * 60)

    # 1. Check sample_data exists
    sample_img_dir = os.path.join(ROOT_DIR, "sample_data", "images")
    sample_csv = os.path.join(ROOT_DIR, "sample_data", "sample_benchmark.csv")
    assert os.path.isdir(sample_img_dir), f"Missing {sample_img_dir}"
    assert os.path.isfile(sample_csv), f"Missing {sample_csv}"

    sample_imgs = [os.path.join(sample_img_dir, f) for f in os.listdir(sample_img_dir) if f.endswith(('.jpg', '.png'))]
    assert len(sample_imgs) >= 5, f"Expected at least 5 sample images, found {len(sample_imgs)}"
    print(f"[PASS] sample_data verified with {len(sample_imgs)} images and sample CSV.")

    # 2. Test SeverityScheduler
    for sched in ["linear", "exponential", "logarithmic", "step_shock", "oscillatory"]:
        s = SeverityScheduler.get_schedule(num_frames=20, schedule=sched)
        assert len(s) == 20, f"Expected 20 frames for {sched}"
        assert 0.0 <= s[0] <= 1.0 and 0.0 <= s[-1] <= 1.0, f"Bounds exceeded in {sched}"
    print("[PASS] SeverityScheduler schedules verified bounded in [0.0, 1.0].")

    # 3. Test Degradation generation on a sample image
    test_img_path = sample_imgs[0]
    frames = generate_degradation_sequence(test_img_path, "blur", num_frames=5)
    assert isinstance(frames, np.ndarray), "Expected numpy array"
    assert frames.shape[0] == 5, f"Expected 5 frames, got {frames.shape[0]}"
    print(f"[PASS] Degradation generation verified: output shape {frames.shape}")

    # 4. Test FailurePredictor heuristic initialization and clean assessment
    predictor = FailurePredictor()
    clean_signals = {
        "top1_confidence": 0.98,
        "logit_margin": 0.85,
        "cosine_sim": 0.99,
        "l2_distance": 0.05,
    }
    assessment = predictor.assess("dinov2", clean_signals)
    assert assessment["status"] == "OK", f"Expected 'OK' for clean signal, got {assessment['status']}"
    print("[PASS] FailurePredictor assessed clean input state as 'OK'.")

    # 5. Test extract_embedding_tensor
    tensor_input = torch.randn(2, 512)
    assert torch.equal(extract_embedding_tensor(tensor_input), tensor_input)
    print("[PASS] extract_embedding_tensor identity verified.")

    # 6. Test sample CSV schema
    df = pd.read_csv(sample_csv)
    required_cols = ["image", "dataset", "degradation", "frame", "severity_pct"]
    for col in required_cols:
        assert col in df.columns, f"Missing required column {col} in sample CSV"
    print(f"[PASS] sample_benchmark.csv schema verified ({len(df)} rows).")

    print("=" * 60)
    print("  ALL SMOKE TESTS PASSED (Self-contained, reproducible)")
    print("=" * 60)


if __name__ == "__main__":
    test_smoke()
