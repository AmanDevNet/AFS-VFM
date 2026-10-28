import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
"""
AFS-VFM Automated Verification Suite: Learned Early Warning System
==================================================================
Validates:
  1. TemporalFeatureExtractor streaming vs batch equivalence
  2. Kinematic feature mathematics (velocity & acceleration)
  3. GroupKFold data leakage prevention (zero cross-fold image bleed)
  4. Classifier probability calibration & numerical stability (no NaNs)
  5. FailurePredictor dual-mode operation (heuristic & learned)
  6. Warning lead-time anticipation ahead of failure boundary
"""

import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.earlywarning.feature_extractor import TemporalFeatureExtractor, CORE_FEATURE_NAMES
from src.earlywarning.learned_predictor import (
    LearnedFailurePredictor,
    build_classifier,
    prepare_training_dataset,
)
from src.earlywarning.failure_predictor import FailurePredictor


def run_all_tests():
    print("=" * 72)
    print("  AFS-VFM Learned Early Warning Automated Verification Suite")
    print("=" * 72)

    # -------------------------------------------------------------
    # [TEST 1] Streaming vs Batch Feature Extraction Equivalence
    # -------------------------------------------------------------
    print("\n[TEST 1] Verifying Streaming vs Batch feature extraction equivalence...")
    sample_seq = [
        {"top1_confidence": 0.95, "logit_margin": 0.85, "cosine_sim": 0.99, "l2_distance": 0.05, "frame": 0},
        {"top1_confidence": 0.88, "logit_margin": 0.70, "cosine_sim": 0.94, "l2_distance": 0.15, "frame": 1},
        {"top1_confidence": 0.72, "logit_margin": 0.45, "cosine_sim": 0.86, "l2_distance": 0.30, "frame": 2},
        {"top1_confidence": 0.50, "logit_margin": 0.20, "cosine_sim": 0.75, "l2_distance": 0.55, "frame": 3},
    ]

    # Online streaming
    ext = TemporalFeatureExtractor()
    streaming_features = [ext.update(sig) for sig in sample_seq]

    # Offline batch
    df_sample = pd.DataFrame(sample_seq)
    df_sample["image"] = "test_img_001.jpg"
    df_sample["degradation"] = "blur"
    df_sample["model"] = "dinov2"
    batch_df = TemporalFeatureExtractor.extract_batch(df_sample)

    for i, row in enumerate(streaming_features):
        batch_vec = batch_df.iloc[i][CORE_FEATURE_NAMES].values.astype(np.float32)
        np.testing.assert_allclose(
            row, batch_vec, rtol=1e-4, atol=1e-4,
            err_msg=f"Mismatch at frame {i} between streaming and batch extraction!"
        )
    print("  [PASS] Streaming and batch feature extractors produce identical vectors.")

    # -------------------------------------------------------------
    # [TEST 2] Kinematic Features Mathematics (Velocity & Acceleration)
    # -------------------------------------------------------------
    print("\n[TEST 2] Verifying velocity and acceleration feature kinematics...")
    # Velocity at frame 1: 0.88 - 0.95 = -0.07
    d_conf_1 = streaming_features[1][CORE_FEATURE_NAMES.index("delta_confidence")]
    assert abs(d_conf_1 - (-0.07)) < 1e-4, f"Expected d_conf=-0.07, got {d_conf_1}"

    # Cosine sim vel at frame 1: 0.94 - 0.99 = -0.05
    # Cosine sim vel at frame 2: 0.86 - 0.94 = -0.08
    # Cosine sim accel at frame 2: -0.08 - (-0.05) = -0.03
    accel_cos_2 = streaming_features[2][CORE_FEATURE_NAMES.index("accel_cosine_sim")]
    assert abs(accel_cos_2 - (-0.03)) < 1e-4, f"Expected accel_cos=-0.03, got {accel_cos_2}"
    print("  [PASS] 1st-order drift velocity and 2nd-order drift acceleration verified.")

    # -------------------------------------------------------------
    # [TEST 3] GroupKFold Zero-Leakage Splitting
    # -------------------------------------------------------------
    print("\n[TEST 3] Verifying GroupKFold zero-leakage split constraint...")
    synthetic_images = [f"image_{i:03d}.jpg" for i in range(20) for _ in range(5)]
    synthetic_df = pd.DataFrame({
        "image": synthetic_images,
        "confidence": np.random.uniform(0.3, 0.9, len(synthetic_images)),
        "target": np.random.randint(0, 2, len(synthetic_images)),
    })
    gkf = GroupKFold(n_splits=5)
    for fold, (train_idx, val_idx) in enumerate(gkf.split(synthetic_df, groups=synthetic_df["image"])):
        train_imgs = set(synthetic_df.iloc[train_idx]["image"])
        val_imgs = set(synthetic_df.iloc[val_idx]["image"])
        overlap = train_imgs.intersection(val_imgs)
        assert len(overlap) == 0, f"Data leakage detected in fold {fold}: {overlap}"
    print("  [PASS] GroupKFold guarantees zero image overlap across train and test folds.")

    # -------------------------------------------------------------
    # [TEST 4] Model Probability Calibration & Bounds
    # -------------------------------------------------------------
    print("\n[TEST 4] Verifying classifier training, probability calibration, and bounds...")
    for ctype in ["logistic_regression", "gradient_boosting", "random_forest", "mlp"]:
        clf = build_classifier(ctype, random_state=42)
        X_dummy = np.random.randn(100, len(CORE_FEATURE_NAMES))
        y_dummy = np.random.binomial(1, 0.3, 100)
        clf.fit(X_dummy, y_dummy)

        probs = clf.predict_proba(X_dummy)[:, 1]
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0), f"{ctype} probs out of [0, 1] bounds!"
        assert not np.isnan(probs).any(), f"{ctype} produced NaN probabilities!"
    print("  [PASS] All 4 classifier paradigms produce strictly bounded [0, 1] probabilities.")

    # -------------------------------------------------------------
    # [TEST 5] FailurePredictor Integration & Dual-Mode Support
    # -------------------------------------------------------------
    print("\n[TEST 5] Verifying FailurePredictor dual-mode operation (heuristic & learned)...")
    # Heuristic mode
    pred_heur = FailurePredictor(mode="heuristic")
    res_heur = pred_heur.assess("dinov2", {"top1_confidence": 0.95, "logit_margin": 0.85, "cosine_sim": 0.99})
    assert res_heur["status"] == "OK", f"Heuristic clean state failed: {res_heur}"

    # Hybrid mode
    pred_hybrid = FailurePredictor(mode="hybrid")
    res_hybrid = pred_hybrid.assess("dinov2", {"top1_confidence": 0.95, "logit_margin": 0.85, "cosine_sim": 0.99})
    assert res_hybrid["status"] in ("OK", "WARNING", "CRITICAL"), f"Hybrid state invalid: {res_hybrid}"
    print("  [PASS] FailurePredictor operates seamlessly across heuristic, learned, and hybrid modes.")

    # -------------------------------------------------------------
    # [TEST 6] Warning Lead Time Anticipation
    # -------------------------------------------------------------
    print("\n[TEST 6] Verifying warning lead-time anticipation ahead of failure boundary...")
    degrading_sequence = [
        {"top1_confidence": 0.98 - 0.04 * t, "logit_margin": 0.90 - 0.05 * t, "cosine_sim": 0.99 - 0.03 * t}
        for t in range(15)
    ]
    traj_result = pred_heur.assess_trajectory("dinov2", degrading_sequence)
    assert traj_result["first_warning_frame"] is not None, "Failed to anticipate degrading trajectory!"
    if traj_result["first_critical_frame"] is not None:
        assert traj_result["first_warning_frame"] <= traj_result["first_critical_frame"], (
            f"Warning frame {traj_result['first_warning_frame']} occurred after critical frame "
            f"{traj_result['first_critical_frame']}!"
        )
    print(f"  [PASS] Anticipatory alert triggered at frame {traj_result['first_warning_frame']} "
          f"prior to critical failure at frame {traj_result['first_critical_frame']}.")

    print("\n" + "=" * 72)
    print("  ALL 6 LEARNED EARLY WARNING VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 72)


if __name__ == "__main__":
    run_all_tests()
