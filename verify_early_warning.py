"""
AFS-VFM Automated Verification Suite — Early Warning System
============================================================
Comprehensive test suite verifying:
1. Threshold configuration loading & completeness for all 5 foundation models.
2. Signal evaluation: OK, WARNING, CRITICAL status transitions.
3. Probability estimation & triggering signal attribution.
4. Trajectory assessment & anticipatory lead-time computation.
5. Synthetic trajectory simulation & graceful fallback across all models.
6. Multi-model consistency and numerical bounds validation.

Usage:
    python verify_early_warning.py
"""

import os
import sys
import json
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from src.earlywarning.failure_predictor import FailurePredictor
from src.earlywarning.demo import build_simulated_trajectory


def test_1_threshold_configuration():
    print("[TEST 1] Verifying thresholds.json configuration for all 5 models...")
    thr_path = os.path.join(ROOT_DIR, "src", "earlywarning", "thresholds.json")
    assert os.path.exists(thr_path), f"Thresholds file missing: {thr_path}"

    with open(thr_path, "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    expected_models = ["dinov2", "clip", "detr", "siglip", "dinov2_large"]
    for model in expected_models:
        assert model in thresholds, f"Missing model '{model}' in thresholds.json"
        cfg = thresholds[model]
        assert len(cfg) > 0, f"No signals configured for {model}"
        for sig_name, sig_cfg in cfg.items():
            assert "warning_threshold" in sig_cfg, f"Signal {sig_name} missing 'warning_threshold' in {model}"
            assert "critical_threshold" in sig_cfg, f"Signal {sig_name} missing 'critical_threshold' in {model}"
            assert sig_cfg["warning_threshold"] >= sig_cfg["critical_threshold"], (
                f"Inverted thresholds for {model}.{sig_name}: warning={sig_cfg['warning_threshold']} < critical={sig_cfg['critical_threshold']}"
            )

    print("  [PASS] All 5 models configured with valid warning & critical bounds.")


def test_2_clean_inference_assessment():
    print("[TEST 2] Verifying clean state assessment across all models...")
    predictor = FailurePredictor()

    clean_signals = {
        "dinov2": {"top1_confidence": 0.92, "logit_margin": 2.50, "cosine_sim": 0.98},
        "clip": {"top1_confidence": 0.88, "logit_margin": 2.10, "cosine_sim": 0.96},
        "detr": {"top_detection_confidence": 0.995, "cosine_sim": 0.95},
        "siglip": {"top1_confidence": 0.90, "logit_margin": 2.30, "cosine_sim": 0.97},
        "dinov2_large": {"top1_confidence": 0.94, "logit_margin": 2.80, "cosine_sim": 0.98},
    }

    for model, signals in clean_signals.items():
        res = predictor.assess(model, signals)
        assert res["status"] == "OK", f"Expected OK for clean {model}, got {res['status']}"
        assert len(res["triggering_signals"]) == 0, f"Clean {model} triggered unexpected signals: {res['triggering_signals']}"
        assert res["failure_probability"] < 0.65, f"Clean {model} has high failure probability: {res['failure_probability']}"

    print("  [PASS] All 5 models correctly identify clean healthy states as 'OK'.")


def test_3_warning_and_critical_transitions():
    print("[TEST 3] Verifying WARNING and CRITICAL status transitions...")
    predictor = FailurePredictor()

    # Degraded test cases
    test_cases = [
        # (model, signals, expected_status, expected_trigger)
        ("dinov2", {"top1_confidence": 0.50, "logit_margin": 1.50, "cosine_sim": 0.95}, "WARNING", "top1_confidence"),
        ("dinov2", {"top1_confidence": 0.15, "logit_margin": 0.05, "cosine_sim": 0.40}, "CRITICAL", "top1_confidence"),
        ("clip", {"top1_confidence": 0.28, "logit_margin": 1.20, "cosine_sim": 0.95}, "WARNING", "top1_confidence"),
        ("detr", {"top_detection_confidence": 0.95, "cosine_sim": 0.90}, "WARNING", "top_detection_confidence"),
        ("detr", {"top_detection_confidence": 0.50, "cosine_sim": 0.45}, "CRITICAL", "top_detection_confidence"),
        ("siglip", {"top1_confidence": 0.30, "logit_margin": 1.20, "cosine_sim": 0.90}, "WARNING", "top1_confidence"),
        ("dinov2_large", {"top1_confidence": 0.20, "logit_margin": 0.10, "cosine_sim": 0.45}, "CRITICAL", "top1_confidence"),
    ]

    for model, signals, exp_status, exp_trig in test_cases:
        res = predictor.assess(model, signals)
        assert res["status"] == exp_status, f"{model} expected status {exp_status}, got {res['status']}"
        assert exp_trig in res["triggering_signals"], f"{model} expected trigger {exp_trig}, got {res['triggering_signals']}"

    print("  [PASS] Status transitions and triggering attribution validated.")


def test_4_trajectory_lead_time():
    print("[TEST 4] Verifying trajectory lead-time anticipation before failure...")
    predictor = FailurePredictor()

    for model in ["dinov2", "clip", "siglip", "dinov2_large"]:
        traj, actual_failure = build_simulated_trajectory(model, n_frames=20)
        res = predictor.assess_trajectory(model, traj)
        assert "first_warning_frame" in res
        fw = res["first_warning_frame"]
        assert fw is not None, f"Model {model} failed to produce any warning in trajectory"
        assert fw <= actual_failure, f"Model {model} produced warning ({fw}) after actual failure ({actual_failure})"

    print("  [PASS] Anticipatory warnings precede failure across foundation models.")


def test_5_synthetic_fallback_and_bounds():
    print("[TEST 5] Verifying synthetic trajectory generator and value bounds...")
    for model in ["dinov2", "clip", "detr", "siglip", "dinov2_large"]:
        traj, failure_frame = build_simulated_trajectory(model, n_frames=25)
        assert len(traj) == 25, f"Expected 25 frames, got {len(traj)} for {model}"
        assert 1 <= failure_frame <= 25, f"Failure frame {failure_frame} out of bounds"

        for idx, frame_dict in enumerate(traj):
            for k, v in frame_dict.items():
                assert not np.isnan(v), f"NaN encountered in frame {idx} of {model}: {k}={v}"
                if "confidence" in k or "cosine" in k:
                    assert 0.0 <= v <= 1.0, f"Value {k}={v} out of unit interval [0, 1] in {model}"
                elif "margin" in k:
                    assert v >= 0.0, f"Logit margin {k}={v} cannot be negative in {model}"

    print("  [PASS] Synthetic trajectories are strictly bounded and NaN-free.")


def test_6_abrupt_critical_transition():
    print("[TEST 6] Verifying sudden CRITICAL transition correctly sets first_warning_frame...")
    predictor = FailurePredictor()

    # Trajectory that starts healthy (frames 1-3), then drops directly to CRITICAL at frame 4
    abrupt_traj = [
        {"top1_confidence": 0.95, "logit_margin": 2.50, "cosine_sim": 0.98}, # Frame 1: OK
        {"top1_confidence": 0.93, "logit_margin": 2.40, "cosine_sim": 0.97}, # Frame 2: OK
        {"top1_confidence": 0.91, "logit_margin": 2.30, "cosine_sim": 0.96}, # Frame 3: OK
        {"top1_confidence": 0.10, "logit_margin": 0.02, "cosine_sim": 0.35}, # Frame 4: CRITICAL (abrupt shock)
        {"top1_confidence": 0.08, "logit_margin": 0.01, "cosine_sim": 0.30}, # Frame 5: CRITICAL
    ]

    res = predictor.assess_trajectory("dinov2", abrupt_traj)
    assert res["trajectory"][2]["status"] == "OK"
    assert res["trajectory"][3]["status"] == "CRITICAL"
    assert res["first_critical_frame"] == 4, f"Expected first_critical_frame=4, got {res['first_critical_frame']}"
    assert res["first_warning_frame"] == 4, f"Expected first_warning_frame=4 on abrupt drop, got {res['first_warning_frame']}"

    print("  [PASS] Abrupt transition to CRITICAL reliably triggers early warning alert frame.")


def run_all_tests():
    print("=" * 70)
    print("  AFS-VFM Early Warning Automated Verification Suite")
    print("=" * 70)
    test_1_threshold_configuration()
    test_2_clean_inference_assessment()
    test_3_warning_and_critical_transitions()
    test_4_trajectory_lead_time()
    test_5_synthetic_fallback_and_bounds()
    test_6_abrupt_critical_transition()
    print("=" * 70)
    print("  ALL 6 EARLY WARNING VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
