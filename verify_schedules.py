"""
Verification suite for Limitation 2 Fix: Non-Linear Severity Schedulers.

Tests:
1. Mathematical properties, monotonicity, and boundary compliance of all schedules.
2. Edge cases (single-frame sequence, invalid schedules, boundary conditions).
3. End-to-end integration with DegradationPipeline across all 5 degradation types.
"""

import sys
import os
import numpy as np
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from degradation import DegradationPipeline, SeverityScheduler, generate_degradation_sequence


def main():
    print("=" * 65)
    print("  VERIFYING LIMITATION 2 FIX: Non-Linear Severity Schedulers")
    print("=" * 65)

    n = 20

    # ────────────────────────────────────────────────────────────
    # Test 1: Linear Schedule
    # ------------------------------------------------------------
    # Test 1: Linear Schedule
    # ------------------------------------------------------------
    print("\n[Test 1] Linear Schedule:")
    lin = SeverityScheduler.get_schedule(n, schedule="linear")
    print(f"  Shape: {lin.shape}, Range: [{lin.min():.4f}, {lin.max():.4f}]")
    assert lin[0] == 0.0, f"Linear start should be 0.0, got {lin[0]}"
    assert lin[-1] == 1.0, f"Linear end should be 1.0, got {lin[-1]}"
    assert np.all(np.diff(lin) >= 0), "Linear schedule must be monotonically non-decreasing"
    print("  [PASS] Linear schedule verified")

    # ------------------------------------------------------------
    # Test 2: Exponential Schedule (Accelerating Failure)
    # ------------------------------------------------------------
    print("\n[Test 2] Exponential Schedule (Accelerating curve):")
    exp = SeverityScheduler.get_schedule(n, schedule="exponential", gamma=3.0)
    print(f"  Shape: {exp.shape}, Range: [{exp.min():.4f}, {exp.max():.4f}]")
    assert exp[0] == 0.0, f"Exp start should be 0.0, got {exp[0]}"
    assert exp[-1] == 1.0, f"Exp end should be 1.0, got {exp[-1]}"
    assert np.all(np.diff(exp) >= 0), "Exp schedule must be monotonically non-decreasing"
    # Exponential with gamma > 0 stays strictly below linear in the interior
    mid_idx = n // 2
    assert exp[mid_idx] < lin[mid_idx], f"Exp at mid ({exp[mid_idx]:.4f}) should be < Lin ({lin[mid_idx]:.4f})"
    print(f"  Midpoint comparison: Exp={exp[mid_idx]:.4f} vs Lin={lin[mid_idx]:.4f} (Confirms convex acceleration)")
    print("  [PASS] Exponential schedule verified")

    # ------------------------------------------------------------
    # Test 3: Logarithmic Schedule (Rapid Early Onset)
    # ------------------------------------------------------------
    print("\n[Test 3] Logarithmic Schedule (Rapid early onset):")
    log_curve = SeverityScheduler.get_schedule(n, schedule="logarithmic", beta=10.0)
    print(f"  Shape: {log_curve.shape}, Range: [{log_curve.min():.4f}, {log_curve.max():.4f}]")
    assert log_curve[0] == 0.0, f"Log start should be 0.0, got {log_curve[0]}"
    assert log_curve[-1] == 1.0, f"Log end should be 1.0, got {log_curve[-1]}"
    assert np.all(np.diff(log_curve) >= 0), "Log schedule must be monotonically non-decreasing"
    # Logarithmic stays strictly above linear in the interior
    assert log_curve[mid_idx] > lin[mid_idx], f"Log at mid ({log_curve[mid_idx]:.4f}) should be > Lin ({lin[mid_idx]:.4f})"
    print(f"  Midpoint comparison: Log={log_curve[mid_idx]:.4f} vs Lin={lin[mid_idx]:.4f} (Confirms concave early onset)")
    print("  [PASS] Logarithmic schedule verified")

    # ------------------------------------------------------------
    # Test 4: Step Shock Schedule (Sudden Catastrophic Occlusion)
    # ------------------------------------------------------------
    print("\n[Test 4] Step Shock Schedule (Instantaneous disruption):")
    shock_frame = 10
    step = SeverityScheduler.get_schedule(n, shock_frame=shock_frame, pre_shock=0.0, post_shock=0.85, schedule="step_shock")
    print(f"  Shape: {step.shape}, Pre-shock: {step[shock_frame-1]:.2f}, Post-shock: {step[shock_frame]:.2f}")
    assert np.all(step[:shock_frame] == 0.0), "Pre-shock values must be 0.0"
    assert np.all(step[shock_frame:] == 0.85), "Post-shock values must be 0.85"
    print("  [PASS] Step Shock schedule verified")

    # ------------------------------------------------------------
    # Test 5: Oscillatory Schedule (Periodic Atmospheric Fluctuations)
    # ------------------------------------------------------------
    print("\n[Test 5] Oscillatory Schedule:")
    osc = SeverityScheduler.get_schedule(n, schedule="oscillatory", amplitude=0.25, frequency=2.0)
    print(f"  Shape: {osc.shape}, Range: [{osc.min():.4f}, {osc.max():.4f}]")
    assert osc.min() >= 0.0, "Oscillatory values must be bounded >= 0.0"
    assert osc.max() <= 1.0, "Oscillatory values must be bounded <= 1.0"
    # An oscillatory signal has non-zero second differences and reversals
    diffs = np.diff(osc)
    has_pos = np.any(diffs > 0)
    has_neg = np.any(diffs < 0)
    assert has_pos and has_neg, "Oscillatory curve must contain both positive and negative fluctuations"
    print("  [PASS] Oscillatory schedule verified")

    # ------------------------------------------------------------
    # Test 6: Edge Cases & Boundary Conditions
    # ------------------------------------------------------------
    print("\n[Test 6] Edge cases & error handling:")
    single = SeverityScheduler.get_schedule(1, schedule="exponential")
    assert len(single) == 1 and single[0] == 0.0, "Single frame edge case failed"

    try:
        SeverityScheduler.get_schedule(0)
        assert False, "Should raise ValueError for num_frames < 1"
    except ValueError:
        pass

    try:
        SeverityScheduler.get_schedule(10, schedule="nonexistent_schedule")
        assert False, "Should raise ValueError for unknown schedule"
    except ValueError:
        pass
    print("  [PASS] Edge cases and input validations verified")

    # ------------------------------------------------------------
    # Test 7: Integration with DegradationPipeline
    # ------------------------------------------------------------
    print("\n[Test 7] DegradationPipeline end-to-end integration:")
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_img[20:80, 20:80] = [200, 100, 50]  # Foreground patch

    deg_types = ["blur", "occlusion", "lighting", "scale", "viewpoint"]
    schedules = ["linear", "exponential", "step_shock", "oscillatory"]

    pipeline = DegradationPipeline(num_frames=10)

    for deg in deg_types:
        for sched in schedules:
            frames = pipeline.generate_sequence(dummy_img, deg, schedule=sched)
            assert frames.shape == (10, 100, 100, 3), f"Output shape mismatch for {deg}/{sched}: {frames.shape}"
            assert frames.dtype == np.uint8, f"Output dtype mismatch for {deg}/{sched}: {frames.dtype}"
            assert pipeline.last_severities is not None and len(pipeline.last_severities) == 10
    print(f"  [PASS] Successfully verified {len(deg_types) * len(schedules)} combinations (5 degradations x 4 schedules)")

    print("\n" + "=" * 65)
    print("  ALL VERIFICATION CHECKS PASSED: Limitation 2 is completely resolved!")
    print("=" * 65)


if __name__ == "__main__":
    main()
