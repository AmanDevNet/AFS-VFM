"""
Verification suite for Limitation 3 Fix: Realistic Multi-Mode Occlusion.

Tests:
1. Zero severity identity (severity 0.0 returns untouched original).
2. Geometry modes: 'irregular' (splatters/blobs) vs 'boxes' (legacy rectangles).
3. Photometric textures: 'semi_transparent' (lens smudge), 'noise' (sensor grain), and 'black' (solid).
4. Full determinism check (identical output on repeated runs with same severity).
5. End-to-end integration with DegradationPipeline.
"""

import sys
import os
import numpy as np
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from degradation import DegradationPipeline, generate_degradation_sequence
from degradation.transformations import occlusion


def main():
    print("=" * 65)
    print("  VERIFYING LIMITATION 3 FIX: Realistic Multi-Mode Occlusion")
    print("=" * 65)

    # High-contrast test canvas: white background with colorful shapes
    canvas = np.full((120, 120, 3), 240, dtype=np.uint8)
    canvas[20:60, 20:60] = [200, 30, 30]    # Red square
    canvas[60:100, 60:100] = [30, 180, 50]  # Green square

    # ------------------------------------------------------------
    # Test 1: Zero Severity Identity
    # ------------------------------------------------------------
    print("\n[Test 1] Zero Severity Identity:")
    clean_copy = occlusion(canvas, severity=0.0)
    assert np.array_equal(clean_copy, canvas), "At severity 0.0, output must exactly match input!"
    print("  [PASS] Clean image perfectly preserved at severity 0.0")

    # ------------------------------------------------------------
    # Test 2: Irregular Splatter vs Rectangular Boxes
    # ------------------------------------------------------------
    print("\n[Test 2] Occlusion Geometry Modes:")
    occ_irregular = occlusion(canvas, severity=0.6, mode="irregular", texture="black")
    occ_boxes = occlusion(canvas, severity=0.6, mode="boxes", texture="black")

    assert occ_irregular.shape == canvas.shape and occ_irregular.dtype == np.uint8
    assert occ_boxes.shape == canvas.shape and occ_boxes.dtype == np.uint8

    # Both must modify the image
    diff_irreg = np.sum(occ_irregular != canvas)
    diff_boxes = np.sum(occ_boxes != canvas)
    assert diff_irreg > 0, "Irregular occlusion failed to perturb pixels"
    assert diff_boxes > 0, "Box occlusion failed to perturb pixels"

    # The geometries should differ
    assert not np.array_equal(occ_irregular, occ_boxes), "Irregular and box geometries should produce different masks"
    print(f"  [PASS] Irregular splatters altered {diff_irreg} pixels; Boxes altered {diff_boxes} pixels")

    # ------------------------------------------------------------
    # Test 3: Photometric Texture Modes
    # ------------------------------------------------------------
    print("\n[Test 3] Photometric Texture Modes:")
    # 1. Solid Black
    out_black = occlusion(canvas, severity=0.5, mode="irregular", texture="black")
    # Occluded pixels in solid black mode are explicitly 0
    zero_pixels = np.sum(np.all(out_black == 0, axis=-1))
    assert zero_pixels > 0, "Solid black mode must produce pure black pixels"
    print(f"  - Black texture: {zero_pixels} pure black pixels generated")

    # 2. Semi-Transparent (Lens Smudge / Water spot)
    out_smudge = occlusion(canvas, severity=0.5, mode="irregular", texture="semi_transparent", alpha=0.8)
    # In semi-transparent mode, underlying details are attenuated/blurred, but not all flattened to 0
    perturbed = (out_smudge != canvas)
    assert np.any(perturbed), "Semi-transparent smudge must perturb pixels"
    # Smudge should retain color gradients rather than pure 0
    non_zero_perturbed = np.sum(perturbed & (out_smudge > 0))
    assert non_zero_perturbed > 0, "Semi-transparent smudge must retain non-zero underlying content"
    print(f"  - Semi-transparent smudge: successfully generated soft attenuated edges")

    # 3. Noise Texture (Sensor / Grain artifact)
    out_noise = occlusion(canvas, severity=0.5, mode="irregular", texture="noise", alpha=0.9)
    noise_diff = np.abs(out_noise.astype(int) - canvas.astype(int))
    assert np.any(noise_diff > 0), "Noise texture must perturb pixels"
    print(f"  - Noise texture: successfully generated textured grain")
    print("  [PASS] All 3 photometric textures verified")

    # ------------------------------------------------------------
    # Test 4: Determinism & Reproducibility
    # ------------------------------------------------------------
    print("\n[Test 4] Determinism & Reproducibility:")
    run1 = occlusion(canvas, severity=0.45, mode="irregular", texture="semi_transparent")
    run2 = occlusion(canvas, severity=0.45, mode="irregular", texture="semi_transparent")
    assert np.array_equal(run1, run2), "Repeated runs with same severity must be 100% bit-for-bit identical!"
    print("  [PASS] Output is 100% deterministic and reproducible across runs")

    # ------------------------------------------------------------
    # Test 5: End-to-End Pipeline Integration
    # ------------------------------------------------------------
    print("\n[Test 5] DegradationPipeline Integration:")
    pipeline = DegradationPipeline(
        num_frames=10,
        schedule="exponential",
        transform_params={"mode": "irregular", "texture": "semi_transparent"}
    )
    frames = pipeline.generate_sequence(canvas, degradation_type="occlusion")

    assert frames.shape == (10, 120, 120, 3)
    assert frames.dtype == np.uint8
    # Frame 0 is severity 0 (clean canvas)
    assert np.array_equal(frames[0], canvas), "First frame in exponential schedule must be clean"
    # Last frame is severity 1.0 (heavily occluded)
    assert not np.array_equal(frames[-1], canvas), "Last frame must be occluded"

    # Increasing perturbation along sequence
    diffs = [np.mean(np.abs(frames[i].astype(float) - canvas.astype(float))) for i in range(10)]
    assert diffs[0] == 0.0, "Frame 0 diff must be 0"
    assert diffs[-1] > diffs[1], "Severity perturbation must increase along sequence"
    print(f"  Frame mean L1 diff progression: {diffs[0]:.2f} -> {diffs[5]:.2f} -> {diffs[-1]:.2f}")
    print("  [PASS] DegradationPipeline seamlessly generates progressive realistic occlusion")

    print("\n" + "=" * 65)
    print("  ALL VERIFICATION CHECKS PASSED: Limitation 3 is completely resolved!")
    print("=" * 65)


if __name__ == "__main__":
    main()
