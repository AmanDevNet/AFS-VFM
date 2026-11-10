"""
Smoke test for the AFS-VFM degradation engine.

Creates a synthetic 256×256 test image and runs every degradation type,
verifying the output shape and dtype.
"""

import sys
import os

# Ensure the 'src' directory is on the path so we can import the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
from degradation import generate_degradation_sequence, DegradationPipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_H, IMG_W = 256, 256
NUM_FRAMES = 20
TEST_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "_test_image.png")

DEGRADATION_TYPES = ["blur", "lighting", "occlusion", "scale", "viewpoint"]


def create_test_image() -> None:
    """Write a deterministic 256×256 colour-gradient PNG to disk."""
    import cv2

    # Simple gradient: R increases left→right, G increases top→bottom
    r = np.tile(np.linspace(0, 255, IMG_W, dtype=np.uint8), (IMG_H, 1))
    g = np.tile(
        np.linspace(0, 255, IMG_H, dtype=np.uint8).reshape(-1, 1), (1, IMG_W)
    )
    b = np.full((IMG_H, IMG_W), 128, dtype=np.uint8)
    rgb = np.stack([r, g, b], axis=-1)

    # cv2 expects BGR
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(TEST_IMAGE_PATH, bgr)


def run_tests() -> bool:
    """Run all smoke tests.  Returns True if every test passes."""
    create_test_image()
    all_passed = True

    for deg_type in DEGRADATION_TYPES:
        try:
            frames = generate_degradation_sequence(TEST_IMAGE_PATH, deg_type)
            expected_shape = (NUM_FRAMES, IMG_H, IMG_W, 3)

            shape_ok = frames.shape == expected_shape
            dtype_ok = frames.dtype == np.uint8

            if shape_ok and dtype_ok:
                print(f"  [PASS]  {deg_type:<12}  shape={frames.shape}  dtype={frames.dtype}")
            else:
                print(
                    f"  [FAIL]  {deg_type:<12}  "
                    f"shape={frames.shape} (expected {expected_shape})  "
                    f"dtype={frames.dtype} (expected uint8)"
                )
                all_passed = False
        except Exception as exc:
            print(f"  [FAIL]  {deg_type:<12}  ERROR: {exc}")
            all_passed = False

    # --- Pipeline class direct usage test ---
    print("\n  --- DegradationPipeline direct usage ---")
    try:
        from degradation.utils import load_image

        img = load_image(TEST_IMAGE_PATH)
        pipeline = DegradationPipeline(num_frames=NUM_FRAMES)
        seq = pipeline.generate_sequence(img, "blur")
        assert seq.shape == (NUM_FRAMES, IMG_H, IMG_W, 3)
        assert seq.dtype == np.uint8
        print(f"  [PASS]  DegradationPipeline.generate_sequence  shape={seq.shape}")
    except Exception as exc:
        print(f"  [FAIL]  DegradationPipeline.generate_sequence  ERROR: {exc}")
        all_passed = False

    # Cleanup
    if os.path.exists(TEST_IMAGE_PATH):
        os.remove(TEST_IMAGE_PATH)

    return all_passed


if __name__ == "__main__":
    print("=" * 60)
    print("  AFS-VFM Degradation Engine — Smoke Test")
    print("=" * 60)
    passed = run_tests()
    print("=" * 60)
    if passed:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 60)
    sys.exit(0 if passed else 1)
