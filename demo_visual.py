"""
Visual Demo — AFS-VFM Degradation Engine
==========================================

Generates labeled grid visualizations and progression sequences across
environmental, sensor, semantic, and compound degradation modes.

Usage:
    python demo_visual.py
    python demo_visual.py --degradations blur gaussian_noise fog_haze night_motion
    python demo_visual.py --image path/to/image.jpg --out_dir results/custom_demo
"""

import os
import sys
import argparse
import urllib.request

import cv2
import numpy as np

# Ensure the 'src' directory is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from degradation import generate_degradation_sequence

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
DEFAULT_OUTPUT_DIR = os.path.join(DATA_DIR, "demo_output")
SAMPLE_IMAGE_PATH = os.path.join(DATA_DIR, "sample_dog.jpg")

SAMPLE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/"
    "b/bd/Golden_Retriever_Dukedestination.jpg/"
    "640px-Golden_Retriever_Dukedestination.jpg"
)

CANONICAL_DEGRADATIONS = [
    "blur", "occlusion", "lighting", "scale", "viewpoint",
    "gaussian_noise", "fog_haze", "contrast",
]

COMPOUND_PRESETS = [
    "night_motion", "sensor_stress", "adverse_weather", "turbulent_flight",
]

GRID_COLS = 5
GRID_ROWS = 2
FRAME_INDICES = [0, 2, 4, 6, 8, 10, 12, 14, 17, 19]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_sample_image() -> str:
    """Download a sample image or create a synthetic fallback."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(SAMPLE_IMAGE_PATH):
        print(f"  Using existing sample image: {SAMPLE_IMAGE_PATH}")
        return SAMPLE_IMAGE_PATH

    print("  Attempting sample image download ...")
    try:
        req = urllib.request.Request(SAMPLE_URL, headers={"User-Agent": "AFS-VFM-Demo/1.0"})
        with urllib.request.urlopen(req, timeout=5) as response:
            with open(SAMPLE_IMAGE_PATH, "wb") as f:
                f.write(response.read())
        print(f"  Saved to: {SAMPLE_IMAGE_PATH}")
    except Exception as exc:
        print(f"  Download unavailable ({exc}). Generating high-contrast synthetic image ...")
        create_synthetic_image(SAMPLE_IMAGE_PATH)

    return SAMPLE_IMAGE_PATH


def create_synthetic_image(path: str) -> None:
    """Create a colorful 512x512 synthetic image as a reproducible fallback."""
    h, w = 512, 512
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Smooth color gradients
    for y in range(h):
        for x in range(w):
            img[y, x] = [
                int(255 * x / w),
                int(255 * y / h),
                int(128 + 64 * np.sin(x * 0.05)),
            ]

    # Geometric structures
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.circle(bgr, (256, 256), 100, (0, 255, 255), -1)
    cv2.rectangle(bgr, (60, 60), (200, 200), (255, 100, 50), -1)
    cv2.putText(bgr, "AFS-VFM", (130, 410), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
    cv2.imwrite(path, bgr)
    print(f"  Synthetic image generated: {path}")


def add_label(image: np.ndarray, text: str) -> np.ndarray:
    """Add a labeled badge to the top-left of an image frame."""
    result = image.copy()
    bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr, (0, 0), (160, 32), (0, 0, 0), -1)
    cv2.putText(bgr, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_grid(frames: np.ndarray) -> np.ndarray:
    """Arrange key sequence frames into a 2x5 labeled grid."""
    num_f = len(frames)
    indices = [int(i * (num_f - 1) / 9) for i in range(10)]
    selected = [frames[i] for i in indices]

    labelled = []
    for idx, frame in zip(indices, selected):
        severity_pct = int(idx / (num_f - 1) * 100) if num_f > 1 else 0
        labelled.append(add_label(frame, f"F{idx+1:02d} ({severity_pct:3d}%)"))

    rows = []
    for r in range(GRID_ROWS):
        start = r * GRID_COLS
        row_frames = labelled[start:start + GRID_COLS]
        rows.append(np.hstack(row_frames))

    return np.vstack(rows)


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

def run_visual_demo(
    image_path: str = None,
    degradations: list = None,
    out_dir: str = None,
    num_frames: int = 20,
):
    print("=" * 65)
    print("  AFS-VFM Degradation Engine - Visual Demonstration")
    print("=" * 65)

    img_path = image_path or download_sample_image()
    output_dir = out_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    deg_list = degradations or (CANONICAL_DEGRADATIONS + ["night_motion", "sensor_stress"])
    print(f"  Source Image:   {img_path}")
    print(f"  Output Dir:     {output_dir}")
    print(f"  Degradations ({len(deg_list)}): {', '.join(deg_list)}\n")

    for deg_type in deg_list:
        print(f"  Generating sequence for '{deg_type}' ...", end=" ")
        try:
            frames = generate_degradation_sequence(img_path, deg_type, num_frames=num_frames)
            grid = build_grid(frames)

            # Save grid
            out_grid = os.path.join(output_dir, f"demo_{deg_type}.jpg")
            cv2.imwrite(out_grid, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Save clean baseline and extreme corrupted frame
            first_path = os.path.join(output_dir, f"{deg_type}_f01_clean.jpg")
            last_path = os.path.join(output_dir, f"{deg_type}_f{num_frames:02d}_degraded.jpg")
            cv2.imwrite(first_path, cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
            cv2.imwrite(last_path, cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR))

            print(f"[OK] -> {os.path.basename(out_grid)}")
        except Exception as exc:
            print(f"[FAIL]: {exc}")

    print("\n" + "=" * 65)
    print(f"  All demo grids successfully saved to: {output_dir}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AFS-VFM Degradation Engine Visual Demonstration")
    parser.add_argument("--image", type=str, default=None, help="Path to input test image")
    parser.add_argument("--degradations", nargs="+", default=None, help="List of degradations or compound recipes")
    parser.add_argument("--out_dir", type=str, default=None, help="Target directory for output grids")
    parser.add_argument("--num_frames", type=int, default=20, help="Number of frames in sequence")
    args = parser.parse_args()

    run_visual_demo(
        image_path=args.image,
        degradations=args.degradations,
        out_dir=args.out_dir,
        num_frames=args.num_frames,
    )
