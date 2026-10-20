"""
Dataset Downloader — AFS-VFM Research Project
===============================================

Downloads the two standard benchmark datasets required for Phase 1:
  1. COCO val2017      (~500 images sampled) — direct from cocodataset.org
  2. ImageNet-1k val   (~1000 images sampled) — via HuggingFace datasets

Usage:
    python download_dataset.py
"""

import os
import sys
import random
import shutil
import zipfile
import urllib.request

import argparse

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
COCO_DIR = os.path.join(DATA_DIR, "coco_val")
IMAGENET_DIR = os.path.join(DATA_DIR, "imagenet_val")

DEFAULT_COCO_COUNT = 5000
DEFAULT_IMAGENET_COUNT = 5000

random.seed(42)  # Reproducible sampling


# ── COCO val2017 (direct download, no auth) ─────────────────────────────────

COCO_ZIP_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ZIP_PATH = os.path.join(DATA_DIR, "val2017.zip")


def download_coco(sample_count: int = DEFAULT_COCO_COUNT, out_dir: str = COCO_DIR):
    """Download COCO val2017, extract, and sample images."""
    if os.path.isdir(out_dir) and len(os.listdir(out_dir)) >= sample_count:
        print(f"  [SKIP] COCO already has {len(os.listdir(out_dir))} images in {out_dir}")
        return True

    os.makedirs(DATA_DIR, exist_ok=True)

    # Download the zip
    if not os.path.exists(COCO_ZIP_PATH):
        print(f"  Downloading COCO val2017 (~1 GB) ...")
        print(f"  URL: {COCO_ZIP_URL}")
        print(f"  This may take a few minutes depending on your internet speed.")
        try:
            req = urllib.request.Request(COCO_ZIP_URL, headers={
                "User-Agent": "AFS-VFM-Research/1.0"
            })
            with urllib.request.urlopen(req) as response:
                total = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                with open(COCO_ZIP_PATH, "wb") as f:
                    while True:
                        chunk = response.read(1024 * 1024)  # 1MB chunks
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = int(downloaded / total * 100)
                            mb = downloaded / (1024 * 1024)
                            print(f"\r  Progress: {mb:.0f} MB ({pct}%)", end="", flush=True)
            print()  # newline after progress
        except Exception as exc:
            print(f"\n  FAILED to download COCO: {exc}")
            return False
    else:
        print(f"  [SKIP] COCO zip already downloaded")

    # Extract the zip
    print(f"  Extracting COCO val2017 ...")
    extract_dir = os.path.join(DATA_DIR, "val2017_full")
    try:
        with zipfile.ZipFile(COCO_ZIP_PATH, "r") as zf:
            zf.extractall(DATA_DIR)
        # The zip extracts to DATA_DIR/val2017/
        extracted = os.path.join(DATA_DIR, "val2017")
        if os.path.isdir(extracted):
            os.rename(extracted, extract_dir)
    except Exception as exc:
        print(f"  FAILED to extract: {exc}")
        return False

    # Sample images
    print(f"  Sampling {sample_count} images ...")
    all_images = [f for f in os.listdir(extract_dir) if f.endswith(".jpg")]
    sampled = random.sample(all_images, min(sample_count, len(all_images)))

    os.makedirs(out_dir, exist_ok=True)
    for img in sampled:
        shutil.copy2(os.path.join(extract_dir, img), os.path.join(out_dir, img))

    print(f"  COCO: {len(sampled)} images saved to {out_dir}")

    # Clean up the full extraction and zip to save disk space
    shutil.rmtree(extract_dir, ignore_errors=True)
    if os.path.exists(COCO_ZIP_PATH):
        os.remove(COCO_ZIP_PATH)
        print(f"  Cleaned up zip and temp files to save space")

    return True


# ── ImageNet-1k validation (via HuggingFace datasets) ───────────────────────

def download_imagenet(sample_count: int = DEFAULT_IMAGENET_COUNT, out_dir: str = IMAGENET_DIR, per_class_limit: int = 5):
    """Download ImageNet-1k validation split and sample images with stratification."""
    if os.path.isdir(out_dir) and len(os.listdir(out_dir)) >= sample_count:
        print(f"  [SKIP] ImageNet already has {len(os.listdir(out_dir))} images in {out_dir}")
        return True

    try:
        from datasets import load_dataset
    except ImportError:
        print("  ERROR: 'datasets' library not installed.")
        print("  Run: pip install datasets huggingface_hub")
        return False

    print(f"  Loading ImageNet-1k validation split from HuggingFace ...")

    try:
        dataset = load_dataset(
            "ILSVRC/imagenet-1k",
            split="validation",
            streaming=True,
        )
    except Exception as exc:
        error_msg = str(exc)
        if "gated" in error_msg.lower() or "access" in error_msg.lower() or "401" in error_msg:
            print(f"\n  ACCESS DENIED — You need to accept the ImageNet license first:")
            print(f"  1. Go to: https://huggingface.co/datasets/ILSVRC/imagenet-1k")
            print(f"  2. Click 'Agree and access repository'")
            print(f"  3. Then run: huggingface-cli login")
            print(f"  4. Re-run this script")
        else:
            print(f"\n  FAILED to load ImageNet: {exc}")
        return False

    os.makedirs(out_dir, exist_ok=True)
    print(f"  Streaming {sample_count} images (stratified sampling) ...")

    count = 0
    class_counts = {}
    target_per_class = max(1, sample_count // 1000)

    for sample in dataset:
        if count >= sample_count:
            break

        image = sample["image"]  # PIL Image
        label = sample["label"]

        # Stratified balance across 1,000 classes
        curr_class_count = class_counts.get(label, 0)
        if curr_class_count >= target_per_class and len(class_counts) < 1000:
            continue

        class_counts[label] = curr_class_count + 1
        count += 1

        # Convert grayscale to RGB to avoid OpenCV shape errors later
        if image.mode != "RGB":
            image = image.convert("RGB")

        filename = f"imagenet_{count:06d}_class{label}.jpg"
        filepath = os.path.join(out_dir, filename)
        image.save(filepath, "JPEG", quality=95)

        if count % 100 == 0:
            print(f"    Saved {count}/{sample_count} ...")

    print(f"  ImageNet: {count} images saved to {out_dir} across {len(class_counts)} classes")
    return True


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AFS-VFM Dataset Downloader (Scalable Curation)")
    parser.add_argument("--coco_count", type=int, default=DEFAULT_COCO_COUNT, help="Number of COCO images to download (default: 5000)")
    parser.add_argument("--imagenet_count", type=int, default=DEFAULT_IMAGENET_COUNT, help="Number of ImageNet images to stream (default: 5000)")
    parser.add_argument("--out_dir", type=str, default=DATA_DIR, help="Base data directory (default: data/)")
    args = parser.parse_args()

    coco_out = os.path.join(args.out_dir, "coco_val")
    inet_out = os.path.join(args.out_dir, "imagenet_val")

    print("=" * 60)
    print("  AFS-VFM Scalable Dataset Downloader")
    print(f"  Target: {args.coco_count} COCO + {args.imagenet_count} ImageNet = {args.coco_count + args.imagenet_count} Total")
    print("=" * 60)

    print("\n--- COCO val2017 ---")
    coco_ok = download_coco(sample_count=args.coco_count, out_dir=coco_out)

    print("\n--- ImageNet-1k validation ---")
    imagenet_ok = download_imagenet(sample_count=args.imagenet_count, out_dir=inet_out)

    print("\n" + "=" * 60)
    print(f"  COCO:     {'OK' if coco_ok else 'FAILED'}")
    print(f"  ImageNet: {'OK' if imagenet_ok else 'FAILED'}")

    if coco_ok and imagenet_ok:
        coco_count = len(os.listdir(coco_out)) if os.path.exists(coco_out) else 0
        inet_count = len(os.listdir(inet_out)) if os.path.exists(inet_out) else 0
        print(f"\n  Total base images: {coco_count + inet_count}")
        print(f"    COCO:     {coco_count} images in {coco_out}")
        print(f"    ImageNet: {inet_count} images in {inet_out}")

    print("=" * 60)


if __name__ == "__main__":
    main()
