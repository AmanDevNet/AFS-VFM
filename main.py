"""
AFS-VFM Full Benchmark — main.py
==================================

Master orchestration script for Phase 1.

For each base image in data/coco_val/ and data/imagenet_val/:
    For each degradation type (blur, occlusion, lighting, scale, viewpoint):
        1. Generate a 20-frame degradation sequence  (Track A)
        2. Feed each frame through DINOv2, CLIP, DETR  (Track B)
        3. Record when each model's prediction diverges from its
           clean-image baseline  (is_failure = True)

Results are exported to  results/full_benchmark.csv

Usage:
    python main.py              # Full benchmark (1500 images)
    python main.py --pilot      # Quick pilot run (~20 images)
    python main.py --batch 1    # Run images 1-500 only
    python main.py --batch 2    # Run images 501-1000 only
    python main.py --batch 3    # Run images 1001-1500 only
"""

import os
import sys
import csv
import time
import random
import argparse

# Force unbuffered output so progress shows up in terminal immediately
os.environ["PYTHONUNBUFFERED"] = "1"

try:
    import torch
except ImportError:
    torch = None

import numpy as np
from PIL import Image

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

try:
    from degradation import generate_degradation_sequence
    from models.model_loader import VisionModelLoader
    from models.clip_imagenet1k import get_or_build_evaluator
except ImportError:
    generate_degradation_sequence = None
    VisionModelLoader = None
    get_or_build_evaluator = None

# ── Configuration ───────────────────────────────────────────────────────────
COCO_DIR = os.path.join(ROOT_DIR, "data", "coco_val")
IMAGENET_DIR = os.path.join(ROOT_DIR, "data", "imagenet_val")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

DEGRADATION_TYPES = ["blur", "occlusion", "lighting", "scale", "viewpoint"]
NUM_FRAMES = 20

# Pilot mode: grab a small sample to validate the pipeline quickly
PILOT_COCO_COUNT = 5
PILOT_IMAGENET_COUNT = 15

# CLIP zero-shot candidate labels (mapped to common ImageNet / COCO classes)
CLIP_CANDIDATE_LABELS = [
    "a dog", "a cat", "a car", "a bicycle", "an airplane",
    "a clock", "fruit", "a person", "a coffee cup", "a traffic scene",
    "a bird", "a boat", "a horse", "a chair", "a bottle",
    "a bus", "a truck", "a train", "a flower", "a tree",
]


# ── Collect image paths ────────────────────────────────────────────────────

def collect_images(pilot=False):
    """Collect image file paths from both dataset directories."""
    images = []

    for directory, label in [(COCO_DIR, "coco"), (IMAGENET_DIR, "imagenet")]:
        if not os.path.isdir(directory):
            print(f"  WARNING: {directory} not found, skipping.")
            continue
        files = sorted([
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        if pilot:
            count = PILOT_COCO_COUNT if label == "coco" else PILOT_IMAGENET_COUNT
            random.seed(42)
            files = random.sample(files, min(count, len(files)))
        images.extend(files)

    return images


# ── Inference helpers ───────────────────────────────────────────────────────

def infer_dinov2(model, processor, frame, device) -> str:
    """Run DINOv2 classification on a single frame and return the label."""
    inputs = processor(images=frame, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    idx = outputs.logits.argmax(-1).item()
    return model.config.id2label[idx]


def infer_clip(model, processor, frame, device, labels: list = None, evaluator = None) -> str:
    """Run CLIP zero-shot classification and return the best label."""
    if evaluator is not None:
        inputs = processor(images=frame, return_tensors="pt").to(device)
        with torch.no_grad():
            if hasattr(model, "get_image_features"):
                feat = model.get_image_features(**inputs)
            else:
                outputs = model(**inputs)
                feat = outputs.image_embeds if hasattr(outputs, "image_embeds") else outputs[0]
        res = evaluator.evaluate(feat)
        return res["prediction"]

    inputs = processor(text=labels, images=frame, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    best_idx = probs.argmax(-1).item()
    return labels[best_idx]


def infer_siglip(model, processor, frame, device, labels: list = None, evaluator = None) -> str:
    """Run SigLIP zero-shot classification and return the best label."""
    if evaluator is not None:
        inputs = processor(images=frame, return_tensors="pt").to(device)
        with torch.no_grad():
            if hasattr(model, "get_image_features"):
                feat = model.get_image_features(**inputs)
            else:
                outputs = model(**inputs)
                feat = outputs.image_embeds if hasattr(outputs, "image_embeds") else outputs[0]
        res = evaluator.evaluate(feat)
        return res["prediction"]

    inputs = processor(text=labels, images=frame, return_tensors="pt", padding="max_length").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # SigLIP uses sigmoid BCE loss rather than softmax
    probs = torch.sigmoid(outputs.logits_per_image)
    best_idx = probs.argmax(-1).item()
    return labels[best_idx]


def infer_detr(model, processor, frame, device) -> str:
    """Run DETR object detection and return the top-confidence class label."""
    inputs = processor(images=frame, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits.softmax(-1)[0, :, :-1]  # drop the 'no-object' class
    max_score, max_idx = logits.max(dim=1)
    best_det = max_score.argmax().item()
    class_id = max_idx[best_det].item()
    return model.config.id2label[class_id]


# ── Main benchmark loop ────────────────────────────────────────────────────

def run_benchmark(
    pilot: bool = False,
    batch: int = None,
    shard_id: int = None,
    num_shards: int = None,
    batch_size: int = 500,
    models: list = None,
    clip_classes: str = "1000",
):
    models_to_run = [m.lower() for m in models] if models else ["dinov2", "clip", "detr"]

    # Collect images
    image_files = collect_images(pilot=pilot)
    if not image_files:
        print("\n  ERROR: No images found. Run 'python download_dataset.py' first.")
        sys.exit(1)

    total_pool_size = len(image_files)

    if pilot:
        mode_label = "PILOT (quick validation)"
        csv_path = os.path.join(RESULTS_DIR, "pilot_benchmark.csv")
    elif num_shards is not None and num_shards > 1:
        s_id = shard_id if shard_id is not None else 0
        shards_dir = os.path.join(RESULTS_DIR, "shards")
        os.makedirs(shards_dir, exist_ok=True)
        csv_path = os.path.join(shards_dir, f"shard_{s_id:03d}_of_{num_shards:03d}.csv")
        # Deterministic interleaved sharding across image pool
        image_files = [img for i, img in enumerate(image_files) if i % num_shards == s_id]
        mode_label = f"SHARD {s_id+1}/{num_shards} ({len(image_files)} of {total_pool_size} images)"
    elif batch:
        start_idx = (batch - 1) * batch_size
        end_idx = batch * batch_size
        image_files = image_files[start_idx:end_idx]
        mode_label = f"BATCH {batch} (images {start_idx+1}-{min(end_idx, total_pool_size)})"
        csv_path = os.path.join(RESULTS_DIR, f"batch_{batch}.csv")
    else:
        mode_label = f"FULL ({total_pool_size} images)"
        csv_path = os.path.join(RESULTS_DIR, "full_benchmark.csv")

    print("=" * 70)
    print(f"  AFS-VFM Benchmark - {mode_label}")
    print(f"  Active Models: {', '.join(m.upper() for m in models_to_run)}")
    print(f"  Output CSV:    {csv_path}")
    print("=" * 70)

    # ── Resume support: check for existing partial results ──────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fieldnames = ["image", "dataset", "degradation", "frame", "severity_pct"]
    for m in models_to_run:
        fieldnames.extend([f"{m}_prediction", f"{m}_failure"])

    completed_images = set()
    all_results = []

    if os.path.exists(csv_path):
        # Load existing results and figure out which images are fully done
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_results.append(row)
        # An image is "complete" if it has all 5 degradations x 20 frames = 100 rows
        from collections import Counter
        img_counts = Counter(r["image"] for r in all_results)
        expected_rows_per_image = len(DEGRADATION_TYPES) * NUM_FRAMES
        for img_name, count in img_counts.items():
            if count >= expected_rows_per_image:
                completed_images.add(img_name)
        print(f"\n  RESUMING: Found {len(completed_images)} already-completed images")
        print(f"  Skipping those and continuing from where we left off ...\n")

    remaining = [p for p in image_files if os.path.basename(p) not in completed_images]

    total_inferences = len(remaining) * len(DEGRADATION_TYPES) * NUM_FRAMES * len(models_to_run)
    print(f"\n  Total images:       {len(image_files)}")
    print(f"  Already completed:  {len(completed_images)}")
    print(f"  Remaining:          {len(remaining)}")
    print(f"  Degradation types:  {DEGRADATION_TYPES}")
    print(f"  Frames/sequence:    {NUM_FRAMES}")
    print(f"  Total inferences:   {total_inferences}")

    if not remaining:
        print("\n  All images already processed! Nothing to do.")
        return

    # ── Load models ─────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("  Loading Vision Foundation Models ...")
    print("-" * 70)

    loader = VisionModelLoader()
    device = loader.device

    loaded_models = {}
    for m in models_to_run:
        if hasattr(loader, f"load_{m}"):
            loaded_models[m] = getattr(loader, f"load_{m}")()
        elif hasattr(loader, "load_model"):
            loaded_models[m] = loader.load_model(m)
        else:
            raise ValueError(f"Unknown model: {m}")
        print(f"  [OK] Loaded {m.upper()}")

    print("  All requested models loaded successfully.\n")

    # ── Initialize Zero-Shot 1000-Class Evaluators (Limitation 10) ─────
    evaluators = {}
    if str(clip_classes).lower() in ("1000", "imagenet1k"):
        if "clip" in models_to_run:
            print("  [TextCache] Initializing ImageNet-1K cached evaluator for CLIP...")
            evaluators["clip"] = get_or_build_evaluator(
                model=loaded_models["clip"][0],
                processor=loaded_models["clip"][1],
                model_type="clip",
                device=device,
            )
        if "siglip" in models_to_run:
            print("  [TextCache] Initializing ImageNet-1K cached evaluator for SigLIP...")
            evaluators["siglip"] = get_or_build_evaluator(
                model=loaded_models["siglip"][0],
                processor=loaded_models["siglip"][1],
                model_type="siglip",
                device=device,
            )
    else:
        print(f"  [TextCache] Using legacy {len(CLIP_CANDIDATE_LABELS)}-label candidate subset.")

    # ── Helper to save CSV incrementally ────────────────────────────────
    def save_csv():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_results)

    # ── Run the benchmark ───────────────────────────────────────────────
    start_time = time.time()
    total_images = len(image_files)

    for img_idx, img_path in enumerate(remaining, len(completed_images) + 1):
        img_file = os.path.basename(img_path)
        dataset_source = "coco" if "coco_val" in img_path else "imagenet"

        print(f"\n{'='*70}")
        print(f"  Image [{img_idx}/{total_images}]: {img_file} ({dataset_source})")
        print(f"{'='*70}")

        for deg_type in DEGRADATION_TYPES:
            print(f"\n  >> Degradation: {deg_type}")

            # Generate the 20-frame sequence (Track A)
            try:
                frames = generate_degradation_sequence(img_path, deg_type, num_frames=NUM_FRAMES)
            except Exception as exc:
                print(f"     ERROR generating sequence: {exc}")
                continue

            # Track baselines for each model
            baselines = {m: None for m in models_to_run}

            for frame_idx in range(NUM_FRAMES):
                frame = frames[frame_idx]
                severity_pct = int(frame_idx / (NUM_FRAMES - 1) * 100)

                # Convert numpy array to PIL Image for model processors
                if isinstance(frame, np.ndarray):
                    frame_pil = Image.fromarray(frame)
                else:
                    frame_pil = frame

                row_data = {
                    "image": img_file,
                    "dataset": dataset_source,
                    "degradation": deg_type,
                    "frame": frame_idx + 1,
                    "severity_pct": severity_pct,
                }
                status_parts = []

                for m in models_to_run:
                    m_model, m_proc = loaded_models[m]
                    try:
                        if m in ("dinov2", "dinov2_large"):
                            pred = infer_dinov2(m_model, m_proc, frame_pil, device)
                        elif m == "clip":
                            pred = infer_clip(
                                m_model, m_proc, frame_pil, device,
                                labels=CLIP_CANDIDATE_LABELS,
                                evaluator=evaluators.get("clip"),
                            )
                        elif m == "siglip":
                            pred = infer_siglip(
                                m_model, m_proc, frame_pil, device,
                                labels=CLIP_CANDIDATE_LABELS,
                                evaluator=evaluators.get("siglip"),
                            )
                        elif m == "detr":
                            pred = infer_detr(m_model, m_proc, frame_pil, device)
                        else:
                            pred = "UNKNOWN_MODEL"
                    except Exception:
                        pred = "ERROR"

                    if frame_idx == 0:
                        baselines[m] = pred
                    m_fail = (pred != baselines[m])
                    if m_fail:
                        status_parts.append(f"[{m.upper()} FAIL]")

                    row_data[f"{m}_prediction"] = pred
                    row_data[f"{m}_failure"] = m_fail

                status = (" " + " ".join(status_parts)) if status_parts else ""
                print(f"     Frame {frame_idx+1:02d} ({severity_pct:3d}%){status}")

                all_results.append(row_data)

        # ── SAVE AFTER EVERY IMAGE (crash-safe!) ───────────────────────
        save_csv()
        elapsed_so_far = time.time() - start_time
        imgs_done = img_idx - len(completed_images)
        rate = elapsed_so_far / imgs_done if imgs_done else 0
        remaining_est = rate * (len(remaining) - imgs_done)
        print(f"\n  [SAVED] ({img_idx}/{total_images} images done, "
              f"~{remaining_est/60:.0f} min remaining)")

    # ── Final summary ───────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  BENCHMARK COMPLETE")
    print(f"  Images processed: {total_images}")
    print(f"  Total rows:       {len(all_results)}")
    print(f"  Time elapsed:     {elapsed:.1f}s")
    print(f"  Results saved to: {csv_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AFS-VFM Scalable Benchmark Runner")
    parser.add_argument("--pilot", action="store_true", help="Run a small pilot test (~20 images)")
    parser.add_argument("--batch", type=int, default=None,
                        help="Run sequential batch N (e.g., 1 to 20)")
    parser.add_argument("--batch_size", type=int, default=500,
                        help="Images per batch (default: 500)")
    parser.add_argument("--shard_id", type=int, default=None,
                        help="0-indexed shard ID for distributed execution (0 to num_shards-1)")
    parser.add_argument("--num_shards", type=int, default=None,
                        help="Total number of distributed shards (e.g. 20)")
    parser.add_argument("--models", nargs="+", default=["dinov2", "clip", "detr"],
                        help="Models to evaluate: dinov2 clip detr siglip dinov2_large")
    parser.add_argument("--clip_classes", choices=["1000", "20", "imagenet1k", "subset"], default="1000",
                        help="Zero-shot candidate hierarchy: '1000' (full ImageNet-1K cached) or '20' (legacy subset)")
    args = parser.parse_args()
    run_benchmark(
        pilot=args.pilot,
        batch=args.batch,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        batch_size=args.batch_size,
        models=args.models,
        clip_classes=args.clip_classes,
    )
