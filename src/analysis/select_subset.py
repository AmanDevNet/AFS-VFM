"""
AFS-VFM Phase 2B - Strategic Image Subset Selection
====================================================

Selects a stratified sample of images from Phase 2A failure boundary results:
  - "fragile" images (models fail early)
  - "robust" images (models survive longest or never fail)
  - "disagreement" images (models fail at very different frames)

This ensures Part 2B's signal probing covers the most scientifically
interesting cases, not just random images.

Supports arbitrary model suites including the expanded 5-model VFM benchmark.

Usage:
    python -m src.analysis.select_subset
    python -m src.analysis.select_subset --subset_size 300
    python -m src.analysis.select_subset --models dinov2 clip siglip
"""

import os
import argparse
from typing import Optional, Sequence

import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BOUNDARIES_CSV = os.path.join(ROOT_DIR, "results", "phase2a", "failure_boundaries.csv")
OUTPUT_DIR = os.path.join(ROOT_DIR, "results", "phase2b")
SUBSET_SIZE = 300  # total images to select


def select_strategic_subset(
    boundaries_path: str = BOUNDARIES_CSV,
    subset_size: int = SUBSET_SIZE,
    models: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Select a stratified, scientifically interesting subset of images for signal probing.

    Parameters
    ----------
    boundaries_path : str
        Path to failure_boundaries.csv.
    subset_size : int
        Total number of images to select (split equally into 3 strata).
    models : Optional[Sequence[str]]
        Specific models to score across (default: all models present in boundaries).
    """
    print("=" * 60)
    print("  Phase 2B - Strategic Image Subset Selection")
    print("=" * 60)

    fb = pd.read_csv(boundaries_path)
    print(f"\n  Loaded {len(fb):,} boundary records")
    print(f"  Unique images: {fb['image'].nunique()}")

    # Filter to specified models if provided
    if models is not None and len(models) > 0:
        fb = fb[fb["model"].isin(models)].copy()

    # Pivot: one row per (image, degradation), columns = model failure frames
    pivot = fb.pivot_table(
        index=["image", "dataset"],
        columns="model",
        values="first_failure_frame",
        aggfunc="mean"
    ).reset_index()

    metadata_cols = {"image", "dataset", "n_survived"}
    eval_model_cols = [c for c in pivot.columns if c not in metadata_cols]
    if not eval_model_cols:
        eval_model_cols = [c for c in ["dinov2", "clip", "detr"] if c in pivot.columns]

    print(f"  Scoring across {len(eval_model_cols)} models: {eval_model_cols}")

    # Score each image
    # mean_failure: average failure frame across all models (higher = more robust)
    pivot["mean_failure"] = pivot[eval_model_cols].mean(axis=1)
    # disagreement: std of failure frames across models (higher = more interesting)
    pivot["disagreement"] = pivot[eval_model_cols].std(axis=1) if len(eval_model_cols) > 1 else 0.0

    # n_survived: how many model-degradation combos survived (NaN = survived)
    surv = fb[fb["survived"] == True].groupby("image").size().reset_index(name="n_survived")
    pivot = pivot.merge(surv, on="image", how="left")
    pivot["n_survived"] = pivot["n_survived"].fillna(0)

    per_category = max(1, subset_size // 3)

    # ---- Category 1: Fragile images (fail early across all models) ----
    fragile = pivot.dropna(subset=["mean_failure"]).nsmallest(per_category, "mean_failure")
    print(f"\n  Fragile images (models fail early): {len(fragile)}")
    if len(fragile) > 0:
        print(f"    Mean failure frame range: {fragile['mean_failure'].min():.1f} - {fragile['mean_failure'].max():.1f}")

    # ---- Category 2: Robust images (survive longest) ----
    robust = pivot.nlargest(per_category, "n_survived")
    # Remove any overlap with fragile
    robust = robust[~robust["image"].isin(fragile["image"])]
    if len(robust) < per_category:
        # Fill from next most robust by mean_failure
        remaining = pivot[~pivot["image"].isin(fragile["image"]) & ~pivot["image"].isin(robust["image"])]
        extra = remaining.nlargest(per_category - len(robust), "mean_failure")
        robust = pd.concat([robust, extra])
    robust = robust.head(per_category)
    print(f"  Robust images (models survive longest): {len(robust)}")
    if len(robust) > 0:
        print(f"    Mean survivals: {robust['n_survived'].mean():.1f}")

    # ---- Category 3: Disagreement images (models fail at different times) ----
    used = set(fragile["image"]).union(set(robust["image"]))
    remaining = pivot[~pivot["image"].isin(used)]
    disagree = remaining.dropna(subset=["disagreement"]).nlargest(per_category, "disagreement")
    print(f"  Disagreement images (models disagree on when to fail): {len(disagree)}")
    if len(disagree) > 0:
        print(f"    Mean disagreement (std of failure frames): {disagree['disagreement'].mean():.1f}")

    # Combine
    fragile = fragile.copy()
    robust = robust.copy()
    disagree = disagree.copy()

    fragile["category"] = "fragile"
    robust["category"] = "robust"
    disagree["category"] = "disagreement"

    subset = pd.concat([
        fragile[["image", "dataset", "category", "mean_failure", "disagreement", "n_survived"]],
        robust[["image", "dataset", "category", "mean_failure", "disagreement", "n_survived"]],
        disagree[["image", "dataset", "category", "mean_failure", "disagreement", "n_survived"]],
    ]).reset_index(drop=True)

    print(f"\n  Total selected: {len(subset)} images")
    print(f"    COCO:     {(subset['dataset'] == 'coco').sum()}")
    print(f"    ImageNet: {(subset['dataset'] == 'imagenet').sum()}")

    return subset


def main():
    parser = argparse.ArgumentParser(description="AFS-VFM Phase 2B - Strategic Image Subset Selection")
    parser.add_argument("--boundaries", default=BOUNDARIES_CSV, help="Path to failure_boundaries.csv")
    parser.add_argument("--subset_size", type=int, default=SUBSET_SIZE, help="Total number of images to select")
    parser.add_argument("--models", nargs="*", default=None, help="Models to score across")
    args = parser.parse_args()

    subset = select_strategic_subset(
        boundaries_path=args.boundaries,
        subset_size=args.subset_size,
        models=args.models,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "selected_images.csv")
    subset.to_csv(out_path, index=False)
    print(f"\n  Saved to: {out_path}")

    # Also save just the image filenames
    img_list_path = os.path.join(OUTPUT_DIR, "image_list.txt")
    with open(img_list_path, "w", encoding="utf-8") as f:
        for img in subset["image"]:
            f.write(str(img) + "\n")
    print(f"  Image list: {img_list_path}")


if __name__ == "__main__":
    main()
