"""
AFS-VFM Phase 2B - Strategic Image Subset Selection
====================================================

Selects 300 images from the Phase 2A failure boundary results,
stratified to include:
  - 100 "fragile" images (all 3 models fail early)
  - 100 "robust" images (models survive longest or never fail)
  - 100 "disagreement" images (models fail at very different frames)

This ensures Part 2B's signal probing covers the most scientifically
interesting cases, not just random images.

Usage:
    python -m src.analysis.select_subset
"""

import os
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BOUNDARIES_CSV = os.path.join(ROOT_DIR, "results", "phase2a", "failure_boundaries.csv")
OUTPUT_DIR = os.path.join(ROOT_DIR, "results", "phase2b")
SUBSET_SIZE = 300  # total images to select


def select_strategic_subset(boundaries_path: str = BOUNDARIES_CSV) -> pd.DataFrame:
    """Select 300 strategically interesting images for signal probing."""
    print("=" * 60)
    print("  Phase 2B - Strategic Image Subset Selection")
    print("=" * 60)

    fb = pd.read_csv(boundaries_path)
    print(f"\n  Loaded {len(fb):,} boundary records")
    print(f"  Unique images: {fb['image'].nunique()}")

    # Pivot: one row per (image, degradation), columns = model failure frames
    # We average across all 5 degradation types per image for scoring
    pivot = fb.pivot_table(
        index=["image", "dataset"],
        columns="model",
        values="first_failure_frame",
        aggfunc="mean"
    ).reset_index()

    # Score each image
    # mean_failure: average failure frame across all models (higher = more robust)
    pivot["mean_failure"] = pivot[["dinov2", "clip", "detr"]].mean(axis=1)
    # disagreement: std of failure frames across models (higher = more interesting)
    pivot["disagreement"] = pivot[["dinov2", "clip", "detr"]].std(axis=1)
    # n_survived: how many model-degradation combos survived (NaN = survived)
    surv = fb[fb["survived"] == True].groupby("image").size().reset_index(name="n_survived")
    pivot = pivot.merge(surv, on="image", how="left")
    pivot["n_survived"] = pivot["n_survived"].fillna(0)

    # ---- Category 1: Fragile images (fail early across all models) ----
    fragile = pivot.dropna(subset=["mean_failure"]).nsmallest(100, "mean_failure")
    print(f"\n  Fragile images (all models fail early): {len(fragile)}")
    print(f"    Mean failure frame range: {fragile['mean_failure'].min():.1f} - {fragile['mean_failure'].max():.1f}")

    # ---- Category 2: Robust images (survive longest) ----
    robust = pivot.nlargest(100, "n_survived")
    # Remove any overlap with fragile
    robust = robust[~robust["image"].isin(fragile["image"])]
    if len(robust) < 100:
        # Fill from next most robust by mean_failure
        remaining = pivot[~pivot["image"].isin(fragile["image"]) & ~pivot["image"].isin(robust["image"])]
        extra = remaining.nlargest(100 - len(robust), "mean_failure")
        robust = pd.concat([robust, extra])
    robust = robust.head(100)
    print(f"  Robust images (models survive longest): {len(robust)}")
    print(f"    Mean survivals: {robust['n_survived'].mean():.1f}")

    # ---- Category 3: Disagreement images (models fail at different times) ----
    used = set(fragile["image"]).union(set(robust["image"]))
    remaining = pivot[~pivot["image"].isin(used)]
    disagree = remaining.dropna(subset=["disagreement"]).nlargest(100, "disagreement")
    print(f"  Disagreement images (models disagree on when to fail): {len(disagree)}")
    print(f"    Mean disagreement (std of failure frames): {disagree['disagreement'].mean():.1f}")

    # Combine
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
    subset = select_strategic_subset()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "selected_images.csv")
    subset.to_csv(out_path, index=False)
    print(f"\n  Saved to: {out_path}")

    # Also save just the image filenames (for easy loading in notebook)
    img_list_path = os.path.join(OUTPUT_DIR, "image_list.txt")
    with open(img_list_path, "w") as f:
        for img in subset["image"]:
            f.write(img + "\n")
    print(f"  Image list: {img_list_path}")


if __name__ == "__main__":
    main()
