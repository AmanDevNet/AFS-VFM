"""
AFS-VFM Shard Merger & Verification Tool
========================================
Discovers, validates, and merges distributed benchmark shard CSVs
(e.g., shard_000_of_020.csv, batch_1.csv) into a unified master dataset.

Features:
  - Schema consistency validation across shards
  - Duplicate detection and resolution
  - Missing frame / trajectory integrity checking
  - Summary metrics and integrity reporting
"""

import os
import sys
import glob
import argparse
from typing import List, Dict, Any, Optional
import pandas as pd


def discover_shard_files(input_dir: str, pattern: Optional[str] = None) -> List[str]:
    """Find all shard and batch CSV files in the input directory."""
    if pattern:
        search_path = os.path.join(input_dir, pattern)
        files = glob.glob(search_path)
    else:
        files = glob.glob(os.path.join(input_dir, "shard_*.csv"))
        if not files:
            files = glob.glob(os.path.join(input_dir, "batch_*.csv"))
        if not files:
            files = glob.glob(os.path.join(input_dir, "*.csv"))

    # Exclude previously merged files or summaries
    excluded_names = {"full_benchmark.csv", "full_benchmark_10k.csv", "pilot_benchmark.csv"}
    valid_files = [f for f in files if os.path.basename(f) not in excluded_names]
    return sorted(valid_files)


def merge_and_validate_shards(
    input_dir: str,
    output_path: str,
    pattern: Optional[str] = None,
    expected_shards: Optional[int] = None,
    expected_frames_per_image: int = 100,  # 5 degradations * 20 frames
) -> pd.DataFrame:
    """Validate and combine shard files into a single master benchmark file."""
    print("=" * 72)
    print("  AFS-VFM Shard Merger & Data Integrity Verification Tool")
    print("=" * 72)
    print(f"Scanning directory: {input_dir}")

    shard_files = discover_shard_files(input_dir, pattern)
    if not shard_files:
        raise FileNotFoundError(f"No shard CSV files found in {input_dir}")

    print(f"Discovered {len(shard_files)} shard file(s):")
    for f in shard_files:
        print(f"  - {os.path.basename(f)} ({os.path.getsize(f) / (1024*1024):.1f} MB)")

    if expected_shards and len(shard_files) < expected_shards:
        print(f"\n[WARNING] Expected {expected_shards} shards, but found {len(shard_files)}.")

    dataframes = []
    base_columns = None

    print("\nValidating schema consistency...")
    for f in shard_files:
        df = pd.read_csv(f)
        if base_columns is None:
            base_columns = list(df.columns)
        else:
            missing_cols = set(base_columns) - set(df.columns)
            extra_cols = set(df.columns) - set(base_columns)
            if missing_cols:
                print(f"  [ERROR] {os.path.basename(f)} missing columns: {missing_cols}")
            if extra_cols:
                print(f"  [WARN] {os.path.basename(f)} has extra columns: {extra_cols}")
        dataframes.append(df)

    merged = pd.concat(dataframes, ignore_index=True)
    initial_rows = len(merged)
    print(f"\nTotal concatenated rows: {initial_rows:,}")

    # Deduplicate rows based on (image, degradation, frame)
    key_cols = [c for c in ["image", "degradation", "frame"] if c in merged.columns]
    if key_cols:
        merged = merged.drop_duplicates(subset=key_cols, keep="last")
        deduped_rows = len(merged)
        if deduped_rows < initial_rows:
            print(f"Removed {initial_rows - deduped_rows:,} duplicate rows.")

    # Integrity check: count frames per image
    if "image" in merged.columns:
        image_counts = merged["image"].value_counts()
        total_unique_images = len(image_counts)
        incomplete_images = image_counts[image_counts < expected_frames_per_image]

        print(f"Unique images:       {total_unique_images:,}")
        if len(incomplete_images) > 0:
            print(f"[WARNING] {len(incomplete_images)} images have fewer than {expected_frames_per_image} frames!")
        else:
            print(f"[PASS] All {total_unique_images:,} images have exactly {expected_frames_per_image} frames.")

    # Save merged dataset
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    merged.to_csv(output_path, index=False)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print("\n" + "=" * 72)
    print(f"SUCCESS: Merged benchmark written to: {output_path}")
    print(f"Final Row Count:  {len(merged):,} rows")
    print(f"Final File Size:  {file_size_mb:.1f} MB")
    print("=" * 72)

    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge AFS-VFM distributed benchmark shards")
    parser.add_argument("--input_dir", type=str, default="results/shards",
                        help="Directory containing shard CSV files (default: results/shards)")
    parser.add_argument("--output", type=str, default="results/full_benchmark_10k.csv",
                        help="Path for output merged CSV file (default: results/full_benchmark_10k.csv)")
    parser.add_argument("--pattern", type=str, default=None,
                        help="Filename glob pattern to match (e.g. 'shard_*.csv')")
    parser.add_argument("--expected_shards", type=int, default=None,
                        help="Expected number of shards to verify completeness")
    args = parser.parse_args()

    merge_and_validate_shards(
        input_dir=args.input_dir,
        output_path=args.output,
        pattern=args.pattern,
        expected_shards=args.expected_shards,
    )
