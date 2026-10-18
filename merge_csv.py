"""
AFS-VFM Benchmark CSV Merging Utility
======================================
Merges batch CSV files (e.g., batch_1.csv, batch_2.csv, batch_3.csv) into
a unified benchmark dataset with schema validation and deduplication support.

Usage:
    python merge_csv.py
    python merge_csv.py --pattern "batch_*.csv" --output results/full_benchmark.csv
    python merge_csv.py --files results/batch_1.csv results/batch_2.csv
"""

import os
import glob
import argparse
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(ROOT_DIR, "results")
DEFAULT_OUTPUT = os.path.join(DEFAULT_RESULTS_DIR, "full_benchmark.csv")


def merge_csvs(
    files: list = None,
    pattern: str = None,
    input_dir: str = None,
    output_path: str = None,
    dedup: bool = False,
):
    results_dir = input_dir or DEFAULT_RESULTS_DIR
    out_file = output_path or DEFAULT_OUTPUT

    # Determine file paths
    resolved_files = []
    if files:
        for f in files:
            path = f if os.path.isabs(f) else os.path.join(results_dir, f)
            resolved_files.append(path)
    elif pattern:
        search_path = pattern if os.path.isabs(pattern) else os.path.join(results_dir, pattern)
        resolved_files = sorted(glob.glob(search_path))
    else:
        # Default: batch_1.csv, batch_2.csv, batch_3.csv
        candidate_names = ["batch_1.csv", "batch_2.csv", "batch_3.csv"]
        resolved_files = [os.path.join(results_dir, c) for c in candidate_names]

    if not resolved_files:
        print(f"Error: No files found matching criteria in {results_dir}")
        return False

    print("=" * 60)
    print("  AFS-VFM Benchmark CSV Merger")
    print("=" * 60)

    dataframes = []
    for filepath in resolved_files:
        if not os.path.exists(filepath):
            print(f"  [ERROR] File not found: {filepath}")
            return False

        print(f"  Loading {os.path.basename(filepath)} ...", end=" ")
        df = pd.read_csv(filepath)
        print(f"({len(df):,} rows, {len(df.columns)} columns)")
        dataframes.append(df)

    # Validate column compatibility
    base_cols = list(dataframes[0].columns)
    for i, df in enumerate(dataframes[1:], 2):
        missing = set(base_cols) - set(df.columns)
        extra = set(df.columns) - set(base_cols)
        if missing or extra:
            print(f"  [WARNING] File {i} schema differs: missing={missing}, extra={extra}")

    print("\n  Merging datasets ...")
    combined = pd.concat(dataframes, ignore_index=True)

    if dedup:
        initial_len = len(combined)
        dedup_subset = [c for c in ["image", "dataset", "degradation", "frame"] if c in combined.columns]
        if dedup_subset:
            combined = combined.drop_duplicates(subset=dedup_subset)
            print(f"  Deduplicated: removed {initial_len - len(combined):,} duplicate rows")

    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    combined.to_csv(out_file, index=False)

    print("\n" + "=" * 60)
    print("  MERGE COMPLETE")
    print(f"  Total rows:    {len(combined):,}")
    print(f"  Total columns: {len(combined.columns)}")
    print(f"  Saved to:      {out_file}")
    print("=" * 60)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge AFS-VFM batch CSV files")
    parser.add_argument("--files", nargs="+", default=None, help="Specific list of CSV files to merge")
    parser.add_argument("--pattern", type=str, default=None, help="Glob pattern for files (e.g. 'batch_*.csv')")
    parser.add_argument("--input_dir", type=str, default=None, help="Base directory for files (default: results/)")
    parser.add_argument("--output", type=str, default=None, help="Target merged output CSV (default: results/full_benchmark.csv)")
    parser.add_argument("--dedup", action="store_true", help="Deduplicate rows based on (image, degradation, frame)")
    args = parser.parse_args()

    merge_csvs(
        files=args.files,
        pattern=args.pattern,
        input_dir=args.input_dir,
        output_path=args.output,
        dedup=args.dedup,
    )
