"""
AFS-VFM Benchmark Data Quality & Robustness Analyzer
=====================================================
Analyzes benchmark CSV outputs (full or pilot) across all evaluated models,
computing overall failure rates, degradation sensitivity, severity brackets,
and top hallucinations.

Usage:
    python analyze_benchmark.py
    python analyze_benchmark.py --input results/pilot_benchmark.csv
    python analyze_benchmark.py --input results/full_benchmark.csv --save_report results/report.txt
"""

import os
import sys
import argparse
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(ROOT_DIR, "results", "full_benchmark.csv")
PILOT_INPUT = os.path.join(ROOT_DIR, "results", "pilot_benchmark.csv")


def discover_models(df: pd.DataFrame, specified_models: list = None) -> list:
    """Discover evaluated models by inspecting failure columns."""
    if specified_models:
        return [m.lower() for m in specified_models]
    found = [col[:-8] for col in df.columns if col.endswith("_failure")]
    return sorted(found) if found else ["dinov2", "clip", "detr"]


def analyze_benchmark(input_path: str = None, save_report: str = None, models: list = None):
    # Resolve input path
    if input_path is None:
        if os.path.exists(DEFAULT_INPUT):
            input_path = DEFAULT_INPUT
        elif os.path.exists(PILOT_INPUT):
            input_path = PILOT_INPUT
        else:
            raise FileNotFoundError(f"No benchmark CSV found at {DEFAULT_INPUT} or {PILOT_INPUT}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Specified benchmark CSV not found: {input_path}")

    print(f"Loading benchmark data from: {input_path}")
    df = pd.read_csv(input_path)

    active_models = discover_models(df, models)

    output_lines = []

    def p(line=""):
        print(line)
        output_lines.append(line)

    p("=" * 70)
    p("  AFS-VFM BENCHMARK - DATA QUALITY & ROBUSTNESS REPORT")
    p("=" * 70)
    p(f"Total rows:           {len(df):,}")
    p(f"Unique images:        {df['image'].nunique():,}" if "image" in df.columns else "Unique images: N/A")
    if "dataset" in df.columns:
        p(f"Datasets:             {dict(df['dataset'].value_counts())}")
    if "degradation" in df.columns:
        p(f"Degradation types:    {sorted(df['degradation'].unique())}")
    if "frame" in df.columns:
        p(f"Frames per sequence:  {df['frame'].nunique()}")
    if "severity_pct" in df.columns:
        p(f"Severity range:       {df['severity_pct'].min()}% - {df['severity_pct'].max()}%")
    p(f"Evaluated models:     {', '.join(m.upper() for m in active_models)}")
    p()

    # 1. Overall failure rates
    p("--- FAILURE RATES BY MODEL ---")
    for m in active_models:
        col = f"{m}_failure"
        if col in df.columns:
            # Handle booleans or string booleans
            failures = (df[col] == True) | (df[col].astype(str).str.lower() == "true")
            rate = failures.sum() / len(df) * 100
            p(f"  {m.upper():14s}: {rate:5.1f}% overall failure rate ({failures.sum():,} / {len(df):,})")
        else:
            p(f"  {m.upper():14s}: [Column {col} missing in CSV]")
    p()

    # 2. Failure rate by degradation type
    if "degradation" in df.columns:
        p("--- FAILURE RATE BY DEGRADATION TYPE ---")
        header = f"  {'Degradation':16s}" + "".join(f"{m.upper():>12s}" for m in active_models)
        p(header)
        p("  " + "-" * (len(header) - 2))

        for deg in sorted(df["degradation"].unique()):
            sub = df[df["degradation"] == deg]
            row_str = f"  {deg:16s}"
            for m in active_models:
                col = f"{m}_failure"
                if col in sub.columns and len(sub) > 0:
                    failures = (sub[col] == True) | (sub[col].astype(str).str.lower() == "true")
                    r = failures.sum() / len(sub) * 100
                    row_str += f"{r:11.1f}%"
                else:
                    row_str += f"{'N/A':>12s}"
            p(row_str)
        p()

    # 3. Failure rate by severity bracket
    if "severity_pct" in df.columns:
        p("--- FAILURE RATE BY SEVERITY BRACKET ---")
        brackets = [
            (0, 20, "Low (0-20%)"),
            (21, 50, "Med (21-50%)"),
            (51, 80, "High (51-80%)"),
            (81, 100, "Extreme (81-100%)"),
        ]
        header = f"  {'Severity Bracket':22s}" + "".join(f"{m.upper():>12s}" for m in active_models)
        p(header)
        p("  " + "-" * (len(header) - 2))

        for lo, hi, label in brackets:
            sub = df[(df["severity_pct"] >= lo) & (df["severity_pct"] <= hi)]
            row_str = f"  {label:22s}"
            for m in active_models:
                col = f"{m}_failure"
                if col in sub.columns and len(sub) > 0:
                    failures = (sub[col] == True) | (sub[col].astype(str).str.lower() == "true")
                    r = failures.sum() / len(sub) * 100
                    row_str += f"{r:11.1f}%"
                else:
                    row_str += f"{'N/A':>12s}"
            p(row_str)
        p()

    # 4. Hallucinations examples under extreme degradation
    if "degradation" in df.columns and "severity_pct" in df.columns:
        scale100 = df[(df["degradation"] == "scale") & (df["severity_pct"] == 100)]
        if not scale100.empty:
            p("--- TOP HALLUCINATIONS UNDER SEVERE CORRUPTION (Scale @ 100%) ---")
            for m in active_models:
                fail_col = f"{m}_failure"
                pred_col = f"{m}_prediction"
                if fail_col in scale100.columns and pred_col in scale100.columns:
                    failed = scale100[(scale100[fail_col] == True) | (scale100[fail_col].astype(str).str.lower() == "true")]
                    top_h = dict(failed[pred_col].value_counts().head(5))
                    p(f"  {m.upper():12s} top hallucinations: {top_h}")
            p()

    p("=" * 70)
    p("  REPORT COMPLETE")
    p("=" * 70)

    if save_report:
        os.makedirs(os.path.dirname(os.path.abspath(save_report)), exist_ok=True)
        with open(save_report, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
        print(f"\nReport written to: {save_report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AFS-VFM Benchmark Data Quality Report")
    parser.add_argument("--input", type=str, default=None, help="Path to benchmark CSV (default: results/full_benchmark.csv)")
    parser.add_argument("--save_report", type=str, default=None, help="Optional path to export text report")
    parser.add_argument("--models", nargs="+", default=None, help="Subset of models to analyze")
    args = parser.parse_args()
    analyze_benchmark(input_path=args.input, save_report=args.save_report, models=args.models)
