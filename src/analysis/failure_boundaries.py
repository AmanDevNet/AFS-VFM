"""
AFS-VFM Phase 2A - Failure Boundary Analysis
=============================================

Analyzes benchmark CSVs to:
  1. Find the exact first-failure frame for every (image, degradation, model) triplet
  2. Compute failure statistics per (model x degradation) group
  3. Run statistical significance tests (Kruskal-Wallis, Mann-Whitney U)
  4. Build failure transition matrices (what classes do models drift toward?)
  5. Generate a human-readable summary report

Supports dynamic model discovery across standard and expanded model registries
(DINOv2-Base, DINOv2-Large, CLIP, SigLIP, DETR).

Usage:
    python -m src.analysis.failure_boundaries
    python -m src.analysis.failure_boundaries --csv path/to/benchmark.csv
    python -m src.analysis.failure_boundaries --models dinov2 clip siglip
"""

import os
import sys
import argparse
from itertools import combinations
from typing import Optional, Sequence, List, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

# ── Paths & Model Registries ────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CSV = os.path.join(ROOT_DIR, "results", "full_benchmark.csv")
OUTPUT_DIR = os.path.join(ROOT_DIR, "results", "phase2a")

SUPPORTED_MODELS: list[str] = ["dinov2", "dinov2_large", "clip", "siglip", "detr"]
DEFAULT_MODELS: list[str] = ["dinov2", "clip", "detr"]
MODELS: list[str] = DEFAULT_MODELS  # Backward-compatibility alias
NUM_FRAMES: int = 20


def discover_models(df: pd.DataFrame, models: Optional[Sequence[str]] = None) -> list[str]:
    """Dynamically discover models present in a benchmark or boundaries DataFrame."""
    if models is not None and len(models) > 0:
        return list(models)

    discovered = []
    # Check for '{model}_failure' pattern in columns
    for col in df.columns:
        if col.endswith("_failure"):
            candidate = col[:-8]
            if candidate not in discovered:
                discovered.append(candidate)

    # Check if 'model' column exists with categorical values
    if "model" in df.columns:
        for val in df["model"].dropna().unique():
            s_val = str(val)
            if s_val not in discovered:
                discovered.append(s_val)

    return discovered if discovered else list(DEFAULT_MODELS)


# ═════════════════════════════════════════════════════════════════════════════
# Step 1: First-Failure Frame Extraction
# ═════════════════════════════════════════════════════════════════════════════

def extract_failure_boundaries(
    df: pd.DataFrame, models: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """Find the first frame where each model fails for every (image, degradation) pair.

    Returns a DataFrame with columns:
        image, dataset, degradation, model, first_failure_frame,
        first_failure_severity_pct, survived
    """
    active_models = discover_models(df, models)
    print(f"\n  [Step 1] Extracting first-failure frames for {len(active_models)} models: {active_models} ...")
    records = []

    # Group by (image, degradation) — each group has 20 rows (one per frame)
    grouped = df.groupby(["image", "dataset", "degradation"])

    for (image, dataset, degradation), group in grouped:
        group = group.sort_values("frame")

        for model in active_models:
            fail_col = f"{model}_failure"
            if fail_col not in group.columns:
                continue

            # Find rows where this model failed
            failures = group[group[fail_col] == True]

            if len(failures) == 0:
                # Model survived all 20 frames
                records.append({
                    "image": image,
                    "dataset": dataset,
                    "degradation": degradation,
                    "model": model,
                    "first_failure_frame": np.nan,
                    "first_failure_severity_pct": np.nan,
                    "survived": True,
                })
            else:
                first = failures.iloc[0]
                records.append({
                    "image": image,
                    "dataset": dataset,
                    "degradation": degradation,
                    "model": model,
                    "first_failure_frame": int(first["frame"]),
                    "first_failure_severity_pct": int(first["severity_pct"]),
                    "survived": False,
                })

    result = pd.DataFrame(records)
    n_total = len(result)
    n_survived = result["survived"].sum()
    print(f"         {n_total:,} triplets analyzed")
    print(f"         {n_survived:,} survived all frames ({n_survived/max(1, n_total)*100:.1f}%)")
    print(f"         {n_total - n_survived:,} had at least one failure")
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Step 2: Failure Statistics Per Group
# ═════════════════════════════════════════════════════════════════════════════

def compute_failure_statistics(
    boundaries: pd.DataFrame, models: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """Compute failure stats for each (model x degradation) group."""
    active_models = discover_models(boundaries, models)
    print(f"\n  [Step 2] Computing failure statistics for {len(active_models)} models ...")
    records = []

    for model in active_models:
        model_data = boundaries[boundaries["model"] == model]

        for deg in sorted(boundaries["degradation"].unique()):
            group = model_data[model_data["degradation"] == deg]
            total = len(group)
            if total == 0:
                continue
            survived = group["survived"].sum()
            failed = group[~group["survived"]]
            failure_frames = failed["first_failure_frame"].dropna()

            record = {
                "model": model,
                "degradation": deg,
                "total_images": total,
                "survived_count": int(survived),
                "failed_count": int(total - survived),
                "survival_rate_pct": round(survived / total * 100, 2),
                "failure_rate_pct": round((total - survived) / total * 100, 2),
            }

            if len(failure_frames) > 0:
                record["mean_failure_frame"] = round(failure_frames.mean(), 2)
                record["median_failure_frame"] = round(failure_frames.median(), 2)
                record["std_failure_frame"] = round(failure_frames.std(), 2)
                record["min_failure_frame"] = int(failure_frames.min())
                record["max_failure_frame"] = int(failure_frames.max())
                # Fragility: higher = fails earlier = more fragile
                record["fragility_score"] = round(
                    1 - (failure_frames.mean() / NUM_FRAMES), 3
                )
                # Percentiles
                record["p25_failure_frame"] = round(failure_frames.quantile(0.25), 1)
                record["p75_failure_frame"] = round(failure_frames.quantile(0.75), 1)
            else:
                for col in ["mean_failure_frame", "median_failure_frame",
                            "std_failure_frame", "min_failure_frame",
                            "max_failure_frame", "fragility_score",
                            "p25_failure_frame", "p75_failure_frame"]:
                    record[col] = np.nan

            records.append(record)

    result = pd.DataFrame(records)
    print(f"         {len(result)} groups analyzed ({len(active_models)} models x {len(result)//max(1, len(active_models))} degradations)")
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Step 3: Statistical Significance Tests
# ═════════════════════════════════════════════════════════════════════════════

def run_statistical_tests(
    boundaries: pd.DataFrame, models: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """Run Kruskal-Wallis and pairwise Mann-Whitney U tests."""
    active_models = discover_models(boundaries, models)
    print(f"\n  [Step 3] Running statistical significance tests across {len(active_models)} models ...")

    # Only use triplets where the model actually failed (has a failure frame)
    failed = boundaries[~boundaries["survived"]].copy()
    records = []
    degradations = sorted(boundaries["degradation"].unique())

    for deg in degradations:
        deg_data = failed[failed["degradation"] == deg]

        # Collect failure frames per model
        model_groups = {}
        for model in active_models:
            frames = deg_data[deg_data["model"] == model]["first_failure_frame"].dropna()
            model_groups[model] = frames.values

        # ── Kruskal-Wallis omnibus comparison ──
        groups_for_kw = [g for g in model_groups.values() if len(g) >= 2]
        if len(groups_for_kw) >= 2:
            h_stat, kw_p = stats.kruskal(*groups_for_kw)
        else:
            h_stat, kw_p = np.nan, np.nan

        records.append({
            "degradation": deg,
            "test": "kruskal_wallis",
            "comparison": "all_models",
            "statistic": round(h_stat, 4) if not np.isnan(h_stat) else np.nan,
            "p_value": kw_p,
            "p_value_corrected": kw_p,  # no correction needed for omnibus
            "significant_005": kw_p < 0.05 if not np.isnan(kw_p) else False,
            "effect_size": np.nan,
            "effect_size_label": "",
            "n_group_a": sum(len(g) for g in model_groups.values()),
            "n_group_b": np.nan,
        })

        # ── Pairwise Mann-Whitney U with Bonferroni correction ──
        pairs = list(combinations(active_models, 2))
        n_pairs = len(pairs)

        for model_a, model_b in pairs:
            ga = model_groups.get(model_a, np.array([]))
            gb = model_groups.get(model_b, np.array([]))

            if len(ga) < 2 or len(gb) < 2:
                continue

            u_stat, mw_p = stats.mannwhitneyu(ga, gb, alternative="two-sided")
            p_corrected = min(mw_p * max(1, n_pairs), 1.0)  # Bonferroni

            # Cohen's d (using pooled std)
            pooled_std = np.sqrt(
                ((len(ga) - 1) * ga.std()**2 + (len(gb) - 1) * gb.std()**2)
                / max(1, (len(ga) + len(gb) - 2))
            )
            cohens_d = abs(ga.mean() - gb.mean()) / pooled_std if pooled_std > 0 else 0

            # Effect size label
            if cohens_d < 0.2:
                es_label = "negligible"
            elif cohens_d < 0.5:
                es_label = "small"
            elif cohens_d < 0.8:
                es_label = "medium"
            else:
                es_label = "large"

            records.append({
                "degradation": deg,
                "test": "mann_whitney_u",
                "comparison": f"{model_a}_vs_{model_b}",
                "statistic": round(u_stat, 4),
                "p_value": mw_p,
                "p_value_corrected": p_corrected,
                "significant_005": p_corrected < 0.05,
                "effect_size": round(cohens_d, 4),
                "effect_size_label": es_label,
                "n_group_a": len(ga),
                "n_group_b": len(gb),
            })

    result = pd.DataFrame(records)
    n_sig = result[result["test"] == "mann_whitney_u"]["significant_005"].sum()
    n_total = len(result[result["test"] == "mann_whitney_u"])
    print(f"         {n_sig}/{max(1, n_total)} pairwise comparisons are significant (p < 0.05, Bonferroni)")
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Step 4: Failure Transition Matrices
# ═════════════════════════════════════════════════════════════════════════════

def build_transition_matrices(
    df: pd.DataFrame, models: Optional[Sequence[str]] = None
) -> dict:
    """For each model, find what class predictions drift TO when failure occurs."""
    active_models = discover_models(df, models)
    print(f"\n  [Step 4] Building failure transition matrices for {len(active_models)} models ...")
    transitions = {}

    for model in active_models:
        pred_col = f"{model}_prediction"
        fail_col = f"{model}_failure"

        if pred_col not in df.columns or fail_col not in df.columns:
            continue

        # Get baseline predictions (frame 1) and failure predictions
        baselines = df[df["frame"] == 1][["image", "degradation", pred_col]].copy()
        baselines = baselines.rename(columns={pred_col: "baseline_prediction"})

        failed_rows = df[df[fail_col] == True].copy()

        if len(failed_rows) == 0:
            transitions[model] = pd.DataFrame()
            continue

        # Merge to get baseline alongside each failure
        merged = failed_rows.merge(
            baselines, on=["image", "degradation"], how="left"
        )

        # Count transitions: baseline_class -> failed_class
        trans = (
            merged.groupby(["baseline_prediction", pred_col])
            .size()
            .reset_index(name="count")
            .rename(columns={pred_col: "failure_prediction"})
            .sort_values("count", ascending=False)
        )
        transitions[model] = trans

        n_unique = len(trans)
        top = trans.iloc[0] if len(trans) > 0 else None
        print(f"         {model.upper()}: {n_unique} unique transitions")
        if top is not None:
            print(f"           Top: '{top['baseline_prediction']}' -> "
                  f"'{top['failure_prediction']}' ({top['count']} times)")

    return transitions


# ═════════════════════════════════════════════════════════════════════════════
# Step 5: Generate Human-Readable Report
# ═════════════════════════════════════════════════════════════════════════════

def generate_report(
    boundaries: pd.DataFrame,
    statistics: pd.DataFrame,
    tests: pd.DataFrame,
    transitions: dict,
    models: Optional[Sequence[str]] = None,
) -> str:
    """Generate a human-readable text report summarizing all Phase 2A findings."""
    active_models = discover_models(boundaries, models)
    print("\n  [Step 5] Generating summary report ...")

    lines = []
    lines.append("=" * 70)
    lines.append("AFS-VFM -- Phase 2A: Failure Boundary Analysis Report")
    lines.append("=" * 70)
    lines.append("")

    # ── Overall Summary ──
    total = len(boundaries)
    survived = boundaries["survived"].sum()
    lines.append("--- OVERALL SUMMARY ---")
    lines.append(f"  Total (image, degradation, model) triplets: {total:,}")
    lines.append(f"  Survived all 20 frames: {survived:,} ({survived/max(1, total)*100:.1f}%)")
    lines.append(f"  Had at least one failure: {total - survived:,} ({(total-survived)/max(1, total)*100:.1f}%)")
    lines.append("")

    # Per-model survival
    lines.append("--- PER-MODEL SURVIVAL RATE ---")
    for model in active_models:
        m = boundaries[boundaries["model"] == model]
        s = m["survived"].sum()
        total_m = len(m)
        if total_m > 0:
            lines.append(f"  {model.upper():12s}: {s}/{total_m} survived "
                          f"({s/total_m*100:.1f}% survival rate)")
        else:
            lines.append(f"  {model.upper():12s}: 0/0 survived")
    lines.append("")

    # ── Failure Statistics Table ──
    lines.append("--- FAILURE STATISTICS (model x degradation) ---")
    lines.append(f"  {'Model':<12s} {'Degradation':<12s} {'FailRate%':>9s} "
                  f"{'MeanFrame':>9s} {'MedianFr':>9s} {'StdFr':>7s} "
                  f"{'Fragility':>9s} {'Survived%':>9s}")
    lines.append("  " + "-" * 84)

    for _, row in statistics.iterrows():
        mf = f"{row['mean_failure_frame']:.1f}" if pd.notna(row['mean_failure_frame']) else "N/A"
        md = f"{row['median_failure_frame']:.1f}" if pd.notna(row['median_failure_frame']) else "N/A"
        sd = f"{row['std_failure_frame']:.1f}" if pd.notna(row['std_failure_frame']) else "N/A"
        fg = f"{row['fragility_score']:.3f}" if pd.notna(row['fragility_score']) else "N/A"
        lines.append(f"  {row['model']:<12s} {row['degradation']:<12s} "
                      f"{row['failure_rate_pct']:>8.1f}% {mf:>9s} {md:>9s} "
                      f"{sd:>7s} {fg:>9s} {row['survival_rate_pct']:>8.1f}%")
    lines.append("")

    # ── Most/Least Fragile ──
    valid = statistics.dropna(subset=["fragility_score"])
    if len(valid) > 0:
        most_fragile = valid.loc[valid["fragility_score"].idxmax()]
        least_fragile = valid.loc[valid["fragility_score"].idxmin()]
        lines.append("--- KEY FINDINGS ---")
        lines.append(f"  Most fragile:  {most_fragile['model'].upper()} under "
                      f"{most_fragile['degradation']} "
                      f"(fragility={most_fragile['fragility_score']:.3f}, "
                      f"mean failure at frame {most_fragile['mean_failure_frame']:.1f})")
        lines.append(f"  Most robust:   {least_fragile['model'].upper()} under "
                      f"{least_fragile['degradation']} "
                      f"(fragility={least_fragile['fragility_score']:.3f}, "
                      f"mean failure at frame {least_fragile['mean_failure_frame']:.1f})")
        lines.append("")

    # ── Robustness Ranking Per Degradation ──
    lines.append("--- ROBUSTNESS RANKING (by mean failure frame, higher = more robust) ---")
    for deg in sorted(statistics["degradation"].unique()):
        deg_data = valid[valid["degradation"] == deg].sort_values(
            "mean_failure_frame", ascending=False
        )
        ranking = " > ".join(
            f"{r['model'].upper()}(fr={r['mean_failure_frame']:.1f})"
            for _, r in deg_data.iterrows()
        )
        lines.append(f"  {deg:<12s}: {ranking}")
    lines.append("")

    # ── Statistical Tests Summary ──
    lines.append("--- STATISTICAL SIGNIFICANCE (Bonferroni-corrected) ---")
    kw_tests = tests[tests["test"] == "kruskal_wallis"]
    for _, row in kw_tests.iterrows():
        sig = "***" if row["significant_005"] else "n.s."
        h_str = f"{row['statistic']:.2f}" if pd.notna(row['statistic']) else "N/A"
        p_str = f"{row['p_value']:.2e}" if pd.notna(row['p_value']) else "N/A"
        lines.append(f"  {row['degradation']:<12s} Kruskal-Wallis H={h_str}, p={p_str} {sig}")

    lines.append("")
    lines.append("  Pairwise Mann-Whitney U (significant only):")
    mw_sig = tests[(tests["test"] == "mann_whitney_u") & (tests["significant_005"])]
    if len(mw_sig) == 0:
        lines.append("    (none significant after Bonferroni correction)")
    else:
        for _, row in mw_sig.iterrows():
            lines.append(f"    {row['degradation']:<12s} {row['comparison']:<24s} "
                          f"p_corrected={row['p_value_corrected']:.4f}, "
                          f"d={row['effect_size']:.3f} ({row['effect_size_label']})")
    lines.append("")

    # ── Top Hallucination Attractors ──
    lines.append("--- TOP HALLUCINATION ATTRACTORS (most common failure predictions) ---")
    for model in active_models:
        trans = transitions.get(model, pd.DataFrame())
        if len(trans) == 0:
            continue
        top5 = trans.head(5)
        lines.append(f"  {model.upper()}:")
        for _, row in top5.iterrows():
            lines.append(f"    '{row['baseline_prediction']}' -> "
                          f"'{row['failure_prediction']}' ({row['count']} times)")
    lines.append("")

    lines.append("=" * 70)
    lines.append("End of Phase 2A Report")
    lines.append("=" * 70)

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main(csv_path: str = DEFAULT_CSV, models: Optional[Sequence[str]] = None):
    print("=" * 70)
    print("  AFS-VFM Phase 2A - Failure Boundary Analysis")
    print("=" * 70)

    # Load data
    print(f"\n  Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} rows, {df['image'].nunique():,} unique images")

    active_models = discover_models(df, models)
    print(f"  Active model registry ({len(active_models)} models): {active_models}")

    # Ensure boolean columns are actual booleans
    for model in active_models:
        col = f"{model}_failure"
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower() == "true"

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trans_dir = os.path.join(OUTPUT_DIR, "transition_matrices")
    os.makedirs(trans_dir, exist_ok=True)

    # Step 1: Extract failure boundaries
    boundaries = extract_failure_boundaries(df, models=active_models)
    boundaries.to_csv(os.path.join(OUTPUT_DIR, "failure_boundaries.csv"), index=False)
    print(f"         > Saved failure_boundaries.csv ({len(boundaries):,} rows)")

    # Step 2: Compute statistics
    statistics = compute_failure_statistics(boundaries, models=active_models)
    statistics.to_csv(os.path.join(OUTPUT_DIR, "failure_statistics.csv"), index=False)
    print(f"         > Saved failure_statistics.csv ({len(statistics)} rows)")

    # Step 3: Statistical tests
    tests = run_statistical_tests(boundaries, models=active_models)
    tests.to_csv(os.path.join(OUTPUT_DIR, "statistical_tests.csv"), index=False)
    print(f"         > Saved statistical_tests.csv ({len(tests)} rows)")

    # Step 4: Transition matrices
    transitions = build_transition_matrices(df, models=active_models)
    for model, trans_df in transitions.items():
        path = os.path.join(trans_dir, f"{model}_transitions.csv")
        trans_df.to_csv(path, index=False)
        print(f"         > Saved {model}_transitions.csv ({len(trans_df)} rows)")

    # Step 5: Generate report
    report = generate_report(boundaries, statistics, tests, transitions, models=active_models)
    report_path = os.path.join(OUTPUT_DIR, "phase2a_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n         > Saved phase2a_report.txt")

    # Print the report to console too
    print("\n")
    print(report)

    print(f"\n  All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AFS-VFM Phase 2A - Failure Boundary Analysis")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to benchmark CSV")
    parser.add_argument("--models", nargs="*", default=None, help="Models to evaluate (default: auto-discover)")
    args = parser.parse_args()
    main(csv_path=args.csv, models=args.models)
