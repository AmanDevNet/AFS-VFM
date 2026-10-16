"""
AFS-VFM Phase 2C - Anticipatory Signal Detection
=================================================

THE CORE HYPOTHESIS TEST: Do internal model signals degrade
BEFORE the model's final prediction actually changes?

Uses data from:
  - Phase 2A: failure_boundaries.csv (WHEN each model fails)
  - Phase 2B: frame_signals.csv (WHAT signals look like at each frame)

Produces:
  1. Failure-aligned signal trajectories (re-indexed relative to failure frame)
  2. Pre-failure window analysis (signal slope, acceleration, breakpoints)
  3. Survived vs failed comparison (control group)
  4. Predictability metrics (lead time, SNR, AUROC)

Usage:
    python -m src.analysis.anticipatory_signals
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BOUNDARIES_CSV = os.path.join(ROOT_DIR, "results", "phase2a", "failure_boundaries.csv")
SIGNALS_CSV = os.path.join(ROOT_DIR, "results", "phase2b", "signals", "frame_signals.csv")
OUTPUT_DIR = os.path.join(ROOT_DIR, "results", "phase2c")

MODELS = ["dinov2", "clip", "detr"]
NUM_FRAMES = 20
ANTICIPATORY_WINDOW = 5  # frames before failure to analyze


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Merge failure boundaries with signal data
# ═══════════════════════════════════════════════════════════════════════════════

def merge_boundaries_and_signals(boundaries: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """For each (image, degradation, model), attach the first_failure_frame to every signal row."""
    print("\n  [Step 1] Merging failure boundaries with signal trajectories ...")

    # Boundaries has one row per (image, degradation, model) with first_failure_frame
    # Signals has one row per (image, degradation, frame) with signals for ALL models
    # We need to "unpivot" signals to be per-model

    all_merged = []

    for model in MODELS:
        # Get boundary data for this model
        model_bounds = boundaries[boundaries["model"] == model][
            ["image", "degradation", "first_failure_frame", "survived"]
        ].copy()

        # Get signal columns for this model
        if model == "dinov2":
            sig_cols = ["dinov2_top1_confidence", "dinov2_logit_margin",
                        "dinov2_cosine_sim", "dinov2_l2_distance", "dinov2_top5_overlap"]
        elif model == "clip":
            sig_cols = ["clip_top1_confidence", "clip_logit_margin",
                        "clip_cosine_sim", "clip_l2_distance"]
        else:  # detr
            detr_candidates = [
                "detr_top_detection_confidence", "detr_objectness",
                "detr_num_detections", "detr_cosine_sim", "detr_l2_distance"
            ]
            sig_cols = [c for c in detr_candidates if c in signals.columns]
            if not sig_cols:
                sig_cols = ["detr_top_detection_confidence", "detr_objectness", "detr_num_detections"]

        # Subset signals to relevant columns
        sig_subset = signals[["image", "degradation", "frame", "severity_pct"] + sig_cols].copy()

        # Rename to generic names for unified analysis
        rename_map = {}
        for col in sig_cols:
            generic = col.replace(f"{model}_", "")
            rename_map[col] = generic
        sig_subset = sig_subset.rename(columns=rename_map)

        # Merge with boundaries
        merged = sig_subset.merge(model_bounds, on=["image", "degradation"], how="inner")
        merged["model"] = model

        # Compute relative_frame (0 = failure frame, negative = before failure)
        merged["relative_frame"] = np.nan
        failed_mask = ~merged["survived"]
        merged.loc[failed_mask, "relative_frame"] = (
            merged.loc[failed_mask, "frame"] - merged.loc[failed_mask, "first_failure_frame"]
        )

        all_merged.append(merged)

    result = pd.concat(all_merged, ignore_index=True)
    n_failed = result[~result["survived"]].drop_duplicates(["image", "degradation", "model"]).shape[0]
    n_survived = result[result["survived"]].drop_duplicates(["image", "degradation", "model"]).shape[0]
    print(f"         {len(result):,} signal rows merged")
    print(f"         {n_failed:,} failed triplets, {n_survived:,} survived triplets")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Failure-Aligned Signal Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_failure_aligned_signals(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute average signal values at each relative frame position."""
    print("\n  [Step 2] Building failure-aligned signal profiles ...")

    failed = merged[~merged["survived"]].copy()

    # For each model, compute mean signal at each relative frame
    records = []
    signal_cols = ["top1_confidence", "logit_margin", "cosine_sim"]

    for model in MODELS:
        model_data = failed[failed["model"] == model]

        # Only look at relative frames from -WINDOW to +5
        for rel_frame in range(-ANTICIPATORY_WINDOW, 6):
            frame_data = model_data[model_data["relative_frame"] == rel_frame]

            if len(frame_data) == 0:
                continue

            record = {
                "model": model,
                "relative_frame": rel_frame,
                "n_samples": len(frame_data),
            }

            for sig in signal_cols:
                if sig in frame_data.columns:
                    vals = frame_data[sig].dropna()
                    if len(vals) > 0:
                        record[f"{sig}_mean"] = vals.mean()
                        record[f"{sig}_std"] = vals.std()
                        record[f"{sig}_median"] = vals.median()

            records.append(record)

    result = pd.DataFrame(records)
    print(f"         {len(result)} aligned profile points computed")

    # Print the key insight: do signals drop BEFORE failure?
    for model in MODELS:
        m = result[result["model"] == model]
        if "cosine_sim_mean" in m.columns and len(m) > 0:
            pre = m[m["relative_frame"] == -1]
            at_fail = m[m["relative_frame"] == 0]
            baseline = m[m["relative_frame"] == -ANTICIPATORY_WINDOW]

            if len(pre) > 0 and len(at_fail) > 0 and len(baseline) > 0:
                drop_before = baseline["cosine_sim_mean"].values[0] - pre["cosine_sim_mean"].values[0]
                drop_at = pre["cosine_sim_mean"].values[0] - at_fail["cosine_sim_mean"].values[0]
                print(f"         {model.upper()}: cosine_sim drops {drop_before:.4f} in pre-failure window, "
                      f"{drop_at:.4f} at failure")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Pre-Failure Window Statistics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_prefailure_statistics(merged: pd.DataFrame) -> pd.DataFrame:
    """For each failed triplet, compute signal slope and acceleration in the pre-failure window."""
    print("\n  [Step 3] Computing pre-failure window statistics ...")

    failed = merged[~merged["survived"]].copy()
    records = []

    signal_cols = {
        "dinov2": ["top1_confidence", "logit_margin", "cosine_sim", "l2_distance", "top5_overlap"],
        "clip": ["top1_confidence", "logit_margin", "cosine_sim", "l2_distance"],
        "detr": ["top_detection_confidence", "objectness", "cosine_sim"],
    }

    for model in MODELS:
        model_data = failed[failed["model"] == model]
        triplets = model_data.groupby(["image", "degradation"])

        for (image, deg), group in triplets:
            fail_frame = group["first_failure_frame"].iloc[0]

            # Pre-failure window: frames from max(1, fail-WINDOW) to fail-1
            pre_window = group[
                (group["frame"] >= max(1, fail_frame - ANTICIPATORY_WINDOW)) &
                (group["frame"] < fail_frame)
            ].sort_values("frame")

            if len(pre_window) < 2:
                continue

            record = {
                "image": image,
                "degradation": deg,
                "model": model,
                "first_failure_frame": fail_frame,
                "window_size": len(pre_window),
            }

            for sig in signal_cols.get(model, []):
                if sig not in pre_window.columns:
                    continue
                vals = pre_window[sig].dropna().values
                if len(vals) < 2:
                    continue

                # Slope: linear regression over the pre-failure window
                x = np.arange(len(vals))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, vals)
                record[f"{sig}_slope"] = slope
                record[f"{sig}_r_squared"] = r_value ** 2
                record[f"{sig}_p_value"] = p_value

                # Total change in window
                record[f"{sig}_total_change"] = vals[-1] - vals[0]

                # Is the trend negative? (signal decreasing = warning sign)
                record[f"{sig}_negative_trend"] = slope < 0

            records.append(record)

    result = pd.DataFrame(records)
    print(f"         {len(result):,} triplet windows analyzed")

    # Summary: what fraction show negative trends?
    for model in MODELS:
        m = result[result["model"] == model]
        if "cosine_sim_negative_trend" in m.columns:
            pct = m["cosine_sim_negative_trend"].mean() * 100
            print(f"         {model.upper()}: {pct:.1f}% show negative cosine_sim trend before failure")
        if "top1_confidence_negative_trend" in m.columns:
            pct = m["top1_confidence_negative_trend"].mean() * 100
            print(f"         {model.upper()}: {pct:.1f}% show negative confidence trend before failure")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Survived vs Failed Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def compare_survived_vs_failed(merged: pd.DataFrame) -> pd.DataFrame:
    """Compare signal trajectories between survived and failed triplets."""
    print("\n  [Step 4] Comparing survived vs failed signal trajectories ...")

    records = []
    signal_cols = ["top1_confidence", "logit_margin", "cosine_sim"]

    for model in MODELS:
        model_data = merged[merged["model"] == model]
        survived = model_data[model_data["survived"] == True]
        failed = model_data[model_data["survived"] == False]

        for frame in range(1, NUM_FRAMES + 1):
            for sig in signal_cols:
                if sig not in model_data.columns:
                    continue

                surv_vals = survived[survived["frame"] == frame][sig].dropna()
                fail_vals = failed[failed["frame"] == frame][sig].dropna()

                if len(surv_vals) < 5 or len(fail_vals) < 5:
                    continue

                # Mann-Whitney U test
                u_stat, p_val = stats.mannwhitneyu(surv_vals, fail_vals, alternative="two-sided")

                # Cohen's d
                pooled_std = np.sqrt(
                    ((len(surv_vals) - 1) * surv_vals.std() ** 2 +
                     (len(fail_vals) - 1) * fail_vals.std() ** 2) /
                    (len(surv_vals) + len(fail_vals) - 2)
                )
                cohens_d = (surv_vals.mean() - fail_vals.mean()) / pooled_std if pooled_std > 0 else 0

                records.append({
                    "model": model,
                    "frame": frame,
                    "signal": sig,
                    "survived_mean": surv_vals.mean(),
                    "failed_mean": fail_vals.mean(),
                    "difference": surv_vals.mean() - fail_vals.mean(),
                    "cohens_d": cohens_d,
                    "p_value": p_val,
                    "significant": p_val < 0.05,
                    "n_survived": len(surv_vals),
                    "n_failed": len(fail_vals),
                })

    result = pd.DataFrame(records)

    # When does the gap between survived and failed become significant?
    for model in MODELS:
        m = result[(result["model"] == model) & (result["signal"] == "cosine_sim")]
        if len(m) == 0:
            continue
        first_sig = m[m["significant"]].sort_values("frame")
        if len(first_sig) > 0:
            first_frame = int(first_sig.iloc[0]["frame"])
            d = first_sig.iloc[0]["cohens_d"]
            print(f"         {model.upper()}: survived vs failed diverges significantly at frame {first_frame} "
                  f"(d={d:.3f})")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Step 5: Predictability Metrics (Lead Time, SNR, AUROC)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_predictability(merged: pd.DataFrame, prefailure: pd.DataFrame) -> pd.DataFrame:
    """Compute lead time, signal-to-noise ratio, and AUROC for failure prediction."""
    print("\n  [Step 5] Computing predictability metrics ...")

    records = []

    signal_cols = {
        "dinov2": ["cosine_sim", "top1_confidence", "logit_margin"],
        "clip": ["cosine_sim", "top1_confidence", "logit_margin"],
        "detr": ["top_detection_confidence"],
    }

    for model in MODELS:
        model_data = merged[merged["model"] == model]
        model_prefail = prefailure[prefailure["model"] == model]

        for sig in signal_cols.get(model, []):
            slope_col = f"{sig}_slope"
            if slope_col not in model_prefail.columns:
                continue

            slopes = model_prefail[slope_col].dropna()
            if len(slopes) < 10:
                continue

            # Lead time: average frames of warning based on when slope becomes negative
            negative_slopes = slopes[slopes < 0]
            lead_time = len(negative_slopes) / len(slopes) * ANTICIPATORY_WINDOW

            # Signal-to-noise ratio: mean slope magnitude / std of slopes
            snr = abs(slopes.mean()) / slopes.std() if slopes.std() > 0 else 0

            # AUROC: Can the signal at frame N-2 predict failure?
            # Binary classification: survived(0) vs failed(1)
            # Feature: signal value at a mid-frame (frame 10)
            survived_vals = model_data[
                (model_data["survived"] == True) & (model_data["frame"] == 10)
            ][sig].dropna()
            failed_vals = model_data[
                (model_data["survived"] == False) & (model_data["frame"] == 10)
            ][sig].dropna()

            auroc = np.nan
            if len(survived_vals) >= 5 and len(failed_vals) >= 5:
                labels = np.concatenate([np.zeros(len(survived_vals)), np.ones(len(failed_vals))])
                scores = np.concatenate([survived_vals.values, failed_vals.values])
                # For signals where higher = more robust (like cosine_sim),
                # we need to flip so higher score = more likely to fail
                try:
                    auroc = roc_auc_score(labels, -scores)
                except ValueError:
                    auroc = np.nan

            records.append({
                "model": model,
                "signal": sig,
                "mean_slope": slopes.mean(),
                "pct_negative_trend": (slopes < 0).mean() * 100,
                "lead_time_frames": round(lead_time, 2),
                "snr": round(snr, 4),
                "auroc_at_frame10": round(auroc, 4) if not np.isnan(auroc) else np.nan,
                "n_triplets": len(slopes),
            })

    result = pd.DataFrame(records)
    print(f"         {len(result)} signal-model combinations evaluated")

    for _, row in result.iterrows():
        auroc_str = f"{row['auroc_at_frame10']:.3f}" if pd.notna(row["auroc_at_frame10"]) else "N/A"
        print(f"         {row['model'].upper():6s} {row['signal']:25s} "
              f"AUROC={auroc_str}  lead={row['lead_time_frames']:.1f}fr  "
              f"neg_trend={row['pct_negative_trend']:.0f}%")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Step 6: Generate Report
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(
    aligned: pd.DataFrame,
    prefailure: pd.DataFrame,
    comparison: pd.DataFrame,
    predictability: pd.DataFrame,
) -> str:
    """Generate the core research findings report."""
    print("\n  [Step 6] Generating Phase 2C report ...")

    lines = []
    lines.append("=" * 70)
    lines.append("AFS-VFM -- Phase 2C: Anticipatory Signal Detection Report")
    lines.append("=" * 70)
    lines.append("")
    lines.append("CORE HYPOTHESIS: Internal model signals degrade measurably")
    lines.append("BEFORE the final prediction changes.")
    lines.append("")

    # ── Verdict ──
    # Use the predictability table's pct_negative_trend (already correctly computed)
    if len(predictability) > 0:
        overall_pct = predictability["pct_negative_trend"].mean()
    else:
        overall_pct = 0

    # Also compute average AUROC
    auroc_vals = predictability["auroc_at_frame10"].dropna()
    avg_auroc = auroc_vals.mean() if len(auroc_vals) > 0 else 0

    lines.append("--- VERDICT ---")
    if overall_pct > 65:
        lines.append(f"  HYPOTHESIS SUPPORTED: {overall_pct:.1f}% of pre-failure windows show")
        lines.append(f"  negative signal trends across all models and signals.")
        lines.append(f"  Average AUROC for failure prediction: {avg_auroc:.3f}")
        lines.append(f"  Internal representations DO anticipate failure.")
    elif overall_pct > 50:
        lines.append(f"  HYPOTHESIS PARTIALLY SUPPORTED: {overall_pct:.1f}% of pre-failure windows")
        lines.append(f"  show negative trends. Average AUROC: {avg_auroc:.3f}")
        lines.append(f"  The effect exists but is not universal.")
    else:
        lines.append(f"  HYPOTHESIS NOT SUPPORTED: Only {overall_pct:.1f}% of pre-failure windows")
        lines.append(f"  show negative trends. Average AUROC: {avg_auroc:.3f}")
        lines.append(f"  Internal signals do not reliably anticipate failure.")
    lines.append("")

    # ── Pre-failure signal trends ──
    lines.append("--- PRE-FAILURE SIGNAL TRENDS (negative = signal drops before failure) ---")
    for model in MODELS:
        m = prefailure[prefailure["model"] == model]
        lines.append(f"\n  {model.upper()}:")
        for col in sorted(m.columns):
            if col.endswith("_slope"):
                sig_name = col.replace("_slope", "")
                slopes = m[col].dropna()
                if len(slopes) > 0:
                    neg_pct = (slopes < 0).mean() * 100
                    lines.append(f"    {sig_name:25s}: mean_slope={slopes.mean():.6f}, "
                                 f"{neg_pct:.0f}% negative")
    lines.append("")

    # ── Survived vs Failed divergence ──
    lines.append("--- SURVIVED vs FAILED: WHEN DO THEY DIVERGE? ---")
    for model in MODELS:
        m = comparison[(comparison["model"] == model) & (comparison["signal"] == "cosine_sim")]
        if len(m) == 0:
            m = comparison[(comparison["model"] == model)]
        sig_frames = m[m["significant"]].sort_values("frame")
        if len(sig_frames) > 0:
            first = sig_frames.iloc[0]
            lines.append(f"  {model.upper()}: significant divergence at frame {int(first['frame'])} "
                         f"(d={first['cohens_d']:.3f}, p={first['p_value']:.2e})")
        else:
            lines.append(f"  {model.upper()}: no significant divergence detected")
    lines.append("")

    # ── Predictability ──
    lines.append("--- PREDICTABILITY METRICS ---")
    lines.append(f"  {'Model':8s} {'Signal':25s} {'AUROC':>7s} {'Lead(fr)':>8s} {'Neg%':>6s} {'SNR':>6s}")
    lines.append("  " + "-" * 65)
    for _, row in predictability.iterrows():
        auroc = f"{row['auroc_at_frame10']:.3f}" if pd.notna(row["auroc_at_frame10"]) else "N/A"
        lines.append(f"  {row['model'].upper():8s} {row['signal']:25s} {auroc:>7s} "
                     f"{row['lead_time_frames']:>8.1f} {row['pct_negative_trend']:>5.0f}% "
                     f"{row['snr']:>6.3f}")
    lines.append("")

    # ── Failure-aligned trajectory ──
    lines.append("--- SIGNAL TRAJECTORY AROUND FAILURE POINT ---")
    lines.append("  (relative_frame: -5=five frames before failure, 0=at failure)")
    for model in MODELS:
        m = aligned[aligned["model"] == model]
        if "cosine_sim_mean" not in m.columns:
            continue
        lines.append(f"\n  {model.upper()} cosine_sim:")
        for _, row in m.sort_values("relative_frame").iterrows():
            if -ANTICIPATORY_WINDOW <= row["relative_frame"] <= 3:
                marker = " <<<" if row["relative_frame"] == 0 else ""
                lines.append(f"    frame {int(row['relative_frame']):+3d}: "
                             f"{row['cosine_sim_mean']:.4f} +/- {row.get('cosine_sim_std', 0):.4f}"
                             f"{marker}")
    lines.append("")

    lines.append("=" * 70)
    lines.append("End of Phase 2C Report")
    lines.append("=" * 70)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  AFS-VFM Phase 2C - Anticipatory Signal Detection")
    print("=" * 70)

    # Load data
    print(f"\n  Loading Phase 2A boundaries: {BOUNDARIES_CSV}")
    boundaries = pd.read_csv(BOUNDARIES_CSV)
    print(f"  Loaded {len(boundaries):,} boundary records")

    print(f"  Loading Phase 2B signals: {SIGNALS_CSV}")
    signals = pd.read_csv(SIGNALS_CSV)
    print(f"  Loaded {len(signals):,} signal records")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Filter boundaries to only images that exist in signals
    signal_images = set(signals["image"].unique())
    boundaries = boundaries[boundaries["image"].isin(signal_images)]
    print(f"  Matched {boundaries['image'].nunique()} images across both datasets")

    # Step 1: Merge
    merged = merge_boundaries_and_signals(boundaries, signals)
    merged.to_csv(os.path.join(OUTPUT_DIR, "merged_signals_boundaries.csv"), index=False)
    print(f"         > Saved merged_signals_boundaries.csv")

    # Step 2: Failure-aligned profiles
    aligned = analyze_failure_aligned_signals(merged)
    aligned.to_csv(os.path.join(OUTPUT_DIR, "failure_aligned_signals.csv"), index=False)
    print(f"         > Saved failure_aligned_signals.csv")

    # Step 3: Pre-failure window stats
    prefailure = compute_prefailure_statistics(merged)
    prefailure.to_csv(os.path.join(OUTPUT_DIR, "prefailure_window_stats.csv"), index=False)
    print(f"         > Saved prefailure_window_stats.csv")

    # Step 4: Survived vs failed comparison
    comparison = compare_survived_vs_failed(merged)
    comparison.to_csv(os.path.join(OUTPUT_DIR, "survived_vs_failed.csv"), index=False)
    print(f"         > Saved survived_vs_failed.csv")

    # Step 5: Predictability metrics
    predictability = compute_predictability(merged, prefailure)
    predictability.to_csv(os.path.join(OUTPUT_DIR, "predictability_metrics.csv"), index=False)
    print(f"         > Saved predictability_metrics.csv")

    # Step 6: Report
    report = generate_report(aligned, prefailure, comparison, predictability)
    report_path = os.path.join(OUTPUT_DIR, "phase2c_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n         > Saved phase2c_report.txt")

    # Print report
    print("\n\n")
    print(report)
    print(f"\n  All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
