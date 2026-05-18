"""
AFS-VFM Phase 3B — Early Warning System Demo
===============================================
Demonstrates the failure predictor by simulating a degradation trajectory
using real Phase 2C data and showing warnings appearing BEFORE failure.

Usage:
    python -m src.earlywarning.demo
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.earlywarning.failure_predictor import FailurePredictor

PHASE2C = os.path.join(ROOT, "results", "phase2c")
FIG_DIR = os.path.join(ROOT, "results", "figures")


def build_simulated_trajectory(model, n_frames=20):
    """Build a realistic trajectory from Phase 2C failure-aligned signals."""
    df = pd.read_csv(os.path.join(PHASE2C, "failure_aligned_signals.csv"))
    mdata = df[df["model"] == model].sort_values("relative_frame")

    if model == "detr" or mdata.empty or mdata.iloc[0:1]["cosine_sim_mean"].isna().all():
        # Simulate with synthetic data for DETR (no cosine_sim data)
        np.random.seed(42)
        trajectory = []
        for frame in range(1, n_frames + 1):
            t = frame / n_frames
            noise = np.random.normal(0, 0.02)
            conf = max(0.05, 0.85 - 0.6 * t**2 + noise)
            trajectory.append({"top_detection_confidence": conf})
        return trajectory, int(n_frames * 0.7)

    # For DINOv2/CLIP: use the failure-aligned mean trajectory
    # Map relative frames (-5 to +5) to absolute frames around a failure at frame 12
    failure_frame = 12
    trajectory = []

    for frame in range(1, n_frames + 1):
        relative = frame - failure_frame
        row = mdata[mdata["relative_frame"] == relative]
        if not row.empty:
            signals = {}
            for sig in ["top1_confidence", "logit_margin", "cosine_sim"]:
                mean_col = f"{sig}_mean"
                if mean_col in row.columns and not row[mean_col].isna().values[0]:
                    signals[sig] = float(row[mean_col].values[0])
            if signals:
                trajectory.append(signals)
                continue

        # Interpolate for frames outside the data range
        t = frame / n_frames
        if frame < failure_frame - 5:
            # Before the pre-failure window: stable high signals
            first_row = mdata[mdata["relative_frame"] == -5]
            base = {}
            for sig in ["top1_confidence", "logit_margin", "cosine_sim"]:
                mean_col = f"{sig}_mean"
                if mean_col in first_row.columns and not first_row.empty:
                    val = first_row[mean_col].values[0]
                    if not np.isnan(val):
                        base[sig] = float(val) + np.random.normal(0, 0.01)
            trajectory.append(base if base else {"top1_confidence": 0.8})
        else:
            # After failure: low signals with noise
            last_row = mdata[mdata["relative_frame"] == 5]
            base = {}
            for sig in ["top1_confidence", "logit_margin", "cosine_sim"]:
                mean_col = f"{sig}_mean"
                if mean_col in last_row.columns and not last_row.empty:
                    val = last_row[mean_col].values[0]
                    if not np.isnan(val):
                        base[sig] = float(val) + np.random.normal(0, 0.01)
            trajectory.append(base if base else {"top1_confidence": 0.3})

    return trajectory, failure_frame


def run_demo():
    """Run the full demo for DINOv2 and CLIP."""
    print("=" * 70)
    print("  AFS-VFM - Failure Early Warning System Demo")
    print("=" * 70)

    # First, learn thresholds if not already done
    thr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thresholds.json")
    if not os.path.exists(thr_path):
        print("\n  Learning thresholds first...")
        from src.earlywarning.learn_thresholds import learn_thresholds
        learn_thresholds()

    predictor = FailurePredictor(thr_path)

    for model in ["dinov2", "clip"]:
        print(f"\n{'-' * 70}")
        print(f"  Model: {model.upper()}")
        print(f"{'-' * 70}")

        trajectory, actual_failure = build_simulated_trajectory(model)
        result = predictor.assess_trajectory(model, trajectory)

        # Print timeline
        print(f"\n  Frame | Status   | Prob  | Triggering Signals")
        print(f"  ------+----------+-------+--------------------")

        statuses = []
        probs = []
        for r in result["trajectory"]:
            frame = r["frame"]
            status = r["status"]
            prob = r["failure_probability"]
            triggers = ", ".join(r["triggering_signals"]) if r["triggering_signals"] else "-"

            # Color coding in terminal
            marker = "  "
            if frame == actual_failure:
                marker = "<< ACTUAL FAILURE"
            elif status == "CRITICAL":
                marker = "[!]"
            elif status == "WARNING":
                marker = "[*]"
            else:
                marker = "[.]"

            print(f"  {frame:5d} | {status:8s} | {prob:.3f} | {triggers}  {marker}")
            statuses.append(status)
            probs.append(prob)

        # Summary
        fw = result["first_warning_frame"]
        fc = result["first_critical_frame"]
        if fw and fw < actual_failure:
            lead = actual_failure - fw
            print(f"\n  [OK] WARNING appeared at frame {fw}, {lead} frames BEFORE failure!")
        elif fw:
            print(f"\n  [!!] WARNING at frame {fw} (after/at failure frame {actual_failure})")
        else:
            print(f"\n  [XX] No warning detected before failure at frame {actual_failure}")

        # Generate visualization
        _plot_timeline(model, result, actual_failure, trajectory)

    print(f"\n{'=' * 70}")
    print(f"  Demo complete. Visualizations saved to: {FIG_DIR}")
    print(f"{'=' * 70}")


def _plot_timeline(model, result, actual_failure, trajectory):
    """Generate a timeline visualization showing warnings before failure."""
    plt.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#c9d1d9",
        "text.color": "#c9d1d9",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "grid.color": "#21262d",
        "figure.dpi": 200,
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"Early Warning Timeline - {model.upper()}",
                 fontsize=16, fontweight="bold", y=0.95)

    frames = [r["frame"] for r in result["trajectory"]]
    probs = [r["failure_probability"] for r in result["trajectory"]]

    # Top: signal values
    signal_colors = {"top1_confidence": "#3498db", "logit_margin": "#2ecc71",
                     "cosine_sim": "#9b59b6", "top_detection_confidence": "#e67e22"}
    all_signals = set()
    for t in trajectory:
        all_signals.update(t.keys())

    for sig in sorted(all_signals):
        vals = [t.get(sig, np.nan) for t in trajectory]
        if not all(np.isnan(v) for v in vals):
            color = signal_colors.get(sig, "#8b949e")
            ax1.plot(frames[:len(vals)], vals, color=color, linewidth=2,
                     marker="o", markersize=4, label=sig, zorder=3)

    # Mark failure point
    ax1.axvline(x=actual_failure, color="#f85149", linestyle="--", linewidth=2,
                alpha=0.8, label="Actual Failure")

    # Shade warning/critical regions
    for r in result["trajectory"]:
        if r["status"] == "WARNING":
            ax1.axvspan(r["frame"] - 0.5, r["frame"] + 0.5,
                       color="#f0ad4e", alpha=0.15, zorder=1)
        elif r["status"] == "CRITICAL":
            ax1.axvspan(r["frame"] - 0.5, r["frame"] + 0.5,
                       color="#f85149", alpha=0.15, zorder=1)

    ax1.set_ylabel("Signal Value", fontsize=12)
    ax1.legend(fontsize=9, loc="lower left", ncol=2)
    ax1.grid(alpha=0.3)

    # Bottom: status bar
    status_colors = {"OK": "#2ecc71", "WARNING": "#f0ad4e", "CRITICAL": "#f85149"}
    for r in result["trajectory"]:
        color = status_colors.get(r["status"], "#8b949e")
        ax2.barh(0, 1, left=r["frame"] - 0.5, color=color, height=0.6, edgecolor="none")

    ax2.axvline(x=actual_failure, color="#f85149", linestyle="--", linewidth=2, alpha=0.8)
    ax2.set_xlabel("Frame", fontsize=12)
    ax2.set_yticks([0])
    ax2.set_yticklabels(["Status"])
    ax2.set_xlim(0.5, len(frames) + 0.5)

    # Add legend for status
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#2ecc71", label="OK"),
                       Patch(facecolor="#f0ad4e", label="WARNING"),
                       Patch(facecolor="#f85149", label="CRITICAL")]
    ax2.legend(handles=legend_elements, loc="upper right", fontsize=9, ncol=3)

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, f"demo_earlywarning_{model}.png"),
                bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


if __name__ == "__main__":
    run_demo()
