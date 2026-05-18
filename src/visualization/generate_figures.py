"""
AFS-VFM Phase 3A — Publication-Quality Figure Generation
=========================================================
Generates all 9 figures for the research paper from Phase 2 CSV data.

Usage:
    python -m src.visualization.generate_figures
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, auc

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHASE2A = os.path.join(ROOT, "results", "phase2a")
PHASE2C = os.path.join(ROOT, "results", "phase2c")
PHASE2D = os.path.join(ROOT, "results", "phase2d")
FIG_DIR = os.path.join(ROOT, "results", "figures")

# ── Style ────────────────────────────────────────────────────────────────
MODEL_COLORS = {"dinov2": "#2ecc71", "clip": "#3498db", "detr": "#e74c3c"}
MODEL_LABELS = {"dinov2": "DINOv2", "clip": "CLIP", "detr": "DETR"}
DEG_ORDER = ["blur", "lighting", "occlusion", "scale", "viewpoint"]
DEG_LABELS = {"blur": "Motion Blur", "lighting": "Lighting", "occlusion": "Occlusion",
              "scale": "Scale", "viewpoint": "Viewpoint"}

def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#c9d1d9",
        "text.color": "#c9d1d9",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "grid.color": "#21262d",
        "grid.alpha": 0.6,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.3,
    })

# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1: Robustness Heatmap
# ═══════════════════════════════════════════════════════════════════════
def fig1_robustness_heatmap():
    print("  [Fig 1] Robustness Heatmap ...")
    df = pd.read_csv(os.path.join(PHASE2A, "failure_statistics.csv"))
    pivot = df.pivot(index="model", columns="degradation", values="survival_rate_pct")
    pivot = pivot.reindex(index=["dinov2", "clip", "detr"], columns=DEG_ORDER)

    fig, ax = plt.subplots(figsize=(10, 4))
    data = pivot.values
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=60)

    ax.set_xticks(range(5))
    ax.set_xticklabels([DEG_LABELS[d] for d in DEG_ORDER], fontsize=12)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["DINOv2", "CLIP", "DETR"], fontsize=13, fontweight="bold")

    for i in range(3):
        for j in range(5):
            val = data[i, j]
            color = "white" if val < 25 else "#0d1117"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Survival Rate (%)", fontsize=11)
    ax.set_title("Model Robustness Under Visual Degradation", fontsize=16, pad=15)
    fig.savefig(os.path.join(FIG_DIR, "fig1_robustness_heatmap.png"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2: Failure Boundary Distributions
# ═══════════════════════════════════════════════════════════════════════
def fig2_failure_distributions():
    print("  [Fig 2] Failure Boundary Distributions ...")
    df = pd.read_csv(os.path.join(PHASE2A, "failure_boundaries.csv"))
    failed = df[df["survived"] == False].copy()

    fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)
    fig.suptitle("Distribution of First-Failure Frames by Degradation Type",
                 fontsize=16, fontweight="bold", y=1.02)

    for idx, deg in enumerate(DEG_ORDER):
        ax = axes[idx]
        deg_data = failed[failed["degradation"] == deg]
        positions = []
        data_groups = []
        colors = []
        for mi, model in enumerate(["dinov2", "clip", "detr"]):
            vals = deg_data[deg_data["model"] == model]["first_failure_frame"].dropna()
            if len(vals) > 0:
                data_groups.append(vals.values)
                positions.append(mi)
                colors.append(MODEL_COLORS[model])

        if data_groups:
            vp = ax.violinplot(data_groups, positions=positions, showmedians=True,
                               showextrema=False, widths=0.7)
            for pi, body in enumerate(vp["bodies"]):
                body.set_facecolor(colors[pi])
                body.set_alpha(0.6)
                body.set_edgecolor(colors[pi])
            vp["cmedians"].set_color("white")
            vp["cmedians"].set_linewidth(2)

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["DINOv2", "CLIP", "DETR"], fontsize=9, rotation=30)
        ax.set_title(DEG_LABELS[deg], fontsize=12)
        ax.set_ylim(0, 21)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("First Failure Frame", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_failure_distributions.png"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3: Signal Trajectory Around Failure (KEY FIGURE)
# ═══════════════════════════════════════════════════════════════════════
def fig3_signal_trajectory():
    print("  [Fig 3] Signal Trajectory Around Failure (KEY FIGURE) ...")
    df = pd.read_csv(os.path.join(PHASE2C, "failure_aligned_signals.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
    fig.suptitle("Internal Signal Trajectory Around Failure Point",
                 fontsize=16, fontweight="bold", y=1.02)

    signals = [
        ("cosine_sim", "Cosine Similarity"),
        ("top1_confidence", "Top-1 Confidence"),
        ("logit_margin", "Logit Margin"),
    ]

    for si, (signal, label) in enumerate(signals):
        ax = axes[si]
        mean_col = f"{signal}_mean"
        std_col = f"{signal}_std"

        for model in ["dinov2", "clip"]:
            mdata = df[df["model"] == model].sort_values("relative_frame")
            if mean_col not in mdata.columns or mdata[mean_col].isna().all():
                continue
            x = mdata["relative_frame"].values
            y = mdata[mean_col].values
            s = mdata[std_col].values
            color = MODEL_COLORS[model]
            ax.plot(x, y, color=color, linewidth=2.5, marker="o", markersize=5,
                    label=MODEL_LABELS[model], zorder=3)
            ax.fill_between(x, y - s, y + s, color=color, alpha=0.15, zorder=2)

        # Mark failure point
        ax.axvline(x=0, color="#f85149", linestyle="--", linewidth=1.5, alpha=0.8, label="Failure Point")
        ax.axvspan(-5, 0, color="#f85149", alpha=0.05)

        ax.set_xlabel("Relative Frame (0 = failure)", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=13)
        ax.legend(fontsize=9, loc="lower left")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_signal_trajectory.png"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 4: AUROC Comparison Bar Chart
# ═══════════════════════════════════════════════════════════════════════
def fig4_auroc_comparison():
    print("  [Fig 4] AUROC Comparison ...")
    df = pd.read_csv(os.path.join(PHASE2C, "predictability_metrics.csv"))

    fig, ax = plt.subplots(figsize=(10, 5))
    y_positions = []
    labels = []
    colors = []
    values = []

    for i, (_, row) in enumerate(df.iterrows()):
        y_positions.append(i)
        labels.append(f"{MODEL_LABELS[row['model']]} — {row['signal']}")
        colors.append(MODEL_COLORS[row["model"]])
        values.append(row["auroc_at_frame10"])

    bars = ax.barh(y_positions, values, color=colors, height=0.6, edgecolor="none", alpha=0.85)

    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=11, fontweight="bold")

    ax.axvline(x=0.5, color="#8b949e", linestyle=":", linewidth=1, alpha=0.6, label="Random (0.5)")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("AUROC", fontsize=12)
    ax.set_xlim(0, 1.0)
    ax.set_title("Failure Prediction: AUROC by Model & Signal", fontsize=14, pad=12)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_auroc_comparison.png"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 5: ROC Curves
# ═══════════════════════════════════════════════════════════════════════
def fig5_roc_curves():
    print("  [Fig 5] ROC Curves ...")
    merged_path = os.path.join(PHASE2C, "merged_signals_boundaries.csv")
    if not os.path.exists(merged_path):
        print("    SKIP: merged_signals_boundaries.csv not found")
        return

    df = pd.read_csv(merged_path)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], ":", color="#8b949e", linewidth=1, label="Random (AUC=0.50)")

    roc_configs = [
        ("dinov2", "dinov2_top1_confidence", "DINOv2 — confidence"),
        ("dinov2", "dinov2_logit_margin", "DINOv2 — logit margin"),
        ("clip", "clip_top1_confidence", "CLIP — confidence"),
        ("clip", "clip_logit_margin", "CLIP — logit margin"),
        ("detr", "detr_top_detection_confidence", "DETR — det. confidence"),
    ]

    linestyles = ["-", "--", "-", "--", "-"]

    for (model, signal_col, label), ls in zip(roc_configs, linestyles):
        fail_col = f"{model}_failure" if f"{model}_failure" in df.columns else None
        # Try to identify failure column
        if fail_col is None:
            for c in df.columns:
                if model in c and "fail" in c.lower():
                    fail_col = c
                    break
        if fail_col is None or signal_col not in df.columns:
            continue

        subset = df[[fail_col, signal_col]].dropna()
        if len(subset) < 10:
            continue

        y_true = subset[fail_col].astype(int).values
        # For failure prediction: low signal = failure, so negate
        scores = -subset[signal_col].values

        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=MODEL_COLORS[model], linewidth=2,
                linestyle=ls, label=f"{label} (AUC={roc_auc:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves: Predicting Model Failure from Internal Signals",
                 fontsize=14, pad=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_roc_curves.png"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 6: Survived vs Failed Divergence Timeline
# ═══════════════════════════════════════════════════════════════════════
def fig6_divergence_timeline():
    print("  [Fig 6] Survived vs Failed Divergence ...")
    df = pd.read_csv(os.path.join(PHASE2C, "survived_vs_failed.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("When Do Survived and Failed Images Diverge?",
                 fontsize=16, fontweight="bold", y=1.02)

    for mi, model in enumerate(["dinov2", "clip"]):
        ax = axes[mi]
        mdata = df[(df["model"] == model) & (df["signal"] == "top1_confidence")]

        frames = mdata["frame"].values
        surv = mdata["survived_mean"].values
        fail = mdata["failed_mean"].values
        sig = mdata["significant"].values

        ax.plot(frames, surv, color="#2ecc71", linewidth=2.5, marker="o",
                markersize=4, label="Survived", zorder=3)
        ax.plot(frames, fail, color="#e74c3c", linewidth=2.5, marker="s",
                markersize=4, label="Failed", zorder=3)

        # Shade significant region
        sig_frames = frames[sig == True] if sig.dtype == bool else frames[sig.astype(str) == "True"]
        if len(sig_frames) > 0:
            ax.axvspan(sig_frames[0], frames[-1], color="#f85149", alpha=0.06,
                       label=f"Significant (from frame {int(sig_frames[0])})")

        ax.set_xlabel("Frame", fontsize=11)
        ax.set_ylabel("Top-1 Confidence", fontsize=11)
        ax.set_title(f"{MODEL_LABELS[model]}", fontsize=13)
        ax.legend(fontsize=9, loc="lower left")
        ax.set_xlim(1, 20)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig6_divergence_timeline.png"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 7: Ensemble Survival Bar Chart
# ═══════════════════════════════════════════════════════════════════════
def fig7_ensemble_benefit():
    print("  [Fig 7] Ensemble Benefit ...")
    df = pd.read_csv(os.path.join(PHASE2D, "ensemble_complementarity.csv"))
    avg = df.groupby("strategy")["survival_rate"].mean().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    strategy_colors = {
        "detr_alone": MODEL_COLORS["detr"],
        "clip_alone": MODEL_COLORS["clip"],
        "dinov2_alone": MODEL_COLORS["dinov2"],
        "clip+detr_ensemble": "#9b59b6",
        "dinov2+detr_ensemble": "#e67e22",
        "dinov2+clip_ensemble": "#1abc9c",
        "all_3_ensemble": "#f1c40f",
    }
    strategy_labels = {
        "detr_alone": "DETR Alone",
        "clip_alone": "CLIP Alone",
        "dinov2_alone": "DINOv2 Alone",
        "clip+detr_ensemble": "CLIP + DETR",
        "dinov2+detr_ensemble": "DINOv2 + DETR",
        "dinov2+clip_ensemble": "DINOv2 + CLIP",
        "all_3_ensemble": "All 3 Models",
    }

    y_pos = range(len(avg))
    colors = [strategy_colors.get(s, "#8b949e") for s in avg.index]
    labels = [strategy_labels.get(s, s) for s in avg.index]

    bars = ax.barh(y_pos, avg.values, color=colors, height=0.6, edgecolor="none", alpha=0.85)
    for bar, val in zip(bars, avg.values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=11, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Average Survival Rate (%)", fontsize=12)
    ax.set_title("Ensemble Strategy: Combined Survival Rates", fontsize=14, pad=12)
    ax.set_xlim(0, 70)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig7_ensemble_benefit.png"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 8: Failure Correlation Matrix
# ═══════════════════════════════════════════════════════════════════════
def fig8_failure_correlation():
    print("  [Fig 8] Failure Correlation ...")
    df = pd.read_csv(os.path.join(PHASE2D, "failure_correlation_matrix.csv"))
    avg = df.groupby("model_pair")[["jaccard_similarity", "overlap_pct"]].mean()

    # Build 3x3 matrix
    models = ["dinov2", "clip", "detr"]
    matrix = np.eye(3)
    pair_map = {
        ("dinov2", "clip"): "dinov2_vs_clip",
        ("dinov2", "detr"): "dinov2_vs_detr",
        ("clip", "detr"): "clip_vs_detr",
    }
    for (m1, m2), pair_name in pair_map.items():
        if pair_name in avg.index:
            val = avg.loc[pair_name, "jaccard_similarity"]
            i, j = models.index(m1), models.index(m2)
            matrix[i, j] = val
            matrix[j, i] = val

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0.3, vmax=1.0)

    ax.set_xticks(range(3))
    ax.set_xticklabels(["DINOv2", "CLIP", "DETR"], fontsize=12, fontweight="bold")
    ax.set_yticks(range(3))
    ax.set_yticklabels(["DINOv2", "CLIP", "DETR"], fontsize=12, fontweight="bold")

    for i in range(3):
        for j in range(3):
            color = "white" if matrix[i, j] > 0.7 else "#0d1117"
            label = "1.00" if i == j else f"{matrix[i, j]:.3f}"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Jaccard Similarity", fontsize=11)
    ax.set_title("Failure Correlation Between Models", fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig8_failure_correlation.png"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 9: Architecture Comparison Radar Chart
# ═══════════════════════════════════════════════════════════════════════
def fig9_architecture_radar():
    print("  [Fig 9] Architecture Radar ...")
    df = pd.read_csv(os.path.join(PHASE2D, "architecture_comparison.csv"))

    categories = ["Robustness\n(survival %)", "Self-Awareness\n(neg trend %)",
                   "Best AUROC", "Mean Failure\nFrame", "Survival\nRate"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
    ax.set_facecolor("#161b22")
    fig.patch.set_facecolor("#0d1117")

    for _, row in df.iterrows():
        model = row["model"]
        values = [
            row["overall_survival_pct"] / 40,  # normalize to ~1
            row["avg_negative_trend_pct"] / 100,
            row["best_auroc"],
            row["mean_failure_frame"] / 20,
            row["overall_survival_pct"] / 40,
        ]
        values += values[:1]  # close polygon
        color = MODEL_COLORS[model]
        ax.plot(angles, values, color=color, linewidth=2.5, label=MODEL_LABELS[model])
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, color="#c9d1d9")
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8, color="#8b949e")
    ax.spines["polar"].set_color("#30363d")
    ax.grid(color="#21262d", alpha=0.6)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.set_title("Architecture Comparison\n", fontsize=15, fontweight="bold",
                 color="#c9d1d9", pad=20)
    fig.savefig(os.path.join(FIG_DIR, "fig9_architecture_radar.png"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  AFS-VFM Phase 3A — Generating Publication Figures")
    print("=" * 60)

    setup_style()
    os.makedirs(FIG_DIR, exist_ok=True)

    fig1_robustness_heatmap()
    fig2_failure_distributions()
    fig3_signal_trajectory()
    fig4_auroc_comparison()
    fig5_roc_curves()
    fig6_divergence_timeline()
    fig7_ensemble_benefit()
    fig8_failure_correlation()
    fig9_architecture_radar()

    print(f"\n  All figures saved to: {FIG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
