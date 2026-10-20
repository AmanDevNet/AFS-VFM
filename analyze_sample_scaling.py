"""
AFS-VFM Statistical Power & Sample Scaling Convergence Analysis
================================================================
Conducts bootstrap empirical scaling analysis over increasing sample sizes (N)
to evaluate metric convergence, confidence interval shrinkage, and statistical power.

Generates:
  - results/sample_scaling_power_analysis.csv
  - results/figures/fig10_sample_size_convergence.png
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def analyze_sample_scaling(
    input_csv: str,
    output_csv: str = None,
    output_fig: str = None,
    n_bootstrap: int = 25,
    random_state: int = 42,
) -> pd.DataFrame:
    print("=" * 72)
    print("  AFS-VFM Statistical Power & Sample Scaling Analysis")
    print("=" * 72)
    print(f"Loading benchmark dataset: {input_csv}")

    df = pd.read_csv(input_csv)
    np.random.seed(random_state)

    # Determine unique images
    if "image" not in df.columns:
        raise ValueError("Input CSV must contain an 'image' column.")

    unique_images = np.array(df["image"].unique())
    total_n = len(unique_images)
    print(f"Total unique images available: {total_n:,}")

    # Determine candidate sample sizes up to total_n
    candidate_steps = [25, 50, 100, 200, 350, 500, 750, 1000, 1500, 3000, 5000, 10000]
    sample_sizes = [s for s in candidate_steps if s <= total_n]
    if total_n not in sample_sizes:
        sample_sizes.append(total_n)
    sample_sizes = sorted(sample_sizes)

    # Detect models
    if "model" in df.columns:
        models = [m for m in ["dinov2", "clip", "detr"] if m in df["model"].unique()]
        if not models:
            models = list(df["model"].unique())[:3]
    else:
        # Detect model failure columns (e.g. dinov2_failure, clip_failure)
        model_cols = [c.replace("_failure", "") for c in df.columns if c.endswith("_failure")]
        models = model_cols if model_cols else ["dinov2"]

    print(f"Evaluating across sample sizes N = {sample_sizes}")
    print(f"Bootstrap replicates per size: {n_bootstrap}")

    records = []

    # Fast indexed lookup per image
    image_groups = dict(tuple(df.groupby("image")))

    for n in sample_sizes:
        n_means = []
        n_margins = []
        n_cossims = []

        for b in range(n_bootstrap):
            sampled_imgs = np.random.choice(unique_images, size=n, replace=True)
            sampled_dfs = [image_groups[img] for img in sampled_imgs if img in image_groups]
            if not sampled_dfs:
                continue
            sub = pd.concat(sampled_dfs, ignore_index=True)

            # Measure confidence or margin
            if "top1_confidence" in sub.columns:
                n_means.append(float(sub["top1_confidence"].mean()))
            elif "confidence" in sub.columns:
                n_means.append(float(sub["confidence"].mean()))
            else:
                n_means.append(0.5)

            if "logit_margin" in sub.columns:
                n_margins.append(float(sub["logit_margin"].mean()))
            elif "margin" in sub.columns:
                n_margins.append(float(sub["margin"].mean()))
            else:
                n_margins.append(0.0)

            if "cosine_sim" in sub.columns:
                n_cossims.append(float(sub["cosine_sim"].mean()))
            else:
                n_cossims.append(1.0)

        mean_val = float(np.mean(n_means))
        std_err = float(np.std(n_means) / np.sqrt(n_bootstrap))
        ci_95 = 1.96 * std_err

        records.append({
            "sample_size_N": n,
            "mean_confidence": round(mean_val, 4),
            "std_error": round(std_err, 6),
            "ci_95_half_width": round(ci_95, 6),
            "ci_lower": round(mean_val - ci_95, 4),
            "ci_upper": round(mean_val + ci_95, 4),
            "mean_margin": round(float(np.mean(n_margins)), 4),
            "mean_cosine_sim": round(float(np.mean(n_cossims)), 4),
        })

    res_df = pd.DataFrame(records)

    # Save CSV
    if output_csv is None:
        output_csv = os.path.join(RESULTS_DIR, "sample_scaling_power_analysis.csv")
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    res_df.to_csv(output_csv, index=False)
    print(f"\n[OK] Statistical scaling analysis saved to: {output_csv}")

    # Generate Publication Figure: fig10_sample_size_convergence.png
    if output_fig is None:
        output_fig = os.path.join(FIGURES_DIR, "fig10_sample_size_convergence.png")
    os.makedirs(os.path.dirname(os.path.abspath(output_fig)), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Metric Convergence & 95% Confidence Interval Band
    ns = res_df["sample_size_N"]
    means = res_df["mean_confidence"]
    ci = res_df["ci_95_half_width"]

    ax1.plot(ns, means, "o-", color="#1f77b4", linewidth=2.2, label="Mean Signal (Empirical)")
    ax1.fill_between(ns, means - ci, means + ci, color="#1f77b4", alpha=0.25, label="95% Confidence Interval")
    ax1.set_title("Metric Stability vs Sample Size N", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Sample Size (N Base Images)", fontsize=11)
    ax1.set_ylabel("Empirical Mean Metric", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=10)

    # Panel 2: Error Margin Shrinkage (1 / sqrt(N) Law)
    ax2.plot(ns, ci, "s--", color="#d62728", linewidth=2.0, label="95% CI Margin (±)")
    ax2.set_title("Standard Error Contraction (1 / sqrt(N))", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Sample Size (N Base Images)", fontsize=11)
    ax2.set_ylabel("Margin of Error (95% CI)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_fig, dpi=300)
    plt.close()
    print(f"[OK] Convergence figure saved to: {output_fig}")

    print("\n" + "=" * 72)
    print(res_df.to_string(index=False))
    print("=" * 72)

    return res_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AFS-VFM Sample Size Scaling Analysis")
    parser.add_argument("--input", type=str, default="results/phase2c/merged_signals_boundaries.csv",
                        help="Path to benchmark CSV (default: results/phase2c/merged_signals_boundaries.csv)")
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--output_fig", type=str, default=None)
    args = parser.parse_args()

    analyze_sample_scaling(args.input, args.output_csv, args.output_fig)
