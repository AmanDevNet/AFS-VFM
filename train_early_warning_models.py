"""
AFS-VFM Limitation 8 — Learned Early Warning Classifiers Benchmark & Training
=============================================================================
Trains and cross-validates multivariate machine learning failure predictors
(ElasticNet Logistic Regression, HistGradientBoosting, Random Forest, and Calibrated MLP)
against the heuristic baseline.

Evaluates performance with GroupKFold cross-validation (grouped by image ID to prevent
data leakage) across AUROC, AUPRC, F1-Score, Brier Calibration Score, and Warning Lead Time.
"""

import os
import json
import argparse
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
)

from src.earlywarning.feature_extractor import TemporalFeatureExtractor, CORE_FEATURE_NAMES
from src.earlywarning.learned_predictor import (
    LearnedFailurePredictor,
    build_classifier,
    prepare_training_dataset,
    MODELS_DIR,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
PHASE2C = os.path.join(ROOT, "results", "phase2c")
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def evaluate_heuristic_baseline(
    df: pd.DataFrame,
    thresholds_path: str,
    model_name: str,
    target_col: str = "target_imminent",
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """
    Evaluate the existing scalar threshold heuristic on the exact same dataset.
    """
    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    m_thr = thresholds.get(model_name, {})
    # Pick primary confidence or cosine signal
    primary_sig = "top1_confidence" if "top1_confidence" in m_thr else list(m_thr.keys())[0]
    warn_thr = m_thr[primary_sig]["warning_threshold"]
    crit_thr = m_thr[primary_sig]["critical_threshold"]

    col_val = df.get(primary_sig, df.get("confidence", 0.5)).fillna(0.5).values
    # Invert so higher score = higher failure risk
    risk_scores = 1.0 - col_val
    preds = (col_val <= warn_thr).astype(int)
    y_true = df[target_col].values

    auroc = float(roc_auc_score(y_true, risk_scores))
    auprc = float(average_precision_score(y_true, risk_scores))
    f1 = float(f1_score(y_true, preds, zero_division=0))
    brier = float(brier_score_loss(y_true, np.clip(risk_scores, 0.0, 1.0)))

    return auroc, auprc, f1, brier, y_true, risk_scores


def compute_warning_lead_time_and_far(
    df: pd.DataFrame,
    pred_probs: np.ndarray,
    threshold: float = 0.35,
) -> Tuple[float, float]:
    """
    Calculate Mean Warning Lead Time (frames before failure) and False Alarm Rate (FAR)
    on surviving trajectories.
    """
    data = df.copy()
    data["prob"] = pred_probs
    data["alert"] = (data["prob"] >= threshold).astype(int)

    lead_times = []
    surviving_alerts = 0
    total_surviving = 0

    group_cols = [c for c in ["image", "degradation"] if c in data.columns]
    for _, group in data.groupby(group_cols, sort=False):
        survived = bool(group["survived"].iloc[0])
        alerts = group[group["alert"] == 1]

        if survived:
            total_surviving += 1
            if not alerts.empty:
                surviving_alerts += 1
        else:
            fff = float(group["first_failure_frame"].iloc[0])
            if not alerts.empty:
                first_warn = float(alerts["frame"].iloc[0])
                if first_warn <= fff:
                    lead_times.append(fff - first_warn)
                else:
                    lead_times.append(0.0)
            else:
                lead_times.append(0.0)

    mean_lead_time = float(np.mean(lead_times)) if lead_times else 0.0
    far = float(surviving_alerts / total_surviving) if total_surviving > 0 else 0.0
    return mean_lead_time, far


def train_and_benchmark_all_models(
    csv_path: Optional[str] = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Main training and comparative benchmarking suite.
    """
    if csv_path is None:
        csv_path = os.path.join(PHASE2C, "merged_signals_boundaries.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Benchmark data file not found at: {csv_path}")

    print("=" * 72)
    print("  AFS-VFM Limitation 8: Learned Early Warning Classifiers Benchmark")
    print("=" * 72)
    print(f"Loading dataset: {csv_path}")
    raw_df = pd.read_csv(csv_path)

    thresholds_path = os.path.join(ROOT, "src", "earlywarning", "thresholds.json")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    models_to_test = [m for m in ["dinov2", "clip", "detr"] if m in raw_df["model"].unique()]
    classifiers = [
        "logistic_regression",
        "gradient_boosting",
        "random_forest",
        "mlp",
    ]

    all_results = []
    roc_data_for_plot = {}

    for model_name in models_to_test:
        print(f"\n>>> Benchmarking Model Architecture: {model_name.upper()} <<<")
        sub_df = raw_df[raw_df["model"] == model_name].copy()
        aug_df, feature_cols, target_col, group_col = prepare_training_dataset(sub_df)

        X = aug_df[feature_cols].fillna(0.0).values
        y = aug_df[target_col].values
        groups = aug_df[group_col].values

        # 1. Baseline Heuristic
        h_auroc, h_auprc, h_f1, h_brier, h_y, h_scores = evaluate_heuristic_baseline(
            aug_df, thresholds_path, model_name, target_col
        )
        h_lead, h_far = compute_warning_lead_time_and_far(aug_df, h_scores, threshold=0.5)

        all_results.append({
            "model": model_name,
            "classifier": "Heuristic (Baseline)",
            "auroc": round(h_auroc, 4),
            "auprc": round(h_auprc, 4),
            "f1_score": round(h_f1, 4),
            "brier_score": round(h_brier, 4),
            "lead_time_frames": round(h_lead, 2),
            "false_alarm_rate": round(h_far, 4),
        })

        # Save for ROC plotting
        roc_data_for_plot[f"{model_name}_heuristic"] = (h_y, h_scores, "Heuristic Baseline")

        # 2. Machine Learning Classifiers with GroupKFold
        gkf = GroupKFold(n_splits=n_splits)

        for ctype in classifiers:
            aurocs = []
            auprcs = []
            f1s = []
            briers = []
            oof_preds = np.zeros(len(y), dtype=np.float32)

            for train_idx, test_idx in gkf.split(X, y, groups=groups):
                X_tr, y_tr = X[train_idx], y[train_idx]
                X_te, y_te = X[test_idx], y[test_idx]

                pipe = build_classifier(ctype, random_state=random_state)
                pipe.fit(X_tr, y_tr)

                probs = pipe.predict_proba(X_te)[:, 1]
                preds = (probs >= 0.5).astype(int)

                oof_preds[test_idx] = probs
                aurocs.append(roc_auc_score(y_te, probs))
                auprcs.append(average_precision_score(y_te, probs))
                f1s.append(f1_score(y_te, preds, zero_division=0))
                briers.append(brier_score_loss(y_te, probs))

            lead_time, far = compute_warning_lead_time_and_far(aug_df, oof_preds, threshold=0.35)

            all_results.append({
                "model": model_name,
                "classifier": ctype.replace("_", " ").title(),
                "auroc": round(float(np.mean(aurocs)), 4),
                "auprc": round(float(np.mean(auprcs)), 4),
                "f1_score": round(float(np.mean(f1s)), 4),
                "brier_score": round(float(np.mean(briers)), 4),
                "lead_time_frames": round(lead_time, 2),
                "false_alarm_rate": round(far, 4),
            })

            roc_data_for_plot[f"{model_name}_{ctype}"] = (y, oof_preds, ctype.replace("_", " ").title())

            # Fit production model on all data and save
            prod_pipe = build_classifier(ctype, random_state=random_state)
            prod_pipe.fit(X, y)

            predictor = LearnedFailurePredictor(
                classifier=prod_pipe,
                feature_names=feature_cols,
                model_name=model_name,
                classifier_type=ctype,
            )
            save_name = f"{model_name}_{ctype}_predictor.joblib"
            predictor.save(os.path.join(MODELS_DIR, save_name))

            # Also set the best non-linear tree model as the default production artifact
            if ctype == "gradient_boosting":
                predictor.save(os.path.join(MODELS_DIR, f"{model_name}_learned_predictor.joblib"))

    # Convert results to DataFrame
    res_df = pd.DataFrame(all_results)
    csv_out = os.path.join(RESULTS_DIR, "early_warning_model_comparison.csv")
    res_df.to_csv(csv_out, index=False)
    print(f"\n[OK] Model comparison metrics saved to: {csv_out}")

    # Generate Figures
    _generate_comparison_figures(roc_data_for_plot, res_df)

    # Print summary table to terminal
    print("\n" + "=" * 90)
    print("  EARLY WARNING CLASSIFIERS: 5-FOLD GROUPKFOLD BENCHMARK RESULTS")
    print("=" * 90)
    print(res_df.to_string(index=False))
    print("=" * 90)

    return res_df


def _generate_comparison_figures(roc_data: Dict[str, Any], res_df: pd.DataFrame) -> None:
    """
    Generate ROC curves and Lead Time comparison plots.
    """
    models = ["dinov2", "clip", "detr"]

    # 1. Cross-Model ROC Curves Figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    palette = {
        "Heuristic Baseline": "#7f7f7f",
        "Logistic Regression": "#1f77b4",
        "Gradient Boosting": "#2ca02c",
        "Random Forest": "#ff7f0e",
        "Mlp": "#9467bd",
    }

    for ax_idx, model_name in enumerate(models):
        ax = axes[ax_idx]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")

        for key, (y_true, y_scores, label) in roc_data.items():
            if key.startswith(model_name):
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                auc_val = roc_auc_score(y_true, y_scores)
                color = palette.get(label, "#333333")
                lw = 2.5 if "Gradient Boosting" in label else 1.5
                ax.plot(fpr, tpr, label=f"{label} ({auc_val:.3f})", color=color, linewidth=lw)

        ax.set_title(f"{model_name.upper()} Early Warning ROC", fontsize=13, fontweight="bold")
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    roc_fig_path = os.path.join(FIGURES_DIR, "early_warning_roc_curves.png")
    plt.savefig(roc_fig_path, dpi=300)
    plt.close()
    print(f"[OK] ROC curve figure saved to: {roc_fig_path}")

    # 2. Warning Lead Time Comparison Bar Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot_lead = res_df.pivot(index="model", columns="classifier", values="lead_time_frames")
    pivot_lead.plot(kind="bar", ax=ax, width=0.8, colormap="tab10")

    ax.set_title("Warning Lead Time Comparison (Frames in Advance of Failure)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Model Architecture", fontsize=11)
    ax.set_ylabel("Mean Lead Time (Frames)", fontsize=11)
    ax.set_xticklabels([m.upper() for m in pivot_lead.index], rotation=0)
    ax.legend(title="Predictor", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    lead_fig_path = os.path.join(FIGURES_DIR, "early_warning_lead_time_comparison.png")
    plt.savefig(lead_fig_path, dpi=300)
    plt.close()
    print(f"[OK] Lead time comparison figure saved to: {lead_fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and benchmark learned early warning models for AFS-VFM")
    parser.add_argument("--csv", type=str, default=None, help="Path to merged_signals_boundaries.csv")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of GroupKFold splits (default: 5)")
    args = parser.parse_args()

    train_and_benchmark_all_models(csv_path=args.csv, n_splits=args.n_splits)
