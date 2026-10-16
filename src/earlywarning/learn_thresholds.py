"""
AFS-VFM Phase 3B — Learn Optimal Failure Thresholds
=====================================================
Learns per-model, per-signal thresholds from Phase 2C data that maximize
F1-score for binary failure prediction.  Outputs thresholds.json.

Usage:
    python -m src.earlywarning.learn_thresholds
"""

import os, json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHASE2C = os.path.join(ROOT, "results", "phase2c")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Signals to evaluate per model
MODEL_SIGNALS = {
    "dinov2": ["top1_confidence", "logit_margin", "cosine_sim"],
    "clip":   ["top1_confidence", "logit_margin", "cosine_sim"],
    "detr":   ["top_detection_confidence", "cosine_sim"],
}


def _optimal_threshold(values, labels, n_candidates=200):
    """Find threshold that maximises F1 for 'signal < threshold => failure'."""
    lo, hi = np.nanmin(values), np.nanmax(values)
    candidates = np.linspace(lo, hi, n_candidates)
    best_f1, best_thr = 0.0, float(lo)
    for thr in candidates:
        preds = (values < thr).astype(int)
        f = f1_score(labels, preds, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thr = float(thr)
    return best_thr, best_f1


def learn_thresholds():
    """Main entry: learn thresholds from merged_signals_boundaries.csv."""
    merged_path = os.path.join(PHASE2C, "merged_signals_boundaries.csv")

    if not os.path.exists(merged_path):
        print("  ERROR: merged_signals_boundaries.csv not found")
        return {}

    df = pd.read_csv(merged_path)

    # The merged file has columns: image, degradation, frame, model,
    # top1_confidence, logit_margin, cosine_sim, first_failure_frame, survived, etc.
    # We need to create binary failure labels: 1 if the image eventually fails

    thresholds = {}

    for model, signals in MODEL_SIGNALS.items():
        thresholds[model] = {}
        mdata = df[df["model"] == model].copy()

        if mdata.empty:
            print(f"  WARN: No data for {model}")
            continue

        # Binary label: 1 = this image fails at some point, 0 = survived
        mdata["will_fail"] = (~mdata["survived"]).astype(int)

        for signal in signals:
            if signal not in mdata.columns:
                print(f"  WARN: Signal {signal} not in data for {model}")
                continue

            subset = mdata[["will_fail", signal]].dropna()
            if len(subset) < 20:
                continue

            labels = subset["will_fail"].astype(int).values
            values = subset[signal].values

            thr, f1 = _optimal_threshold(values, labels)

            # Warning: 40th percentile of failed-only signal values
            # Critical: 15th percentile of failed-only signal values
            failed_vals = values[labels == 1]
            if len(failed_vals) > 5:
                warning_thr = float(np.percentile(failed_vals, 40))
                critical_thr = float(np.percentile(failed_vals, 15))
            else:
                warning_thr = thr * 0.9
                critical_thr = thr * 0.6

            thresholds[model][signal] = {
                "optimal_threshold": round(thr, 6),
                "f1_at_optimal": round(f1, 4),
                "warning_threshold": round(warning_thr, 6),
                "critical_threshold": round(critical_thr, 6),
                "n_samples": int(len(subset)),
            }

    # Save
    out_path = os.path.join(OUT_DIR, "thresholds.json")
    with open(out_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"  Thresholds saved to: {out_path}")
    return thresholds


if __name__ == "__main__":
    print("=" * 60)
    print("  AFS-VFM Phase 3B — Learning Failure Thresholds")
    print("=" * 60)
    thresholds = learn_thresholds()
    print("\n  Learned thresholds:")
    for model, sigs in thresholds.items():
        print(f"\n  {model.upper()}:")
        for sig, vals in sigs.items():
            print(f"    {sig}: warning={vals['warning_threshold']:.4f}, "
                  f"critical={vals['critical_threshold']:.4f}, "
                  f"F1={vals.get('f1_at_optimal', 0):.3f}")
    print("=" * 60)
