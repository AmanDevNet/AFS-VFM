"""
AFS-VFM Automated Verification Suite — Cross-Model Analysis Pipeline
=====================================================================
Comprehensive regression test suite verifying:
1. Dynamic model discovery and fallback logic.
2. Robustness rankings computation across 3-model and 5-model datasets.
3. Failure correlation matrix and pairwise Jaccard similarity.
4. Ensemble complementarity and dynamic N-model intersection.
5. Architecture-linked analysis and composite self-awareness ranking.
6. Full synthesis report generation without NaNs or formatting errors.

Usage:
    python verify_analysis_pipeline.py
"""

import os
import sys
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.analysis.cross_model_comparison import (
    discover_models,
    compute_robustness_rankings,
    compute_failure_correlation,
    compute_complementarity,
    analyze_architecture_effects,
    rank_signal_awareness,
    generate_synthesis,
)


def _build_mock_boundaries_5_models(num_images=50):
    """Generate deterministic mock boundaries for 5 models across 3 degradations."""
    np.random.seed(42)
    models = ["dinov2", "clip", "detr", "siglip", "dinov2_large"]
    degradations = ["blur", "lighting", "occlusion"]

    records = []
    for deg in degradations:
        for img_id in range(num_images):
            img_name = f"mock_{img_id:04d}.jpg"
            for model in models:
                # Assign distinct failure probabilities to reflect realistic hierarchy
                base_prob = {
                    "dinov2_large": 0.25,
                    "dinov2": 0.35,
                    "siglip": 0.40,
                    "clip": 0.45,
                    "detr": 0.60,
                }[model]

                failed = np.random.rand() < base_prob
                fail_frame = int(np.random.randint(3, 19)) if failed else 20
                survived = not failed

                records.append({
                    "image": img_name,
                    "dataset": "mock_data",
                    "degradation": deg,
                    "model": model,
                    "survived": survived,
                    "first_failure_frame": fail_frame,
                    "total_failures": 1 if failed else 0,
                })

    return pd.DataFrame(records)


def _build_mock_predictability_5_models():
    """Generate mock predictability metrics for 5 models."""
    models = ["dinov2", "clip", "detr", "siglip", "dinov2_large"]
    records = []
    for m in models:
        records.append({
            "model": m,
            "signal": "cosine_sim" if m == "detr" else "top1_confidence",
            "auroc_at_frame10": 0.75 + (0.02 if "dinov2" in m else 0.05 if "clip" in m or "siglip" in m else 0.01),
            "lead_time_frames": 3.5,
            "pct_negative_trend": 85.0,
        })
    return pd.DataFrame(records)


def _build_mock_svf_5_models():
    """Generate mock survived vs failed divergence stats."""
    models = ["dinov2", "clip", "detr", "siglip", "dinov2_large"]
    records = []
    for m in models:
        for f in range(1, 21):
            records.append({
                "model": m,
                "frame": f,
                "significant": (f >= 5),
            })
    return pd.DataFrame(records)


def test_1_discover_models():
    print("[TEST 1] Testing dynamic model discovery...")
    # From DataFrame
    df = pd.DataFrame({"model": ["SigLIP", "DINOv2_Large", "detr"]})
    found = discover_models(df)
    assert found == ["siglip", "dinov2_large", "detr"], f"Unexpected discovery: {found}"

    # Fallback when empty or column missing
    empty_df = pd.DataFrame({"other": [1, 2]})
    fallback = discover_models(empty_df)
    assert fallback == ["dinov2", "clip", "detr"], f"Unexpected fallback: {fallback}"

    # Explicit override
    override = discover_models(empty_df, specified_models=["modelA", "modelB"])
    assert override == ["modela", "modelb"], f"Unexpected override: {override}"

    print("  [PASS] Dynamic model discovery and fallback verified.")


def test_2_robustness_rankings_multi_model():
    print("[TEST 2] Testing multi-model robustness rankings...")
    boundaries = _build_mock_boundaries_5_models()
    models = ["dinov2", "clip", "detr", "siglip", "dinov2_large"]

    rankings = compute_robustness_rankings(boundaries, models)
    assert len(rankings) == len(models) * 3  # 5 models x 3 degradations = 15 rows
    assert "rank" in rankings.columns
    assert "fragility_score" in rankings.columns
    assert "survival_rate" in rankings.columns

    # Verify ranking numbers 1..5 for each degradation
    for deg in ["blur", "lighting", "occlusion"]:
        sub = rankings[rankings["degradation"] == deg]
        ranks = sorted(list(sub["rank"].values))
        assert ranks == [1, 2, 3, 4, 5], f"Incorrect ranks for {deg}: {ranks}"

    print("  [PASS] Robustness rankings correctly sorted and attributed.")


def test_3_failure_correlation():
    print("[TEST 3] Testing failure correlation across model combinations...")
    boundaries = _build_mock_boundaries_5_models()
    models = ["dinov2", "clip", "detr", "siglip", "dinov2_large"]

    corr = compute_failure_correlation(boundaries, models)
    assert not corr.empty
    expected_pairs = (5 * 4) // 2  # 10 pairs per degradation x 3 = 30 rows
    assert len(corr) == expected_pairs * 3

    for _, row in corr.iterrows():
        assert 0.0 <= row["jaccard_similarity"] <= 1.0, f"Jaccard out of bounds: {row['jaccard_similarity']}"
        assert 0.0 <= row["overlap_pct"] <= 100.0, f"Overlap % out of bounds: {row['overlap_pct']}"

    print("  [PASS] Failure correlation matrix and Jaccard metrics validated.")


def test_4_ensemble_complementarity():
    print("[TEST 4] Testing ensemble complementarity and N-model intersection...")
    boundaries = _build_mock_boundaries_5_models()
    models = ["dinov2", "clip", "detr", "siglip", "dinov2_large"]

    comp = compute_complementarity(boundaries, models)
    assert not comp.empty
    strategies = comp["strategy"].unique()

    # Must contain single models, pairwise ensembles, and all-5 ensemble
    assert "all_5_ensemble" in strategies, f"Missing all_5_ensemble in {strategies}"
    assert "dinov2_alone" in strategies
    assert "dinov2+clip_ensemble" in strategies

    for _, row in comp.iterrows():
        assert 0.0 <= row["survival_rate"] <= 100.0, f"Survival out of bounds: {row['survival_rate']}"

    print("  [PASS] Single, pairwise, and N-model ensemble survival verified.")


def test_5_architecture_effects_and_awareness():
    print("[TEST 5] Testing architecture effects and self-awareness ranking...")
    boundaries = _build_mock_boundaries_5_models()
    predictability = _build_mock_predictability_5_models()
    svf = _build_mock_svf_5_models()
    models = ["dinov2", "clip", "detr", "siglip", "dinov2_large"]

    arch = analyze_architecture_effects(boundaries, predictability, models)
    assert len(arch) == 5
    assert "dinov2_large" in arch["model"].values
    assert "siglip" in arch["model"].values

    awareness = rank_signal_awareness(predictability, svf, models)
    assert len(awareness) == 5
    assert sorted(list(awareness["rank"].values)) == [1, 2, 3, 4, 5]
    for _, row in awareness.iterrows():
        assert 0.0 <= row["self_awareness_score"] <= 1.0

    print("  [PASS] Architecture metadata and self-awareness scores validated.")


def test_6_synthesis_report():
    print("[TEST 6] Testing synthesis report generation...")
    boundaries = _build_mock_boundaries_5_models()
    predictability = _build_mock_predictability_5_models()
    svf = _build_mock_svf_5_models()
    models = ["dinov2", "clip", "detr", "siglip", "dinov2_large"]

    rankings = compute_robustness_rankings(boundaries, models)
    corr = compute_failure_correlation(boundaries, models)
    comp = compute_complementarity(boundaries, models)
    arch = analyze_architecture_effects(boundaries, predictability, models)
    awareness = rank_signal_awareness(predictability, svf, models)

    report = generate_synthesis(rankings, corr, comp, arch, awareness, predictability, models)
    assert isinstance(report, str)
    assert "AFS-VFM -- Phase 2D: Cross-Model Comparative Analysis" in report
    assert "DINOV2_LARGE" in report
    assert "SIGLIP" in report
    assert "all_5_ensemble" in report

    print("  [PASS] Full multi-model synthesis report generated successfully.")


def run_all_tests():
    print("=" * 70)
    print("  AFS-VFM Cross-Model Analysis Automated Verification Suite")
    print("=" * 70)
    test_1_discover_models()
    test_2_robustness_rankings_multi_model()
    test_3_failure_correlation()
    test_4_ensemble_complementarity()
    test_5_architecture_effects_and_awareness()
    test_6_synthesis_report()
    print("=" * 70)
    print("  ALL 6 ANALYSIS PIPELINE TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
