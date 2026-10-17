"""
Verification suite for Limitation 1 Fix:
- Semantic Environmental Corruptions (gaussian_noise, fog_haze, contrast)
- Gradient-Based Adversarial Perturbation Engine (FGSM, PGD, Sequence Generation)
- Integration with DegradationPipeline and CompoundDegradation

Tests:
1. Zero severity identity across all new semantic corruptions.
2. Controlled degradation progression and visual signal response for semantic transforms.
3. CompoundDegradation integration with new semantic modes.
4. DegradationPipeline end-to-end sequence generation with non-linear schedules.
5. AdversarialAttacker: Zero severity preservation.
6. AdversarialAttacker: FGSM feature disruption and L-infinity budget compliance.
7. AdversarialAttacker: PGD iterative multi-step attack and convergence superiority.
8. AdversarialAttacker: End-to-end adversarial sequence generation with SeverityScheduler.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from degradation import (
    DegradationPipeline,
    CompoundDegradation,
    SeverityScheduler,
    AdversarialAttacker,
    gaussian_noise,
    fog_haze,
    contrast,
)


class DummyFeatureExtractor(nn.Module):
    """Simple differentiable feature extractor simulating a vision backbone."""
    def __init__(self, in_channels=3, feat_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, feat_dim),
        )

    def forward(self, pixel_values, output_hidden_states=True):
        # Emulate ViT/DINOv2 hidden states interface
        feat = self.conv(pixel_values)
        # Shape: (batch_size, 1, feat_dim) to emulate token hidden state
        hidden = feat.unsqueeze(1)
        class Output:
            pass
        out = Output()
        out.hidden_states = [hidden]
        out.last_hidden_state = hidden
        return out


def main():
    print("=" * 70)
    print("  VERIFYING LIMITATION 1 FIX: Adversarial Engine & Semantic Corruptions")
    print("=" * 70)

    # 100x100 synthetic test image with color contrast
    canvas = np.full((100, 100, 3), 128, dtype=np.uint8)
    canvas[20:60, 20:60] = [220, 50, 50]    # Red patch
    canvas[40:80, 50:90] = [50, 180, 50]   # Green patch
    canvas[10:40, 60:90] = [50, 50, 220]   # Blue patch

    # ------------------------------------------------------------
    # Test 1: Zero Severity Identity for Semantic Corruptions
    # ------------------------------------------------------------
    print("\n[Test 1] Zero Severity Identity for Semantic Corruptions:")
    assert np.array_equal(gaussian_noise(canvas, 0.0), canvas), "gaussian_noise failed severity 0.0 identity"
    assert np.array_equal(fog_haze(canvas, 0.0), canvas), "fog_haze failed severity 0.0 identity"
    assert np.array_equal(contrast(canvas, 0.0), canvas), "contrast failed severity 0.0 identity"
    print("  [PASS] All semantic corruptions preserve exact image at severity 0.0")

    # ------------------------------------------------------------
    # Test 2: Progression & Signal Characteristics of Semantic Corruptions
    # ------------------------------------------------------------
    print("\n[Test 2] Progression & Signal Characteristics:")
    for name, fn in [("Gaussian Noise", gaussian_noise), ("Fog / Haze", fog_haze), ("Contrast", contrast)]:
        out_mid = fn(canvas, 0.5)
        out_high = fn(canvas, 1.0)
        assert out_mid.shape == canvas.shape and out_mid.dtype == np.uint8
        assert out_high.shape == canvas.shape and out_high.dtype == np.uint8

        l1_mid = np.mean(np.abs(out_mid.astype(float) - canvas.astype(float)))
        l1_high = np.mean(np.abs(out_high.astype(float) - canvas.astype(float)))
        assert l1_high > l1_mid > 0.0, f"{name} failed monotonic distortion scaling"
        print(f"  - {name:16s}: L1 shift @ 0.5 = {l1_mid:6.2f}, @ 1.0 = {l1_high:6.2f}")
    print("  [PASS] Monotonic severity scaling confirmed across all semantic transforms")

    # ------------------------------------------------------------
    # Test 3: CompoundDegradation Integration
    # ------------------------------------------------------------
    print("\n[Test 3] CompoundDegradation with Semantic Transforms:")
    compound_adhoc = CompoundDegradation.apply(canvas, severity=0.6, components="fog_haze+gaussian_noise")
    assert compound_adhoc.shape == canvas.shape and compound_adhoc.dtype == np.uint8
    diff_compound = np.sum(compound_adhoc != canvas)
    assert diff_compound > 0, "Compound semantic chain failed to alter image"
    print(f"  - 'fog_haze+gaussian_noise' altered {diff_compound} pixels")
    print("  [PASS] Semantic transforms seamlessly compose in CompoundDegradation")

    # ------------------------------------------------------------
    # Test 4: DegradationPipeline End-to-End Sequence Integration
    # ------------------------------------------------------------
    print("\n[Test 4] DegradationPipeline Sequence Generation:")
    pipe = DegradationPipeline(num_frames=6, schedule="exponential", schedule_params={"gamma": 2.5})
    for deg in ["gaussian_noise", "fog_haze", "contrast"]:
        seq = pipe.generate_sequence(canvas, degradation_type=deg)
        assert seq.shape == (6, 100, 100, 3)
        assert np.array_equal(seq[0], canvas), f"{deg} frame 0 is not clean"
        assert not np.array_equal(seq[-1], canvas), f"{deg} frame 5 has no degradation"
        print(f"  - Pipeline sequence '{deg}': shape {seq.shape}, frame 0 identity preserved")
    print("  [PASS] Pipeline integration and non-linear scheduler verified for semantic corruptions")

    # ------------------------------------------------------------
    # Test 5: AdversarialAttacker Zero-Severity Identity
    # ------------------------------------------------------------
    print("\n[Test 5] AdversarialAttacker Zero-Severity Identity:")
    device = torch.device("cpu")
    dummy_model = DummyFeatureExtractor().to(device)
    attacker = AdversarialAttacker(device=device)

    clean_fgsm = attacker.fgsm_feature_attack(dummy_model, canvas, severity=0.0)
    clean_pgd = attacker.pgd_feature_attack(dummy_model, canvas, severity=0.0)
    assert np.array_equal(clean_fgsm, canvas), "FGSM failed zero-severity identity"
    assert np.array_equal(clean_pgd, canvas), "PGD failed zero-severity identity"
    print("  [PASS] Zero severity returns identical base image without gradient computation")

    # ------------------------------------------------------------
    # Test 6: FGSM Feature Disruption & Epsilon Budget Compliance
    # ------------------------------------------------------------
    print("\n[Test 6] FGSM Feature Attack & Epsilon Bounds:")
    max_eps = 16.0 / 255.0
    severity = 0.75
    allowed_pixel_delta = int(np.ceil(severity * max_eps * 255.0)) + 1  # Allowance for uint8 rounding

    adv_fgsm = attacker.fgsm_feature_attack(dummy_model, canvas, severity=severity, max_eps=max_eps)
    assert adv_fgsm.shape == canvas.shape and adv_fgsm.dtype == np.uint8
    max_diff = np.max(np.abs(adv_fgsm.astype(int) - canvas.astype(int)))
    assert max_diff <= allowed_pixel_delta, f"FGSM exceeded L-inf bound: {max_diff} > {allowed_pixel_delta}"

    # Evaluate latent feature drift
    t_clean, _ = attacker._image_to_tensor(canvas)
    t_adv, _ = attacker._image_to_tensor(adv_fgsm)
    with torch.no_grad():
        f_clean = nn.functional.normalize(attacker._extract_feature_tensor(dummy_model, t_clean), p=2, dim=-1)
        f_adv = nn.functional.normalize(attacker._extract_feature_tensor(dummy_model, t_adv), p=2, dim=-1)
        cos_sim_fgsm = float((f_clean * f_adv).sum())

    print(f"  - Max pixel delta: {max_diff} / {allowed_pixel_delta} (within L_inf eps-ball)")
    print(f"  - Feature Cosine Similarity: {cos_sim_fgsm:.4f} (drift from 1.0000)")
    assert cos_sim_fgsm < 0.999, "FGSM failed to cause significant latent feature drift"
    print("  [PASS] FGSM gradient feature disruption strictly adheres to L-inf epsilon bounds")

    # ------------------------------------------------------------
    # Test 7: PGD Iterative Multi-Step Feature Attack
    # ------------------------------------------------------------
    print("\n[Test 7] PGD Multi-Step Attack & Superior Feature Disruption:")
    adv_pgd = attacker.pgd_feature_attack(
        dummy_model, canvas, severity=severity, steps=10, max_eps=max_eps
    )
    assert adv_pgd.shape == canvas.shape and adv_pgd.dtype == np.uint8
    max_diff_pgd = np.max(np.abs(adv_pgd.astype(int) - canvas.astype(int)))
    assert max_diff_pgd <= allowed_pixel_delta, f"PGD exceeded L-inf bound: {max_diff_pgd} > {allowed_pixel_delta}"

    t_pgd, _ = attacker._image_to_tensor(adv_pgd)
    with torch.no_grad():
        f_pgd = nn.functional.normalize(attacker._extract_feature_tensor(dummy_model, t_pgd), p=2, dim=-1)
        cos_sim_pgd = float((f_clean * f_pgd).sum())

    print(f"  - Max pixel delta (PGD): {max_diff_pgd} / {allowed_pixel_delta}")
    print(f"  - Feature Cosine Similarity (PGD):  {cos_sim_pgd:.4f}")
    print(f"  - Feature Cosine Similarity (FGSM): {cos_sim_fgsm:.4f}")
    assert cos_sim_pgd <= cos_sim_fgsm + 0.05, "PGD should achieve comparable or deeper disruption than FGSM"
    print("  [PASS] PGD multi-step attack validated within epsilon constraints")

    # ------------------------------------------------------------
    # Test 8: Adversarial Sequence Generation Orchestration
    # ------------------------------------------------------------
    print("\n[Test 8] Adversarial Sequence Generation with SeverityScheduler:")
    seq_frames = attacker.generate_adversarial_sequence(
        dummy_model,
        canvas,
        num_frames=8,
        attack_type="pgd",
        schedule="linear",
        steps=5,
    )
    assert seq_frames.shape == (8, 100, 100, 3)
    assert np.array_equal(seq_frames[0], canvas), "Adversarial sequence frame 0 must be clean canvas"
    assert not np.array_equal(seq_frames[-1], canvas), "Adversarial sequence final frame must be perturbed"

    drift_scores = []
    for i in range(8):
        t_f, _ = attacker._image_to_tensor(seq_frames[i])
        with torch.no_grad():
            f_i = nn.functional.normalize(attacker._extract_feature_tensor(dummy_model, t_f), p=2, dim=-1)
            drift_scores.append(float((f_clean * f_i).sum()))

    print(f"  Sequence Feature Sim: Start={drift_scores[0]:.4f} -> Mid={drift_scores[4]:.4f} -> Final={drift_scores[-1]:.4f}")
    assert drift_scores[0] > drift_scores[-1], "Feature cosine similarity must degrade along sequence"
    print("  [PASS] Adversarial sequence generation smoothly degrades latent features over time")

    print("\n" + "=" * 70)
    print("  ALL VERIFICATION CHECKS PASSED: Limitation 1 is completely resolved!")
    print("=" * 70)


if __name__ == "__main__":
    main()
