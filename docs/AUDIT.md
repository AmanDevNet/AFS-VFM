# Technical Audit and Limitations Status Matrix

This document provides a systematic, code-grounded audit of the Anticipatory Failure Signals in Vision Foundation Models (AFS-VFM) repository. It tracks all identified methodological limitations, their corresponding architectural fixes, verification test scripts, and current implementation status.

---

## 1. Limitations & Fixes Summary

| ID | Limitation | Technical Problem | Architectural Fix | Verification Script | Status |
|:---|:---|:---|:---|:---|:---:|
| **L1** | Limited Degradation Types | Original benchmark evaluated only spatial and photometric corruptions (blur, lighting, scale, viewpoint, simple occlusion). Deliberate adversarial attacks and semantic atmospheric shifts were absent. | Added untargeted Projected Gradient Descent ($L_\infty$ PGD) with progressive $\epsilon$ schedules in `src/degradation/adversarial.py`, along with physics-based atmospheric fog and rain synthesis. | `tests/verify_adversarial_and_semantic.py` | **Solved** (Code complete; 10k execution pending) |
| **L2** | Linear Severity Trajectories | Severity stepped strictly linearly ($t = 0 \dots 19$), failing to model sudden real-world environmental transitions or threshold cliffs. | Implemented `SeverityScheduler` in `src/degradation/scheduler.py` supporting linear, exponential, logarithmic, sigmoid, and step-shock schedules. | `tests/verify_schedules.py` | **Solved** |
| **L3** | Synthetic Rectangular Occlusion | Occlusion was simulated via synthetic black rectangular patches, which do not reflect realistic sensor obstructions like dirt or water droplets. | Implemented `apply_translucent_smudge` in `src/degradation/transformations.py` using randomized organic splatters, alpha blending, and Gaussian falloffs. | `tests/verify_occlusion.py` | **Solved** |
| **L4** | Isolated Single Corruptions | Degradations were applied in isolation, omitting concurrent real-world stresses (e.g., night motion blur or lens smudges under low light). | Implemented multi-stress compound pipelines in `src/degradation/compound.py` combining blur, lighting, noise, and occlusion. | `tests/verify_compound.py` | **Solved** |
| **L5** | Limited Model Diversity | Initial experiments were limited to DINOv2-Base, CLIP ViT-B/32, and DETR ResNet-50. Cross-scale and alternative contrastive objectives were untested. | Added support for SigLIP (sigmoid loss contrastive learning) and DINOv2-Large (scale comparison) in `src/models/model_loader.py`. | `tests/verify_model_diversity.py` | **Solved** (Pipeline verified; benchmark execution pending) |
| **L6** | Benchmark Scale (1,500 Images) | Baseline of 1,500 images (1,000 ImageNet-1K, 500 COCO) was constrained by single-GPU memory and runtime limits. | Built distributed sharding infrastructure in `main.py` (`--shard_id`, `--num_shards`) and `merge_shards.py` for 10,000 images across 20 shards. Shard 0 (500 images, 50,000 rows) complete. | `results/shards/shard_000_of_020.csv` | **In Progress** (Shard 0 complete; Shards 1–19 queued) |
| **L7** | DETR Embedding & Cosine Similarity NaN | DETR does not produce a single `[CLS]` token. Attempting naive cosine drift yielded zero-vector or numerical errors. | Implemented dual-representation pooling in `src/models/signal_extractor.py`: spatial encoder mean-pooling combined with detection confidence-weighted query latents, plus numerical epsilon guards. | `tests/verify_detr_fix.py` | **Solved** |
| **L8** | Heuristic Single-Threshold Predictor | Initial failure warning relied on static, single-variable heuristic thresholds without cross-feature weighting or probabilistic calibration. | Implemented learned multi-signal predictors (Gradient Boosting, Random Forest, MLP, Logistic Regression) in `src/earlywarning/learned_predictor.py` trained with GroupKFold cross-validation over 8 dynamic features. | `tests/verify_learned_early_warning.py` | **Solved** (Models trained; full benchmark evaluation pending) |
| **L9** | Lack of Temporal Sequence Modeling | Predictions treated each frame independently without temporal accumulation or change-point detection. | Integrated Cumulative Sum (CUSUM) change-point hazard detection in `src/earlywarning/learned_predictor.py` and `src/earlywarning/failure_predictor.py`. (Complex causal GRUs were scoped out to preserve interpretability). | `tests/verify_learned_early_warning.py` | **Solved** |
| **L10** | CLIP 20-Candidate Label Subset | Initial CLIP classification evaluated against a restricted 20-class text prompt list, which could inflate confidence metrics. | Built full 1,000-class ImageNet-1K zero-shot classification pipeline in `src/models/clip_imagenet1k.py` using pre-cached text embeddings across 80 prompt templates. | `tests/verify_clip_1000.py` | **Solved** (Baseline 1,500-image results remain from legacy 20-label subset) |
| **L11** | Sample Size Statistical Power | Initial power analysis was required to establish sample sizing validity for early warning signal divergence. | Completed statistical power analysis (`scipy.stats`) verifying that 1,500 images provide power > 0.99 for observed effect sizes ($d = 0.315$), with 10k scale providing fine-grained per-degradation power. | `results/phase2a/statistical_tests.csv` | **Solved** |

---

## 2. Status Matrix

| Component / Subsystem | Implementation Location | Verification Script | Verification Status | Operational Notes |
|:---|:---|:---|:---:|:---|
| **Adversarial & Atmospheric Engine** | `src/degradation/adversarial.py` | `tests/verify_adversarial_and_semantic.py` | **PASS** | Generates valid PGD perturbations and synthetic weather |
| **Severity Scheduling Engine** | `src/degradation/scheduler.py` | `tests/verify_schedules.py` | **PASS** | 5 schedules validated; monotonic and bounded |
| **Translucent Smudge Occlusion** | `src/degradation/transformations.py` | `tests/verify_occlusion.py` | **PASS** | Dynamic alpha blending with Gaussian falloff |
| **Compound Degradation Chains** | `src/degradation/compound.py` | `tests/verify_compound.py` | **PASS** | Multi-stress chains functional |
| **Expanded Model Architecture Loader** | `src/models/model_loader.py` | `tests/verify_model_diversity.py` | **PASS** | SigLIP and DINOv2-Large loading verified |
| **DETR Dual Representation & Drift** | `src/models/signal_extractor.py` | `tests/verify_detr_fix.py` | **PASS** | Valid cosine similarity; no NaNs across edge cases |
| **Threshold Early Warning Predictor** | `src/earlywarning/failure_predictor.py` | `tests/verify_early_warning.py` | **PASS** | 6/6 test suites pass; verified lead times |
| **Learned Early Warning Predictors** | `src/earlywarning/learned_predictor.py` | `tests/verify_learned_early_warning.py` | **PASS** | Models load cleanly; inference and CUSUM verified |
| **CLIP 1,000-Class Pipeline** | `src/models/clip_imagenet1k.py` | `tests/verify_clip_1000.py` | **PASS** | 7/7 test suites pass; cached text embeddings |
| **Phase 2B Internal Signals Engine** | `src/models/signal_extractor.py` | `tests/verify_phase2b.py` | **PASS** | 10 internal signals verified without missing values |
| **Smoke Test Suite** | `tests/test_smoke.py` | `tests/test_smoke.py` | **PASS** | Validates sample dataset, transforms, and predictor |
| **Distributed Sharded Runner** | `main.py`, `merge_shards.py` | `results/shards/shard_000_of_020.csv` | **PARTIAL** | Shard 0 verified (50,000 rows); Shards 1–19 pending |

---

## 3. Summary of Resolved vs. Open Items

### Fully Resolved
- **DETR Cosine Sim / Drift Tracking:** Resolved via spatial encoder mean-pooling and confidence-weighted query latents.
- **Occlusion Realism:** Replaced black synthetic boxes with organic translucent smudge transformations.
- **Degradation Trajectories:** Implemented flexible non-linear and compound degradation schedules.
- **Model Loader Expansion:** Added support for SigLIP and DINOv2-Large.
- **CLIP 1,000-Class Pipeline:** Developed full ImageNet-1K zero-shot evaluator with pre-cached text embeddings.
- **Learned Predictor Architecture:** Implemented multi-signal tabular and neural predictors with group-aware cross-validation.

### Open Research Tasks
- **Execution of 10,000-Image Benchmark (Shards 1–19):** Shard 0 of 20 (500 images, 50,000 rows) is complete and verified. Shards 1 through 19 remain queued for execution on distributed GPU compute.
- **Test-Set AUROC on 1,000-Class CLIP & DETR Fix:** The baseline predictability table reported in `results/phase2c/predictability_metrics.csv` reflects the legacy 20-label CLIP and original DETR configurations. Retraining and formal benchmark testing on the completed 10k dataset is pending.
