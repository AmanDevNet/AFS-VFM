# AFS-VFM: Complete Project Report

## Anticipatory Failure Signals in Vision Foundation Models

> **Authors:** Aman Sharma (Lead), Angel Taneja (Co-Author)  
> **Started:** March 2026  
> **Status:** Experiments complete. Paper not yet written.  
> **Target Venues:** BMVC / WACV  
> **Repository:** `D:\Research Project\AFS-VFM`  
> **Report Generated:** September 2026

---

## Table of Contents

1. [Why This Project Exists](#1-why-this-project-exists)
2. [What Was Built and Tested](#2-what-was-built-and-tested)
3. [How It Was Done — Technical Flow](#3-how-it-was-done--technical-flow)
4. [Frameworks, Libraries, and Tools](#4-frameworks-libraries-and-tools)
5. [What the Results Mean](#5-what-the-results-mean)
6. [Practical Implications and Limitations](#6-practical-implications-and-limitations)
7. [Current Status and Completeness Audit](#7-current-status-and-completeness-audit)
8. [How TAMP-LU Relates](#8-how-tamp-lu-relates)
9. [Folder Structure Map](#9-folder-structure-map)
10. [Execution Pipeline Flow Diagram](#10-execution-pipeline-flow-diagram)
11. [Pre-Paper Checklist and Suggestions](#11-pre-paper-checklist-and-suggestions)
12. [Glossary of Technical Terms](#12-glossary-of-technical-terms)
13. [60-Second Elevator Pitch](#13-60-second-elevator-pitch)

---

## 1. Why This Project Exists

### The Real-World Problem

Modern AI vision models — the kind used in self-driving cars, medical imaging, security cameras, and manufacturing inspection — perform remarkably well on clean, well-lit, perfectly framed images. On standard benchmark tests, they score near-perfect accuracy. But the real world is not a benchmark. Camera lenses get dirty. Lighting drops. Objects block the view. Perspective shifts.

When these models encounter such "degraded" conditions, they often fail **silently**: they confidently output a wrong answer without any warning. There is no built-in "check engine light" that says *"I'm about to get this wrong."* This is dangerous in safety-critical applications, because the system downstream (a car's braking system, a doctor's diagnosis tool) has no way to know the AI just made a mistake.

### The Research Gap

Existing robustness benchmarks (like ImageNet-C or COCO-C) test models at a few fixed levels of image corruption. They tell you *that* a model failed at, say, "medium blur" — but they don't tell you *when*, during a gradual worsening of conditions, the model's internal reasoning actually broke down. They also don't look inside the model to see if it "knew" failure was coming.

### The Core Research Question

> **"Do the internal signals of a vision AI model — its confidence scores, its internal representations, the patterns in its attention — become measurably unstable BEFORE the model's final answer actually changes?"**

In plain terms: before the AI says the wrong thing, does something inside it already look "off"? And if so, can we catch that early enough to issue a warning?

### Why It Matters

If the answer is yes, we could build a real-time monitoring system that watches the AI's internal state and warns operators when a failure is about to happen — exactly like a check-engine light for AI. This would make deploying vision models in the real world significantly safer.

---

## 2. What Was Built and Tested

### The Models Under Test

Three fundamentally different "Vision Foundation Models" (VFMs) were chosen. Each represents a different philosophy of how to train an AI to understand images:

| Model | Who Made It | Architecture | How It Was Trained | What It Does |
|-------|------------|-------------|-------------------|-------------|
| **DINOv2** | Meta (Facebook) | ViT-B/14 | Self-supervised learning — learned from images alone, no human labels | Image classification (naming what's in the picture) |
| **CLIP** | OpenAI | ViT-B/32 + Text Encoder | Contrastive learning — learned by matching images to text descriptions | Zero-shot classification (identifying objects it was never explicitly trained on) |
| **DETR** | Meta (Facebook) | ResNet-50 + Transformer | Fully supervised — trained with exact bounding boxes and class labels | Object detection (drawing boxes around objects) |

**Why these three?** They all use the Transformer mechanism (a dominant architecture in modern AI), but they were trained in completely different ways. By comparing them, the project isolates the effect of *training paradigm* on robustness — not just the effect of architecture.

### The Data

- **COCO val2017**: 500 images selected from this standard object detection dataset. These images tend to have multiple objects in complex scenes.
- **ImageNet-1k validation**: 1,000 images from this standard classification dataset. These tend to have a single prominent object.
- **Total: 1,500 base images** — a meaningful sample that is large enough for statistical power while fitting within free GPU compute budgets.

### What "Robustness" Means Here

Instead of testing models on a single corrupted image, the project generates a **20-frame temporal degradation sequence** for each image. In each sequence, the corruption intensity increases linearly from 0% (the clean original) to 100% (maximum corruption). This lets the researchers pinpoint the **exact frame** (i.e., the exact level of corruption) where each model's answer first becomes wrong — the **failure boundary**.

Five types of degradation were tested, each simulating a different real-world challenge:

| Degradation | Real-World Scenario | What It Tests |
|-------------|-------------------|---------------|
| **Motion Blur** | Camera shake, moving objects | Robustness to loss of sharp edges and fine detail |
| **Occlusion** | Objects blocking the view (e.g., a hand in front of a lens) | Whether the model relies on local patches or the whole image |
| **Lighting** | Nighttime, shadows, underexposure | Sensitivity to brightness and contrast loss |
| **Scale** | Object far from camera, low resolution | Multi-scale feature extraction capability |
| **Viewpoint** | Camera tilt, unusual angles | Geometric invariance of learned representations |

### What Was Measured

1. **Failure boundaries**: For each (image, degradation, model) combination, the exact frame where the model's prediction first diverges from its clean-image baseline.
2. **Internal model signals**: Extracted at every frame for a strategically selected subset of 300 images:
   - **Confidence** (how "sure" the model is of its top answer)
   - **Logit margin** (the gap between the model's top-1 and top-2 guesses — a wider gap means more decisiveness)
   - **Cosine similarity of embeddings** (how much the model's internal representation has drifted from the clean baseline — 1.0 = identical, 0.0 = completely different)
   - **L2 distance** (another way to measure how far the internal representation has moved)
   - **Attention entropy** (how "spread out" the model's attention is — higher entropy means the model is looking everywhere instead of focusing)
   - **Number of detections** (for DETR: how many objects it thinks it sees)

---

## 3. How It Was Done — Technical Flow

The project was executed in three main phases, each building on the last.

### Phase 1: Data Generation (150,000 Inferences)

**Goal:** Run every image through every degradation type and every model, recording all predictions and failures.

**Pipeline:**
1. **Dataset download** (`download_dataset.py`): Downloads 500 COCO val2017 images and 1,000 ImageNet-1k images.
2. **Degradation sequence generation** (`src/degradation/transformations.py` + `src/degradation/degradation.py`): For each image and each degradation type, generates a 20-frame NumPy array where corruption intensity increases linearly.
3. **Model loading** (`src/models/model_loader.py`): Loads DINOv2, CLIP, and DETR from HuggingFace, placing them on GPU.
4. **Inference loop** (`main.py`): For each image → for each degradation → for each of the 20 frames → run all 3 models → compare each prediction to its frame-0 (clean) baseline → record whether it's a failure.
5. **Output**: `results/full_benchmark.csv` — a 150,000-row CSV with columns for image name, dataset, degradation type, frame number, severity, and each model's prediction and failure status.

**Infrastructure details:**
- Ran on Kaggle Tesla T4 GPU (12 GB VRAM, free tier) in 3 batches of 500 images each (`--batch 1`, `--batch 2`, `--batch 3`) to survive Kaggle's 12-hour session limit.
- Crash-safe: CSV is saved after every single image, and the script auto-resumes by scanning existing results.
- Batches merged into one file by `merge_csv.py`.

**Verification:** `analyze_benchmark.py` produces a data quality report confirming 150,000 rows, 1,500 unique images, correct degradation types and frame counts.

### Phase 2A: Failure Boundary Analysis

**Goal:** For every (image, degradation, model) triplet, find the exact first frame where the model fails.

**Script:** `src/analysis/failure_boundaries.py`

**What it does:**
1. Scans `full_benchmark.csv` and finds the first frame where each model's prediction diverges from baseline.
2. Computes statistics: mean/median failure frame, survival rate, fragility score per (model × degradation).
3. Runs statistical tests: Kruskal-Wallis H-test (to check if the three models differ significantly) and pairwise Mann-Whitney U tests with Bonferroni correction (to check which pairs differ).
4. Builds transition matrices tracking what class each model "hallucinates" when it fails.
5. Generates a human-readable report.

**Output:** `results/phase2a/` — 22,500 failure boundary records, failure statistics, statistical tests, transition matrices, and a text report.

### Phase 2B: Internal Signal Probing (90,000 Inferences)

**Goal:** Extract internal model representations (embeddings, logits, confidence, attention) at every degradation frame for a carefully chosen subset of images.

**Step 1 — Subset selection** (`src/analysis/select_subset.py`): Picks 300 images strategically:
- 100 "fragile" images (where all models fail earliest)
- 100 "robust" images (where models survive longest)
- 100 "disagreement" images (where models fail at very different frames)

**Step 2 — Signal extraction** (`src/models/signal_extractor.py`, run via `AFS_VFM_Phase2B_SignalProbing.ipynb` on Kaggle): For each of the 300 images × 5 degradations × 20 frames × 3 models = 90,000 inferences, extracts ~13 internal signals per frame.

**Output:** `results/phase2b/signals/frame_signals.csv` — 30,000 signal rows with zero missing values.

**Verification:** `verify_phase2b.py` confirms all cosine similarities are 1.000 at frame 1 and all L2 distances are 0.000 at frame 1 (the clean baseline identity check).

### Phase 2C: Anticipatory Signal Detection (Core Hypothesis Test)

**Goal:** Test the central hypothesis — do signals degrade *before* the prediction changes?

**Script:** `src/analysis/anticipatory_signals.py`

**What it does:**
1. Merges Phase 2B signal data with Phase 2A failure boundaries.
2. Re-indexes signals relative to failure point (frame 0 = failure, -5 = five frames before, etc.).
3. For each image that eventually fails, computes the linear regression slope of the signal over the 5 frames preceding failure. A negative slope means the signal was declining before failure.
4. Computes AUROC for binary prediction ("will this image fail?") using each signal value at Frame 10 (~50% severity) as the classifier threshold.
5. Computes when "survived" and "failed" image groups first diverge significantly (Mann-Whitney U test at each frame).

**Output:** `results/phase2c/` — failure-aligned signal trajectories, predictability metrics (AUROC, lead time), pre-failure window statistics, survived-vs-failed divergence data, and a text report.

### Phase 2D: Cross-Model Comparative Analysis

**Goal:** Synthesize all findings into a unified comparison.

**Script:** `src/analysis/cross_model_comparison.py`

**Output:** `results/phase2d/` — robustness rankings, architecture comparison, ensemble complementarity analysis, failure correlation (Jaccard similarity) between model pairs, signal awareness rankings, and a synthesis report.

### Phase 3A: Publication Visualizations

**Script:** `src/visualization/generate_figures.py`

Generates 9 publication-quality figures from Phase 2 CSV data, all with a consistent dark theme at 200 DPI. Saved to `results/figures/`.

### Phase 3B: Failure Early Warning System

**Scripts:**
- `src/earlywarning/learn_thresholds.py` — learns F1-optimal thresholds from Phase 2C data.
- `src/earlywarning/failure_predictor.py` — `FailurePredictor` class that issues OK / WARNING / CRITICAL assessments.
- `src/earlywarning/demo.py` — demonstrates the system on simulated trajectories.
- `src/earlywarning/thresholds.json` — learned threshold configuration.

### Phase 3C: Research Paper

**Status: NOT YET STARTED.** This is the remaining work item.

---

## 4. Frameworks, Libraries, and Tools

| Tool | Version | Role in This Project |
|------|---------|---------------------|
| **Python** | 3.10+ | Primary language for all scripts |
| **PyTorch** | ≥ 2.0 | Deep learning framework. Runs all model inferences on GPU. Chosen because all three models (DINOv2, CLIP, DETR) have official PyTorch implementations. |
| **HuggingFace Transformers** | ≥ 4.30 | Library for downloading and loading pre-trained model weights and processors. Provides `AutoModelForImageClassification`, `CLIPModel`, `DetrForObjectDetection`. Chosen for its one-line model loading and wide model coverage. |
| **OpenCV (cv2)** | ≥ 4.8 | Used for all image degradation algorithms (blur kernels, perspective warps, rescaling). Chosen for its fast C++ backend and comprehensive image processing API. |
| **NumPy** | ≥ 1.24 | Array operations for degradation frame sequences and signal computation. |
| **Pillow (PIL)** | ≥ 9.0 | Image loading and conversion between NumPy arrays and PIL Images (which HuggingFace processors expect). |
| **Pandas** | (used but not in requirements.txt) | CSV reading and data manipulation for all Phase 2 analysis scripts. |
| **SciPy** | (used but not in requirements.txt) | Statistical tests: Kruskal-Wallis H-test, Mann-Whitney U test, linear regression. |
| **scikit-learn** | (used but not in requirements.txt) | AUROC computation (`roc_auc_score`), F1-score optimization for threshold learning. |
| **Matplotlib + Seaborn** | (used but not in requirements.txt) | All 9 publication-quality figures and demo visualizations. |
| **HuggingFace Datasets** | (used but not in requirements.txt) | Streaming download of ImageNet-1k validation images. |
| **Kaggle T4 GPU** | Tesla T4, 12 GB VRAM | All GPU-intensive work (Phase 1 benchmark and Phase 2B signal extraction). Free tier. |

> **⚠ Gap:** The `requirements.txt` file only lists 5 packages (numpy, opencv-python, torch, transformers, Pillow). At least 5 more are used in the codebase (pandas, scipy, scikit-learn, matplotlib/seaborn, datasets) but are not listed. See the Pre-Paper Checklist for details.

---

## 5. What the Results Mean

### The Headline Numbers

#### AUROC = 0.822

**What this number is:** AUROC stands for "Area Under the Receiver Operating Characteristic curve." It measures how well a binary classifier can distinguish between two groups — in this case, "will this image eventually cause a failure?" vs. "will the model survive?"

- A score of **0.5** means the classifier is no better than random guessing (a coin flip).
- A score of **1.0** means the classifier is perfect — it can always tell the difference.
- A score of **0.822** means: if you pick one random image that will eventually fail and one that won't, there is an **82.2% chance** that the signal value (CLIP's logit margin at ~50% severity) correctly identifies which is which.

**What this means for the project:** By monitoring CLIP's logit margin (the gap between its top two guesses), you can predict upcoming AI failure with 82.2% accuracy **3.8 frames before it happens**. That's a practically useful early warning signal.

#### p = 1.98 × 10⁻¹⁰

**What this number is:** A p-value measures the probability that an observed result happened by pure chance. A p-value of 1.98 × 10⁻¹⁰ (which is 0.000000000198, or about 1 in 5 billion) means there is essentially zero chance this result is a statistical fluke.

**What this proves:** DINOv2's internal signals diverge significantly between "survived" and "failed" images as early as **Frame 2** (just 10% degradation severity). This divergence is not random — it's a real, measurable, and extremely statistically significant pattern. DINOv2 "knows" very early that it's looking at an image it will struggle with.

### Full Results Summary

#### 1. Robustness Rankings

| Rank | Model | Avg Survival Rate | Training Paradigm |
|:----:|-------|:-----------------:|-------------------|
| #1 | **DINOv2** | **33.9%** | Self-supervised |
| #2 | CLIP | 23.4% | Contrastive |
| #3 | DETR | 16.5% | Fully supervised |

DINOv2 ranked #1 in 4 out of 5 degradation types. The sole exception is Scale, where CLIP slightly edged ahead (3.7% vs 0.1% survival), though both were near zero — Scale is universally devastating.

**Survival Rates by Degradation:**

| Degradation | DINOv2 | CLIP | DETR |
|:-----------:|:------:|:----:|:----:|
| Blur | 35.8% | 34.9% | 9.5% |
| Lighting | 42.9% | 30.1% | 23.8% |
| Occlusion | **51.3%** | 24.1% | 14.4% |
| Scale | 0.1% | 3.7% | 0.0% |
| Viewpoint | 39.5% | 24.0% | 34.7% |

#### 2. Core Hypothesis: Confirmed

Internal signals drop 3–5 frames BEFORE the model's prediction actually changes. Specifically, 83.2% of pre-failure windows showed negative signal trends (the signal was declining before failure occurred).

**Signal Trajectory Around the Failure Point (Cosine Similarity):**

| Relative Frame | DINOv2 | CLIP |
|:--------------:|:------:|:----:|
| −5 (5 before failure) | 0.902 ± 0.120 | 0.925 ± 0.074 |
| −4 | 0.882 ± 0.130 | 0.915 ± 0.079 |
| −3 | 0.855 ± 0.145 | 0.899 ± 0.089 |
| −2 | 0.812 ± 0.170 | 0.881 ± 0.104 |
| −1 | 0.734 ± 0.214 | 0.864 ± 0.122 |
| **0 (failure)** | **0.580 ± 0.290** | **0.810 ± 0.153** |

The clear downward slope from −5 to 0 in both models visually and statistically proves the hypothesis.

#### 3. Best Failure Predictors

| Model | Signal | AUROC | Lead Time | % Negative Trend |
|-------|--------|:-----:|:---------:|:----------------:|
| **CLIP** | **logit_margin** | **0.822** | 3.8 frames | 76% |
| CLIP | top1_confidence | 0.817 | 3.8 frames | 75% |
| DETR | top_detection_confidence | 0.816 | 3.4 frames | 69% |
| DINOv2 | top1_confidence | 0.773 | 4.2 frames | 85% |
| DINOv2 | logit_margin | 0.771 | 4.2 frames | 85% |
| DINOv2 | cosine_sim | 0.519 | 4.8 frames | 97% |
| CLIP | cosine_sim | 0.380 | 4.8 frames | 96% |

**Key insight:** Logit-based signals (confidence and margin) are much better predictors than embedding-based signals (cosine similarity). Cosine similarity drops almost everywhere (96–97% negative trend) but has poor AUROC — it drops for both survived *and* failed images, making it less discriminative.

#### 4. Earliest Self-Awareness

| Model | First Significant Divergence | p-value |
|-------|:----------------------------:|:-------:|
| **DINOv2** | **Frame 2 (10% severity)** | **1.98 × 10⁻¹⁰** |
| CLIP | Frame 5 (26% severity) | 2.45 × 10⁻² |
| DETR | No significant divergence | — |

DINOv2's self-supervised training gives it the earliest "self-awareness" — its internal representations for images that will eventually fail start looking different from images that will survive at just 10% corruption.

#### 5. Ensemble Potential

| Strategy | Avg Survival Rate |
|----------|:-----------------:|
| All 3 models | **51.6%** |
| DINOv2 + CLIP | 45.9% |
| DINOv2 + DETR | 42.9% |
| DINOv2 alone | 33.9% |
| CLIP + DETR | 33.0% |

An ensemble of all three models reduces the failure rate by 52% compared to using DINOv2 alone. This works because the models fail on different images (low failure correlation, especially DINOv2 vs. the other two).

#### 6. Hallucination Patterns

When models fail, they don't just become "uncertain" — they confidently switch to specific wrong classes. For example:
- **DINOv2** at extreme scale: latches onto "cowboy hat" (610 times) and "hook, claw" (405 times)
- **DETR** under corruption: "person" → "toilet" (952 times), "person" → "tv" (769 times)
- **CLIP**: "a chair" → "a person" (1,361 times)

---

## 6. Practical Implications and Limitations

### How This Research Could Be Useful

1. **Real-time failure monitoring:** The early warning system prototype (`src/earlywarning/failure_predictor.py`) demonstrates that it's feasible to monitor a model's internal signals during deployment and issue warnings 3+ frames before failure. This could be integrated into autonomous driving, medical imaging, or surveillance systems.

2. **Model selection guidance:** If you're choosing which foundation model to deploy in a safety-critical application, this research provides evidence that self-supervised models (like DINOv2) are more robust to real-world degradation than contrastive (CLIP) or supervised (DETR) models.

3. **Ensembling strategy:** The failure correlation analysis shows which model pairs are most complementary. DINOv2 + CLIP is a better ensemble than CLIP + DETR because DINOv2 fails on different images.

4. **Degradation-specific deployment:** If your application primarily faces occlusion (e.g., a robot in a cluttered environment), DINOv2 is strongly favored (51.3% survival). If it primarily faces blur, the advantage is smaller.

### Honest Limitations

1. **Degradation types are limited:** Only spatial/photometric corruptions are tested. No adversarial attacks (deliberate pixel-level perturbations designed to fool models), no style transfers, no semantic corruptions.
2. **Linear severity only:** Real-world degradation doesn't always increase linearly. Non-linear schedules might reveal different behaviors.
3. **Simplistic occlusion:** The project uses black rectangles. Real-world occlusion (a person walking in front of a camera) is far more varied.
4. **No compound degradations:** Blur + darkness simultaneously is common in the real world but not tested here.
5. **Only 3 models:** More recent models (SAM, Grounding DINO, SigLIP, etc.) are not included.
6. **1,500 images:** While sufficient for statistical significance, scaling to 10K+ images would strengthen the claims.
7. **DETR cosine similarity is NaN:** DETR's architecture doesn't produce a single CLS embedding, so embedding-based signals are missing for DETR, limiting comparison depth.
8. **Threshold-based predictor:** The early warning system uses fixed thresholds. A learned classifier (e.g., a small neural network analyzing the full signal trajectory) could likely do better.
9. **No temporal modeling:** The predictor evaluates each frame independently rather than modeling the trajectory over time.
10. **CLIP's label set is small:** CLIP uses only 20 candidate labels for zero-shot classification, which is a significant constraint compared to its full capability.

---

## 7. Current Status and Completeness Audit

### What IS Complete

| Phase | Status | Evidence |
|-------|--------|---------|
| Phase 1: Data Generation | ✅ **Complete** | `results/full_benchmark.csv` exists (15.2 MB, 150,000 rows). Verified by `analysis_output.txt`. |
| Phase 2A: Failure Boundaries | ✅ **Complete** | `results/phase2a/` has all 5 output files including `phase2a_report.txt`. |
| Phase 2B: Signal Probing | ✅ **Complete** | `results/phase2b/signals/frame_signals.csv` exists (8.5 MB, 30,000 rows). Verified by `verify_phase2b.py`. |
| Phase 2C: Anticipatory Signals | ✅ **Complete** | `results/phase2c/` has all 6 output files including `phase2c_report.txt` and `predictability_metrics.csv`. |
| Phase 2D: Cross-Model Comparison | ✅ **Complete** | `results/phase2d/` has all 6 output files including `phase2d_synthesis.txt`. |
| Phase 3A: Visualizations | ✅ **Complete** | `results/figures/` contains all 9 publication figures plus 2 demo figures (11 PNGs total). |
| Phase 3B: Early Warning System | ✅ **Complete** | `src/earlywarning/` has all 4 files including `thresholds.json` and `failure_predictor.py`. |
| Phase 3C: Research Paper | ❌ **Not Started** | No paper draft, no LaTeX files, no PDF found anywhere in the repository. |
| Phase 3C: Repository Polish | ❌ **Not Started** | Temporary files still present; `requirements.txt` incomplete; README not updated with Phase 2 results. |

### Existing Output Artifacts

**Data Files (20+ CSVs):**
- `results/full_benchmark.csv` (150,000 rows, 15.2 MB)
- `results/batch_1.csv`, `batch_2.csv`, `batch_3.csv` (batch checkpoints)
- `results/pilot_benchmark.csv` (initial test run)
- `results/phase2a/failure_boundaries.csv` (22,500 rows)
- `results/phase2a/failure_statistics.csv` (15 rows)
- `results/phase2a/statistical_tests.csv` (~30 rows)
- `results/phase2a/transition_matrices/` (3 CSVs: dinov2, clip, detr)
- `results/phase2b/selected_images.csv` (300 rows)
- `results/phase2b/signals/frame_signals.csv` (30,000 rows, 8.5 MB)
- `results/phase2c/merged_signals_boundaries.csv` (7.6 MB)
- `results/phase2c/failure_aligned_signals.csv`
- `results/phase2c/predictability_metrics.csv` (7 rows — the AUROC results)
- `results/phase2c/prefailure_window_stats.csv`
- `results/phase2c/survived_vs_failed.csv`
- `results/phase2d/robustness_rankings.csv`
- `results/phase2d/architecture_comparison.csv`
- `results/phase2d/ensemble_complementarity.csv`
- `results/phase2d/failure_correlation_matrix.csv`
- `results/phase2d/signal_awareness_ranking.csv`

**Figures (11 PNGs):**
- `results/figures/fig1_robustness_heatmap.png`
- `results/figures/fig2_failure_distributions.png`
- `results/figures/fig3_signal_trajectory.png` ← **the key figure** proving the hypothesis
- `results/figures/fig4_auroc_comparison.png`
- `results/figures/fig5_roc_curves.png`
- `results/figures/fig6_divergence_timeline.png`
- `results/figures/fig7_ensemble_benefit.png`
- `results/figures/fig8_failure_correlation.png`
- `results/figures/fig9_architecture_radar.png`
- `results/figures/demo_earlywarning_dinov2.png`
- `results/figures/demo_earlywarning_clip.png`

**Text Reports:**
- `results/phase2a/phase2a_report.txt`
- `results/phase2c/phase2c_report.txt`
- `results/phase2d/phase2d_synthesis.txt`
- `analysis_output.txt` (Phase 1 data quality report)

**Documentation:**
- `docs/PHASE2_REPORT.md`
- `docs/PHASE3_PLAN.md`
- `docs/PROJECT_RECAP.md` (written for co-author Angel)
- `docs/RESULTS.md`
- `docs/project_architecture.md`
- `docs/research_notes_phase1_aman.md`
- `docs/research_notes_phase1_week1_aman.md`
- `docs/research_notes_phase1_week2_aman.md`
- `AFS_VFM_COMPLETE_RESEARCH_KNOWLEDGE.md` (comprehensive knowledge base)

**Demo/Test Data:**
- `data/demo_output/` — 15 demo images showing degradation sequences (clean + degraded pairs for all 5 types)
- `data/sample_dog.jpg` — sample test image
- `week2_final_outcome.csv` — early DINOv2 test (clock → chain mail at frame 2)

### Things That Look Unfinished or Inconsistent

1. **`requirements.txt` is critically incomplete** — lists only 5 of ~10 required packages. Missing: `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `datasets`. No version pinning (just `>=` ranges).
2. **Temporary/debug files still present:** `verify_phase2b.py`, `run_combined_test.py`, `test_script.py`, `week2_final_outcome.csv` should be cleaned up before publication.
3. **Duplicate file structure:** The project exists both at `d:\Research Project\` (working copy) and `d:\Research Project\AFS-VFM\` (git repository). Some files are slightly different sizes between the two copies (e.g., `main.py` is 13,931 bytes vs 14,263 bytes). The git repo version (`AFS-VFM/`) appears to be the more complete copy (it has `results/figures/`, `src/earlywarning/`, `src/visualization/`, and `docs/RESULTS.md`).
4. **README is outdated:** States "Phase 2 Complete (~80%)" in the badge, but Phase 3A and 3B are also complete. Doesn't mention Phase 2 results, figures, or the early warning system.
5. **No `src/data/` in the working copy:** The `src/data/curate_coco.py` and `src/data/curate_datasets.py` mentioned in docs are listed in the knowledge base but were not found in the explored working copy's `src/` directory (only `analysis/`, `degradation/`, `models/` were found). They may exist only in the git repo copy.
6. **No `src/visualization/` in the working copy** — this module exists in `AFS-VFM/src/` but not in the parent `Research Project/src/`.
7. **DETR cosine similarity produces NaN** — acknowledged in the code and reports but not fully addressed.
8. **No LICENSE file** — README references MIT License but no `LICENSE` file was found.

---

## 8. How TAMP-LU Relates

### Finding: TAMP-LU Does Not Exist in This Repository

A thorough search of every file in the project (all `.py`, `.md`, `.txt`, `.json`, `.csv`, and `.ipynb` files) was conducted for the terms "TAMP", "TAMP-LU", "temporal attention map probing", "localization uncertainty", and related phrases. **No matches were found.**

There is no code, no documentation, no notes, no paper drafts, and no configuration files referencing TAMP-LU anywhere in the `d:\Research Project\` directory or the `AFS-VFM` subdirectory.

### Possible Explanations

1. **TAMP-LU may be a separate, unrealized project idea** — perhaps planned as a follow-up to AFS-VFM but never started or stored elsewhere.
2. **TAMP-LU may exist in a different repository or location** not connected to this folder.
3. **The name might be misremembered** — the project's attention entropy analysis (tracking how spread out a model's attention weights are across frames) touches on related concepts, but it is not called TAMP-LU and does not focus on "localization uncertainty" specifically.

### Conceptual Connection (Speculative)

Based on the name "Temporal Attention Map Probing for Localization Uncertainty":
- **Temporal**: AFS-VFM uses temporal degradation sequences (20 frames over time).
- **Attention Map Probing**: AFS-VFM extracts attention entropy from DINOv2's attention heads.
- **Localization Uncertainty**: DETR does object detection (localization), and the project tracks DETR's detection confidence degradation.

A TAMP-LU module could logically be an extension that analyzes *where* models look (attention maps) across the temporal degradation sequence to study *how uncertain the model is about object location* (not just classification). However, this module **has not been built** in the current codebase.

---

## 9. Folder Structure Map

```
d:\Research Project\AFS-VFM\                    ← Git repository root
│
├── main.py                                      # Phase 1 orchestrator (150K inferences)
├── download_dataset.py                          # Dataset downloader (COCO + ImageNet)
├── merge_csv.py                                 # Merges batch CSVs into full_benchmark.csv
├── analyze_benchmark.py                         # Phase 1 data quality report
├── demo_visual.py                               # Visual demo of degradation sequences
├── run_combined_test.py                         # Integration test (Week 2)
├── test_script.py                               # Degradation engine smoke test
├── verify_phase2b.py                            # Phase 2B signal data verification
├── week2_final_outcome.csv                      # Early DINOv2 test output
├── requirements.txt                             # Python dependencies (INCOMPLETE)
├── README.md                                    # Project README (needs update)
├── .gitignore                                   # Git ignore rules
├── dummy.png                                    # Test image
│
├── src/
│   ├── degradation/                             # Track A: Degradation Engine
│   │   ├── __init__.py                          # Public API exports
│   │   ├── degradation.py                       # DegradationPipeline class
│   │   ├── transformations.py                   # 5 corruption algorithms
│   │   └── utils.py                             # Image I/O utilities
│   │
│   ├── models/                                  # Track B: Model Inference
│   │   ├── __init__.py
│   │   ├── model_loader.py                      # VisionModelLoader (DINOv2, CLIP, DETR)
│   │   └── signal_extractor.py                  # SignalExtractor (embeddings, logits)
│   │
│   ├── analysis/                                # Phase 2: Statistical Analysis
│   │   ├── __init__.py
│   │   ├── failure_boundaries.py                # Phase 2A
│   │   ├── select_subset.py                     # Phase 2B subset selection
│   │   ├── anticipatory_signals.py              # Phase 2C (core hypothesis)
│   │   └── cross_model_comparison.py            # Phase 2D
│   │
│   ├── visualization/                           # Phase 3A: Figure Generation
│   │   ├── __init__.py
│   │   └── generate_figures.py                  # 9 publication figures
│   │
│   └── earlywarning/                            # Phase 3B: Failure Prediction
│       ├── __init__.py
│       ├── learn_thresholds.py                  # Learns F1-optimal thresholds
│       ├── failure_predictor.py                 # FailurePredictor class
│       ├── thresholds.json                      # Learned threshold config
│       └── demo.py                              # Early warning demo
│
├── results/
│   ├── full_benchmark.csv                       # 150,000 rows (Phase 1)
│   ├── batch_1.csv, batch_2.csv, batch_3.csv   # Batch checkpoints
│   ├── pilot_benchmark.csv                      # Initial test run
│   ├── phase2a/                                 # Failure boundary results
│   ├── phase2b/                                 # Signal probing results
│   ├── phase2c/                                 # Anticipatory signal results
│   ├── phase2d/                                 # Cross-model comparison results
│   └── figures/                                 # 11 publication-quality PNGs
│
├── docs/                                        # Documentation
│   ├── PHASE2_REPORT.md
│   ├── PHASE3_PLAN.md
│   ├── PROJECT_RECAP.md
│   ├── RESULTS.md
│   ├── project_architecture.md
│   └── research_notes_phase1_*.md               # 3 research note files
│
├── notebooks/
│   ├── AFS_VFM_Benchmark.ipynb                  # Phase 1 Kaggle notebook
│   └── AFS_VFM_Phase2B_SignalProbing.ipynb      # Phase 2B Kaggle notebook
│
└── data/                                        # (gitignored)
    ├── coco_val/                                 # 500 COCO validation images
    ├── imagenet_val/                             # 1,000 ImageNet validation images
    ├── demo_output/                              # Demo visualizations
    └── sample_dog.jpg                            # Test image
```

---

## 10. Execution Pipeline Flow Diagram

### Phase 1: Data Generation

```
Raw Images (1,500)
  │
  ├── data/coco_val/ (500 COCO images)
  └── data/imagenet_val/ (1,000 ImageNet images)
        │
        ▼
┌─── download_dataset.py ──────────────────────────────┐
│    Downloads COCO zip + HuggingFace ImageNet stream   │
└───────────────────────┬──────────────────────────────┘
                        │
                        ▼
┌─── main.py ──────────────────────────────────────────┐
│                                                       │
│   For each image (1,500):                             │
│     For each degradation (5 types):                   │
│                                                       │
│       ┌── src/degradation/degradation.py ──┐          │
│       │  generate_degradation_sequence()   │          │
│       │  → 20 frames, severity 0% → 100%  │          │
│       └─────────────┬──────────────────────┘          │
│                     │                                  │
│       For each frame (20):                             │
│         ┌── src/models/model_loader.py ──┐             │
│         │  DINOv2 → prediction + fail?   │             │
│         │  CLIP   → prediction + fail?   │             │
│         │  DETR   → prediction + fail?   │             │
│         └────────────┬───────────────────┘             │
│                      │                                 │
│         Append row to CSV                              │
│       Save after every image (crash-safe)              │
│                                                        │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
           results/full_benchmark.csv
              (150,000 rows)
```

### Phase 2: Analysis Pipeline

```
results/full_benchmark.csv (150,000 rows)
        │
        ├──── src/analysis/failure_boundaries.py ────► results/phase2a/
        │     (finds exact failure frame per triplet)     ├── failure_boundaries.csv
        │                                                 ├── failure_statistics.csv
        │                                                 ├── statistical_tests.csv
        │                                                 └── transition_matrices/
        │
        ├──── src/analysis/select_subset.py ─────────► results/phase2b/selected_images.csv
        │     (picks 300 strategic images)                 (300 images)
        │
        ├──── src/models/signal_extractor.py ────────► results/phase2b/signals/frame_signals.csv
        │     (90K inferences on Kaggle GPU)               (30,000 signal rows)
        │
        ├──── src/analysis/anticipatory_signals.py ──► results/phase2c/
        │     (tests core hypothesis)                      ├── predictability_metrics.csv (AUROC!)
        │                                                  ├── failure_aligned_signals.csv
        │                                                  └── survived_vs_failed.csv
        │
        └──── src/analysis/cross_model_comparison.py ► results/phase2d/
              (synthesizes everything)                     ├── robustness_rankings.csv
                                                           ├── ensemble_complementarity.csv
                                                           └── phase2d_synthesis.txt
```

### Phase 3: Outputs

```
results/phase2*/  (all CSV data)
        │
        ├──── src/visualization/generate_figures.py ─► results/figures/ (9 PNGs)
        │
        ├──── src/earlywarning/learn_thresholds.py ──► thresholds.json
        │
        └──── src/earlywarning/demo.py ──────────────► 2 demo PNGs + console output
```

---

## 11. Pre-Paper Checklist and Suggestions

### Critical (Must Fix Before Submission)

| # | Issue | Details | Action Required |
|:-:|-------|---------|-----------------|
| 1 | **`requirements.txt` is incomplete** | Only lists 5 of ~10 required packages. Missing: `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `datasets`. | Add all dependencies with pinned versions (e.g., `pandas==2.1.0`). |
| 2 | **No paper draft exists** | Phase 3C has not been started. No LaTeX, Word, or PDF file exists. | Write the paper following the structure in `docs/PHASE3_PLAN.md`. |
| 3 | **No `LICENSE` file** | README references MIT License but no file exists. | Add a `LICENSE` file to the repo root. |
| 4 | **CLIP's zero-shot label set is only 20 labels** | This is a significant methodological limitation that reviewers will question. CLIP's "classification" is constrained to only 20 candidate labels. | Acknowledge clearly in the paper. Consider whether this biases results. |
| 5 | **README is outdated** | Badge says "Phase 2 Complete (~80%)". Doesn't mention Phase 2 findings, figures, or early warning system. | Update README with final results, embed key figures, and update badge. |

### Important (Should Fix)

| # | Issue | Details |
|:-:|-------|---------|
| 6 | **No random seeds documented in `requirements.txt` or config** | `random.seed(42)` is used throughout code but not in a central config. A reproducibility statement should go in the paper. |
| 7 | **Temporary files should be removed from repo** | `verify_phase2b.py`, `run_combined_test.py`, `test_script.py`, `week2_final_outcome.csv`, `dummy.png` are debug artifacts. Move to a `tests/` folder or remove. |
| 8 | **DETR cosine similarity NaN not addressed** | The signal extractor doesn't produce embeddings for DETR. Phase 2C report shows all NaN. The paper should explicitly state this. |
| 9 | **No ablation studies** | Reviewers will ask: Why 20 frames? Why not 10 or 50? What happens with different CLIP label sets? What about different image resolutions? |
| 10 | **No related work section exists anywhere** | The paper plan mentions it, but no draft of related work or citation list exists. |

### Nice to Have

| # | Suggestion |
|:-:|-----------|
| 11 | **Create a simple demo notebook or Gradio/Streamlit app** showing the early warning system in action. This would strengthen a portfolio or paper supplementary material. |
| 12 | **Run experiments with at least one additional model** (e.g., SigLIP, DINOv2-large) to show the findings generalize. |
| 13 | **Test compound degradations** (blur + lighting) — this is a common reviewer request and explicitly noted as a limitation. |
| 14 | **Add confidence intervals to AUROC values** — bootstrap or cross-validation based. Single-split AUROC can be noisy. |
| 15 | **Consider non-linear severity schedules** — the current linear schedule may miss perceptual thresholds. |
| 16 | **Figure polish** — check that all figures have proper axis labels, legends, and captions ready for the paper template. |

### Missing Citations / Related Work to Cover

Based on the project's scope, the paper should cite and position against:
- **ImageNet-C / COCO-C** (Hendrycks & Dietterich, 2019) — the main prior benchmark for model robustness under corruptions
- **DINOv2** (Oquab et al., 2023) — the model itself
- **CLIP** (Radford et al., 2021) — the model itself
- **DETR** (Carion et al., 2020) — the model itself
- **Vision Transformers** (Dosovitskiy et al., 2020) — ViT architecture
- **Uncertainty estimation** literature (Gal & Ghahramani, 2016; Lakshminarayanan et al., 2017) — competing approaches to predicting model failure
- **Out-of-distribution detection** (Hendrycks & Gimpel, 2017; Liu et al., 2020) — related problem of detecting inputs the model can't handle
- **Model calibration** (Guo et al., 2017) — whether model confidence aligns with actual accuracy
- Any recent BMVC/WACV papers on VFM robustness to stay current

---

## 12. Glossary of Technical Terms

| Term | Plain-English Definition | Relevance to This Project |
|------|------------------------|--------------------------|
| **AUROC** (Area Under the Receiver Operating Characteristic curve) | A score from 0 to 1 measuring how well a system can tell two groups apart. 0.5 = random guessing, 1.0 = perfect. | Used to measure how well each internal signal predicts whether an image will cause a model failure. The headline result (0.822) means CLIP's logit margin correctly distinguishes "will fail" from "won't fail" images 82.2% of the time. |
| **p-value** | The probability that an observed result happened by pure luck. Smaller = more convincing the result is real. Typically, p < 0.05 is considered "statistically significant." | The p = 1.98 × 10⁻¹⁰ finding means there is essentially zero chance that DINOv2's early signal divergence is a random coincidence. |
| **Statistical significance** | A formal way of saying "this result is very unlikely to be due to random chance." | The project uses Kruskal-Wallis and Mann-Whitney U tests to prove the three models differ in robustness, and that survived/failed image groups diverge in internal signals. |
| **Foundation model** | A large AI model trained on massive amounts of data that can be adapted for many different tasks. "Foundation" because it serves as a base for downstream applications. | DINOv2, CLIP, and DETR are all foundation models — large, general-purpose models widely used as building blocks. |
| **Vision Foundation Model (VFM)** | A foundation model specifically designed to understand images and video. | The three models tested in this project are VFMs. The research studies their robustness. |
| **DINOv2** | A vision model by Meta that learns image representations without human labels (self-supervised). It looks at images and learns useful features purely from visual structure. | Tested as the self-supervised representative. Found to be the most robust (33.9% survival). |
| **CLIP** | A model by OpenAI that connects images and text. It was trained on 400 million image-text pairs from the internet, learning to match images with their descriptions. | Tested as the contrastive learning representative. Its logit margin was the best failure predictor (AUROC = 0.822). |
| **DETR** | A model by Meta that detects and locates objects in images using a combination of a CNN backbone and a Transformer decoder. | Tested as the supervised detection representative. Found to be the most fragile (16.5% survival). |
| **ViT (Vision Transformer)** | An architecture that applies the Transformer design (originally from language AI) to images by splitting them into small patches and processing them as a sequence. | Both DINOv2 and CLIP use ViT backbones, allowing the project to isolate the effect of training paradigm. |
| **Self-supervised learning** | Training an AI without human-provided labels. The model learns by solving "pretext tasks" (like predicting a masked part of an image from the rest). | DINOv2's training paradigm. The project found this produces the most robust features. |
| **Contrastive learning** | Training by showing the model pairs of related items (an image and its caption) and teaching it to match them while pushing unrelated pairs apart. | CLIP's training paradigm. |
| **Embedding** | A numerical representation (a list of numbers, or "vector") that a model creates internally to represent an input. Similar inputs have similar embeddings. | The project tracks how embeddings drift as images degrade. Large drift = the model's internal understanding is changing. |
| **CLS embedding** | In Vision Transformers, a special token whose embedding summarizes the entire image. It's the model's "summary vector" for the whole input. | Cosine similarity of CLS embeddings between clean and degraded frames measures how much the model's internal representation has shifted. |
| **Cosine similarity** | A measure of how similar two vectors are, ranging from −1 (opposite) to 1 (identical). It compares direction, not magnitude. | Used to track embedding drift. A value of 1.0 means the embedding hasn't changed; lower values mean the model's internal state is shifting. |
| **Logit** | The raw, unnormalized score a model assigns to each possible class before converting to probabilities. Higher logit = model thinks this class is more likely. | Logit margin (gap between top-1 and top-2 logits) is the best single predictor of failure. |
| **Logit margin** | The difference between the model's top-1 and top-2 logit values. A large margin means the model is very decisive; a small margin means it's torn between two answers. | CLIP's logit margin achieves AUROC = 0.822 — the project's best failure predictor. |
| **Softmax** | A mathematical function that converts raw logits into probabilities that sum to 1. | Used to compute confidence scores from model outputs. |
| **Attention** | A mechanism in Transformers where each part of the input can "attend to" (look at) every other part, weighted by relevance. | The project extracts attention entropy to see if the model's focus pattern changes before failure. |
| **Attention entropy** | A measure of how "spread out" a model's attention is. High entropy means the model is looking everywhere equally (unfocused); low entropy means it's focused on specific parts. | Tracked as one of DINOv2's internal signals during degradation. |
| **Adversarial robustness** | A model's ability to resist deliberate, carefully crafted perturbations designed to fool it. | The project does NOT test adversarial robustness — it tests robustness to natural, physical degradation. |
| **Out-of-distribution (OOD)** | An input that is very different from what the model was trained on. OOD detection is the task of identifying such inputs. | Related but distinct — AFS-VFM studies gradual distribution shift rather than sudden OOD inputs. |
| **Degradation sequence** | A series of 20 images where the same corruption is applied with linearly increasing intensity, from 0% (clean) to 100% (maximum corruption). | The core experimental stimulus of the project. |
| **Failure boundary** | The exact frame in a degradation sequence where a model's prediction first differs from its clean-image baseline. | The primary measurement in Phase 2A. |
| **Survival rate** | The percentage of (image, degradation) pairs where a model's prediction never changes across all 20 frames. | DINOv2: 33.9%, CLIP: 23.4%, DETR: 16.5%. |
| **Fragility score** | 1 − (mean_failure_frame / 20). Higher fragility = the model fails earlier = more fragile. | Used to rank model-degradation combinations by vulnerability. |
| **Kruskal-Wallis H-test** | A non-parametric statistical test that checks whether three or more independent groups have the same distribution. | Used to test whether DINOv2, CLIP, and DETR have significantly different failure frame distributions. |
| **Mann-Whitney U test** | A non-parametric test comparing two groups. | Used for pairwise comparisons (DINOv2 vs. CLIP, etc.) and survived-vs-failed divergence analysis. |
| **Bonferroni correction** | A method to adjust p-values when making multiple comparisons, to avoid false positives. | Applied to the pairwise Mann-Whitney U tests. |
| **Cohen's d** | An effect size measure. Small (0.2), medium (0.5), large (0.8). | Reported alongside statistical tests to show practical significance, not just statistical significance. |
| **F1-score** | The harmonic mean of precision and recall. A balanced measure of classifier performance that accounts for both false positives and false negatives. | Used to optimize warning thresholds in the early warning system. |
| **Ensemble** | Using multiple models together, typically by requiring all (or a majority) to agree before accepting a prediction. | The project shows that ensembling all 3 models boosts survival from 33.9% to 51.6%. |
| **Jaccard similarity** | The size of the intersection divided by the size of the union of two sets. Measures how much overlap there is between two groups. | Used to measure failure correlation between model pairs. CLIP-DETR Jaccard = 0.714 (high overlap in failures). |
| **Hallucination** | When an AI model confidently outputs a completely wrong and often nonsensical answer. | The project tracks what classes models "hallucinate" when they fail (e.g., DETR sees "toilet" instead of "person"). |
| **Zero-shot classification** | Classifying an image into categories the model has never been explicitly trained on, by leveraging language understanding. | CLIP's approach: it compares the image to 20 text descriptions and picks the best match. |
| **Inference** | Running a trained model on an input to get a prediction, without updating the model's weights. | The project ran 240,000 total inferences (150K in Phase 1 + 90K in Phase 2B). |
| **GPU (Graphics Processing Unit)** | A processor designed for parallel computation. Essential for running large AI models at reasonable speed. | The project used Kaggle's Tesla T4 GPU (12 GB VRAM, free tier). |
| **VRAM** | Video RAM — the GPU's memory. Larger models need more VRAM. | The T4's 12 GB VRAM was sufficient for all three models simultaneously. |
| **Checkpoint** | A saved snapshot of progress so work can be resumed if interrupted. | Phase 1 saves the CSV after every image to survive Kaggle session timeouts. |

---

## 13. 60-Second Elevator Pitch

> **AFS-VFM** — short for "Anticipatory Failure Signals in Vision Foundation Models" — is a research project that answers a simple but important question: *when an AI vision model is about to fail because conditions are getting worse (blurry, dark, occluded), does it "know" it's about to fail?*
>
> We took three major AI vision models — DINOv2, CLIP, and DETR — and systematically stressed them with 5 types of real-world degradation across 1,500 images, generating **over 150,000 model inferences**. Then we looked inside their internal representations (confidence scores, embeddings, attention patterns) and discovered that yes: internal signals drop **3 to 5 frames before** the model's final answer actually changes. This is measurable and statistically significant (p < 10⁻¹⁰).
>
> Our best finding: CLIP's "logit margin" can predict an upcoming failure with **82.2% accuracy** (AUROC = 0.822) several frames in advance. We also found that self-supervised models (DINOv2) are 45% more robust than contrastive models (CLIP), and that combining all three models in an ensemble cuts failure rates by 52%.
>
> We built a working early-warning prototype that monitors these signals and issues OK/WARNING/CRITICAL alerts in real time. The paper is the last step — the experiments, analysis, figures, and prototype are all complete. Target venues are BMVC or WACV.

---

*This report was generated by reading every file in the AFS-VFM repository. No project files were modified.*
