# AFS-VFM: Anticipatory Failure Signals in Vision Foundation Models

Vision foundation models exhibit measurable internal signal shifts prior to overt behavioral classification or detection failure under environmental degradation. This project empirically benchmarks and analyzes these precursor signals to quantify warning lead times and failure predictability across distinct model architectures.

---

## 1. Key Results

All values reported below are computed directly from the repository's verified evaluation artifacts.

### Benchmark Scale & Dataset
- **Evaluation Dataset:** 1,500 total base images (1,000 from ImageNet-1K validation, 500 from MS-COCO val2017). *(Source: `results/full_benchmark.csv`)*
- **Evaluated Architectures:** DINOv2-Base (ViT-B/14, self-supervised), CLIP (ViT-B/32, contrastive language-image), and DETR (ResNet-50 backbone, supervised detection).
- **Degradation Profiles:** 5 corruption types (motion blur, lighting reduction, occlusion, scale reduction, viewpoint shift) evaluated across 20 severity frames (0% to 100% severity).
- **Verified Record Count:** Exactly **150,000 rows** in `results/full_benchmark.csv` (1,500 images $\times$ 5 degradations $\times$ 20 frames).

### Model Survival Rates
Survival denotes preserving the clean-frame (frame 0) prediction across the 20-frame degradation progression. *(Source: `results/phase2a/failure_statistics.csv`)*

| Model | Architecture & Objective | Overall Survival Rate | Blur | Lighting | Occlusion | Scale | Viewpoint |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **DINOv2-Base** | ViT-B/14 (Self-Supervised) | **33.9%** | 35.8% | 42.9% | 51.3% | 0.1% | 39.5% |
| **CLIP** | ViT-B/32 (Contrastive Vision-Language) | **23.4%** | 34.9% | 30.1% | 24.1% | 3.7% | 24.0% |
| **DETR** | ResNet-50 + Transformer (Supervised) | **16.5%** | 9.5% | 23.8% | 14.4% | 0.0% | 34.7% |

*Associated visualization: `results/figures/fig1_robustness_heatmap.png`*

### Pre-Failure Trajectory Trend
Across all evaluated models and internal feature signals, **83.2%** of pre-failure observation windows display a negative signal slope (downward trajectory) prior to the first behavioral failure frame. *(Source: `results/phase2c/phase2c_report.txt`, `results/phase2c/prefailure_window_stats.csv`)*

*Associated visualization: `results/figures/fig3_signal_trajectory.png`*

### Predictability & Warning Lead Time
Lead time reflects the mean number of frames between an anticipatory signal crossing a 2-standard-deviation warning threshold and the occurrence of behavioral failure. *(Source: `results/phase2c/predictability_metrics.csv`)*

| Model | Evaluated Signal | AUROC | Mean Lead Time (Frames) |
|:---|:---|:---:|:---:|
| **CLIP** | Logit Margin ($\Delta \text{logit}_{1-2}$) | **0.822** | **3.8** |
| **CLIP** | Top-1 Confidence Score | **0.817** | **3.8** |
| **DETR** | Top Detection Confidence | **0.816** | **3.4** |
| **DINOv2-Base** | Top-1 Confidence Score | **0.773** | **4.2** |
| **DINOv2-Base** | Logit Margin ($\Delta \text{logit}_{1-2}$) | **0.771** | **4.2** |
| **DINOv2-Base** | Latent Cosine Similarity | **0.519** | **4.8** |
| **CLIP** | Latent Cosine Similarity | **0.380** | **4.8** |

*Associated visualization: `results/figures/fig4_auroc_comparison.png`*

### Earliest Statistically Significant Signal Divergence
Evaluating the divergence of internal signals between surviving and failing trajectories via two-sided Mann-Whitney U testing *(Source: `results/phase2c/phase2c_report.txt`, `results/phase2c/survived_vs_failed.csv`)*:
- **DINOv2-Base:** Internal feature divergence reaches statistical significance at **Frame 2** (10.5% degradation severity; Cohen's $d = 0.315$, $p = 1.98 \times 10^{-10}$).
- **CLIP:** Internal confidence divergence reaches statistical significance at **Frame 5** (26.3% degradation severity; Cohen's $d = -0.126$, $p = 2.45 \times 10^{-2}$).
- **DETR:** Shows no statistically significant divergence prior to failure under baseline configuration ($p > 0.05$).

*Associated visualization: `results/figures/fig6_divergence_timeline.png`*

---

## 2. Caveats & Scope of Baseline Results

> **Important Evaluation Scope Note:**  
> The baseline CLIP metrics reported above (including AUROC 0.822) were computed using a restricted 20-candidate label subset (`CLIP_CANDIDATE_LABELS` in legacy evaluation; Audit Item L10). This restricted label space constrains the softmax competition and inflates confidence values relative to a full classification benchmark.  
> 
> A corrected full 1,000-class ImageNet-1K evaluation pipeline utilizing pre-computed text embeddings across 80 prompt templates has been implemented in `src/models/clip_imagenet1k.py` and validated in `tests/verify_clip_1000.py`. **The baseline AUROC table in `results/phase2c/predictability_metrics.csv` was computed BEFORE this 1,000-class fix was implemented.** Re-evaluation on the full class set is currently scheduled as part of the 10,000-image benchmark run.

---

## 3. Project Status

| Milestone / Component | Current Status | Details |
|:---|:---:|:---|
| **1,500-Image Baseline Benchmark** | **Complete** | 150,000 rows across 3 models and 5 degradations (`results/full_benchmark.csv`). |
| **10,000-Image Scaled Benchmark** | **In Progress** | Distributed sharding framework operational. Shard 0 of 20 complete (500 images, 50,000 rows in `results/shards/shard_000_of_020.csv`). No pooled 10k results claimed. |
| **Learned Multi-Signal Predictor** | **Implemented (Evaluation Pending)** | Tabular models (Gradient Boosting, Random Forest, MLP) implemented in `src/earlywarning/learned_predictor.py`. Systematic test-set AUROC evaluation on full data pending. |
| **Research Manuscript** | **In Preparation** | Target submission for peer review. |

---

## 4. Method Overview

```
Input Image (Frame 0: Clean)
       │
       ▼
Degradation Engine (Linear / Non-Linear / Compound) ──► 20-Frame Severity Sequence
       │
       ▼
Model Forward Passes (DINOv2, CLIP, DETR)
       ├──► Behavioral Outputs (Class Prediction, Bounding Boxes) ──► Failure Boundaries
       └──► Internal Signal Probing (Confidence, Margin, Latent Drift, Entropy)
              │
              ▼
Anticipatory Early Warning Predictors (Heuristic Thresholds / CUSUM / Learned Classifiers)
```

1. **Environmental Degradation Sequences:** Images undergo continuous degradation across 20 discrete frames parameterized from 0% (unmodified original) to 100% severity.
2. **Behavioral Failure Tracking:** The model's clean prediction (Frame 0) defines the reference identity. Behavioral failure is identified at the earliest frame where prediction changes.
3. **Internal Signal Probing:** Multi-layer internal signals are logged at each frame:
   - Output decisiveness: Top-1 confidence, top 1-vs-2 logit margin, prediction entropy.
   - Representation geometry: Cosine similarity and $L_2$ distance of image embeddings relative to Frame 0.
4. **Early Warning Detection:** Anticipatory signals are monitored to trigger warnings when deviating beyond baseline stability intervals, yielding lead times prior to behavioral breakdown.

---

## 5. Reproduction & Quickstart

### Environment Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/AmanSharma-1/AFS-VFM.git
cd AFS-VFM
pip install -r requirements.txt
```

### Smoke Test (No Dataset Download Required)
Verify the installation and core pipeline using the bundled sample data:
```bash
python tests/test_smoke.py
```

### Running Individual Benchmark Shards
The benchmark dataset is partitioned into 20 deterministic shards of 500 images each. To run an individual shard (e.g., shard 0):
```bash
python main.py --shard_id 0 --num_shards 20
```

Completed shard CSVs are merged into the consolidated benchmark via:
```bash
python merge_shards.py --input_dir results/shards --output results/full_benchmark_10k.csv
```

---

## 6. Repository Map

```
AFS-VFM/
├── src/                               # Core source modules
│   ├── degradation/                  # Degradation operators, non-linear schedules, compound chains
│   │   ├── adversarial.py            # Untargeted PGD and atmospheric synthesis
│   │   ├── compound.py               # Multi-stress degradation pipelines
│   │   ├── degradation.py            # Primary 20-frame sequence generation engine
│   │   ├── scheduler.py              # Linear, exponential, logarithmic, sigmoid, step schedules
│   │   └── transformations.py        # Core spatial and photometric transforms (blur, lighting, smudge)
│   ├── earlywarning/                 # Anticipatory signal monitoring and prediction
│   │   ├── failure_predictor.py      # Threshold-based predictor and CUSUM hazard detector
│   │   ├── feature_extractor.py      # 8-dimensional dynamic trajectory feature engineering
│   │   ├── learned_predictor.py      # Multi-signal machine learning classifier wrappers
│   │   ├── models/                   # Serialized predictor weights (joblib)
│   │   └── thresholds.json           # Calibrated empirical warning thresholds
│   ├── models/                       # Model loading and representation extraction
│   │   ├── clip_imagenet1k.py        # 1,000-class zero-shot ImageNet evaluator with cached text embeddings
│   │   ├── model_loader.py           # Unified loader for DINOv2, CLIP, DETR, SigLIP, DINOv2-Large
│   │   └── signal_extractor.py       # Internal feature, margin, entropy, and cosine drift extractor
│   ├── data/                         # Dataset utilities and class mappings
│   ├── utils/                        # Logging and runtime utilities
│   └── visualization/                # Figure generation and plotting scripts
├── tests/                             # Unit tests and limitation verification scripts
│   ├── test_smoke.py                 # Smoke test on bundled sample data
│   ├── verify_adversarial_and_semantic.py # PGD and weather transform verification
│   ├── verify_clip_1000.py           # 1,000-class CLIP evaluation verification
│   ├── verify_compound.py            # Multi-stress compound degradation verification
│   ├── verify_detr_fix.py            # DETR dual-representation pooling and drift verification
│   ├── verify_early_warning.py       # Threshold-based early warning verification
│   ├── verify_learned_early_warning.py # Learned classifier inference verification
│   ├── verify_model_diversity.py     # SigLIP and DINOv2-Large loader verification
│   ├── verify_occlusion.py           # Translucent organic smudge verification
│   ├── verify_phase2b.py             # Internal signal extraction verification
│   └── verify_schedules.py           # Severity schedule verification
├── sample_data/                       # Sample images and benchmark slice for smoke testing
│   ├── images/                       # 5 representative test images (clean and degraded)
│   └── sample_benchmark.csv          # 20-row sample trajectory
├── results/                           # Verified empirical evaluation outputs
│   ├── full_benchmark.csv            # 150,000-row baseline evaluation table
│   ├── shards/                       # Distributed shard outputs (shard_000_of_020.csv)
│   ├── figures/                      # Evaluated and cited benchmark figures
│   ├── phase2a/                      # Failure statistics, transition matrices, and boundary tests
│   ├── phase2b/                      # Internal signal extractions (frame_signals.csv)
│   ├── phase2c/                      # Predictability metrics, lead times, and divergence tests
│   └── phase2d/                      # Cross-model correlations and architectural synthesis
├── docs/                              # Technical documentation
│   └── AUDIT.md                      # Code-grounded limitations and resolution audit matrix
├── main.py                            # Primary benchmark execution script with sharding support
├── merge_shards.py                    # Shard aggregation utility
├── requirements.txt                   # Minimal pinned project dependencies
└── LICENSE                            # MIT License
```

---

## 7. Limitations & Open Challenges

A detailed audit of all repository limitations is maintained in [`docs/AUDIT.md`](docs/AUDIT.md). Key items include:

| Item | Focus Area | Status | Summary |
|:---|:---|:---:|:---|
| **L1** | Adversarial & Semantic Stresses | **Solved** | Implemented PGD and atmospheric weather generation. |
| **L2** | Severity Progression Schedules | **Solved** | Added 5 non-linear severity schedules. |
| **L3** | Realistic Occlusion | **Solved** | Implemented translucent organic smudges with alpha blending. |
| **L4** | Compound Degradations | **Solved** | Implemented multi-stress degradation pipelines. |
| **L5** | Model Architecture Diversity | **Solved** | Extended loaders to support SigLIP and DINOv2-Large. |
| **L6** | Benchmark Scaling (10k images) | **In Progress** | Shard 0 of 20 verified (50,000 rows); Shards 1–19 queued. |
| **L7** | DETR Embedding & Drift Tracking | **Solved** | Implemented dual-representation spatial and query pooling. |
| **L8** | Learned Multi-Signal Predictor | **Solved** | Tabular classifiers implemented; benchmark evaluation pending. |
| **L9** | Temporal Change-Point Modeling | **Solved** | Integrated CUSUM hazard detection. |
| **L10** | CLIP 1,000-Class Generalization | **Solved** | Implemented full ImageNet-1K cached text pipeline. Baseline metrics reflect legacy 20-label subset. |

---

## 8. Safety & Alignment Relevance

Safety-critical autonomous systems rely on vision models operating under variable, out-of-distribution real-world conditions. Standard deployments treat these models as black boxes that either output a correct prediction or fail silently without advance notice.

This research demonstrates that internal representation geometry and output decisiveness metrics show detectable degradation several steps before overt behavioral classification failure occurs. Developing reliable internal-state monitoring mechanisms provides a principled foundation for runtime out-of-distribution detection, failure warning, and automated fail-safe handovers to backup systems or human operators.

---

## 9. Authors & Contact

- **Aman Sharma** (Lead Researcher) — `amansharma9373@gmail.com`
- **Angel Taneja** (Co-Author)

---

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.
