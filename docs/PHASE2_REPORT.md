# AFS-VFM Phase 2: Complete Report
## Statistical Analysis & Failure Mapping

---

## 1. What is Phase 2?

Phase 2 is the **analytical core** of the AFS-VFM research project. While Phase 1 ran 150,000 raw inferences to collect failure data, Phase 2 answers the deeper questions:

- **When** does each model fail? (Part 2A)
- **What's happening inside** the model before failure? (Part 2B)
- **Can we see failure coming** before it happens? (Part 2C)
- **How do the 3 models compare** in their failure patterns? (Part 2D)

### Models Studied

| Model | Architecture | Training | Task |
|-------|-------------|----------|------|
| **DINOv2** | ViT-B/14 | Self-supervised (DINO) | Image Classification |
| **CLIP** | ViT-B/32 | Contrastive Learning | Zero-shot Classification |
| **DETR** | ResNet-50 + Transformer | Supervised | Object Detection |

### Degradation Types

Each image was progressively degraded across 20 severity frames (0% to 100%) for:
- **Blur** - Gaussian blur with increasing kernel size
- **Lighting** - Brightness/contrast manipulation
- **Occlusion** - Progressive masking of image regions
- **Scale** - Image downscaling and upscaling
- **Viewpoint** - Perspective/affine transformations

---

## 2. Part 2A - Failure Boundary Analysis

### What We Did

Analyzed the existing 150,000 inference records from Phase 1 to determine the **exact severity frame** where each model's prediction first changes (fails).

### How We Did It

- Script: `src/analysis/failure_boundaries.py`
- Input: `full_benchmark.csv` (150K rows from Phase 1)
- Statistical tests: Kruskal-Wallis H-test, Mann-Whitney U with Bonferroni correction
- Transition matrices: tracking what class each model "hallucinates" when it fails

### Results

**22,500 failure boundary records** (1,500 images x 5 degradations x 3 models)

#### Survival Rates by Model and Degradation

| Degradation | DINOv2 Survival | CLIP Survival | DETR Survival | Most Robust |
|:-----------:|:---------------:|:-------------:|:-------------:|:-----------:|
| Blur | 35.8% | 34.9% | 9.5% | **DINOv2** |
| Lighting | 42.9% | 30.1% | 23.8% | **DINOv2** |
| Occlusion | 51.3% | 24.1% | 14.4% | **DINOv2** |
| Scale | 0.1% | 3.7% | 0.0% | **CLIP** |
| Viewpoint | 39.5% | 24.0% | 34.7% | **DINOv2** |
| **Average** | **33.9%** | **23.4%** | **16.5%** | **DINOv2** |

#### Mean Failure Frame (later = more robust)

| Degradation | DINOv2 | CLIP | DETR |
|:-----------:|:------:|:----:|:----:|
| Blur | 9.5 | 7.3 | 6.6 |
| Lighting | 9.6 | 8.2 | 8.1 |
| Occlusion | 12.2 | 10.0 | 10.4 |
| Scale | 15.7 | 14.5 | 9.1 |
| Viewpoint | 11.9 | 10.1 | 9.3 |

### What This Means

- **DINOv2 is the most robust model**, ranking #1 in 4 out of 5 degradation types
- **DETR is the most fragile**, especially under blur (only 9.5% survival)
- **Scale is universally devastating** - nearly 0% survival for all models
- **Occlusion is DINOv2's strength** - it survives 51.3%

---

## 3. Part 2B - Internal Signal Probing

### What We Did

Extracted **internal model representations** (embeddings, attention weights, logit distributions, confidence scores) at every degradation frame. This is like looking inside the model's "brain" as the image degrades.

### How We Did It

- Script: `src/models/signal_extractor.py`
- Notebook: `AFS_VFM_Phase2B_SignalProbing.ipynb`
- Hardware: **Kaggle Tesla T4 GPU** (1.6 hours runtime)
- Subset: **300 strategically selected images**
- Total inferences: **90,000** (300 images x 5 degradations x 20 frames x 3 models)

### What We Extracted Per Frame

| Model | Signal | Dimension | Purpose |
|-------|--------|:---------:|---------|
| DINOv2 | CLS embedding | 768-d | Tracks how internal representation shifts |
| DINOv2 | Top-1 confidence | scalar | How sure is the model? |
| DINOv2 | Logit margin | scalar | Gap between top-1 and top-2 predictions |
| CLIP | Image embedding | 512-d | Tracks CLIP's visual representation |
| CLIP | Top-1 confidence | scalar | Zero-shot classification certainty |
| CLIP | Logit margin | scalar | Decision boundary sharpness |
| DETR | Detection confidence | scalar | How confident is the best detection? |
| DETR | Objectness score | scalar | Does DETR think it found any object? |

### Results

**30,000 signal rows** - **zero missing values, zero errors**.

#### Signal Degradation with Severity

| Severity | DINOv2 Cosine Sim | CLIP Cosine Sim | DINOv2 Confidence | DETR Detections |
|:--------:|:-----------------:|:---------------:|:-----------------:|:---------------:|
| 0% | 1.000 | 1.000 | 0.764 | 76.6 |
| 26% | 0.935 | 0.934 | 0.738 | 74.8 |
| 52% | 0.838 | 0.845 | 0.691 | 74.2 |
| 78% | 0.751 | 0.786 | 0.644 | 69.4 |
| 100% | 0.487 | 0.661 | 0.460 | 58.8 |

---

## 4. Part 2C - Anticipatory Signal Detection (CORE HYPOTHESIS)

### The Hypothesis

> "Internal model signals degrade measurably BEFORE the model's final prediction actually changes."

### VERDICT: HYPOTHESIS SUPPORTED

> **83.2%** of pre-failure windows show negative signal trends.
> Average AUROC for failure prediction: **0.700**

#### Signal Trajectory Around Failure Point

| Relative Frame | DINOv2 Cosine Sim | CLIP Cosine Sim |
|:--------------:|:-----------------:|:---------------:|
| -5 (5 before failure) | 0.902 | 0.925 |
| -4 | 0.882 | 0.915 |
| -3 | 0.855 | 0.899 |
| -2 | 0.812 | 0.881 |
| -1 (1 before failure) | 0.734 | 0.864 |
| **0 (failure frame)** | **0.580** | **0.810** |

#### Failure Predictability (AUROC)

| Model | Signal | AUROC | Lead Time |
|-------|--------|:-----:|:---------:|
| CLIP | Logit margin | **0.822** | 3.8 frames |
| CLIP | Top-1 confidence | **0.817** | 3.8 frames |
| DETR | Detection confidence | **0.816** | 3.4 frames |
| DINOv2 | Top-1 confidence | **0.773** | 4.2 frames |
| DINOv2 | Logit margin | **0.771** | 4.2 frames |

#### Survived vs Failed Divergence

| Model | First Significant Divergence | p-value |
|-------|:----------------------------:|:-------:|
| **DINOv2** | **Frame 2 (10% severity)** | 1.98e-10 |
| CLIP | Frame 5 (26% severity) | 2.45e-02 |
| DETR | No significant divergence | - |

---

## 5. Part 2D - Cross-Model Comparative Analysis

### Robustness Ranking

| Rank | Model | Avg Survival | Training |
|:----:|-------|:------------:|----------|
| #1 | **DINOv2** | **33.9%** | Self-supervised |
| #2 | CLIP | 23.4% | Contrastive |
| #3 | DETR | 16.5% | Supervised |

### Architecture Insight

Self-supervised training (DINOv2) produces **45% more robust features** than contrastive training (CLIP), despite both using ViT backbones.

### Ensemble Potential

| Strategy | Avg Survival |
|----------|:------------:|
| All 3 ensemble | **51.6%** |
| DINOv2 + CLIP | 45.9% |
| DINOv2 alone | 33.9% |

### Failure Correlation

CLIP and DETR fail on **89% of the same images** - low complementarity.
DINOv2 is more independent - best candidate for ensembling.

---

## 6. Final Conclusions

1. Vision foundation models exhibit **predictable failure patterns**
2. Internal representations drift **3-5 frames before failure**
3. **Self-supervised training** (DINOv2) produces the most robust features
4. **CLIP's logit margin** is the single best failure predictor (AUROC=0.822)
5. **DINOv2 shows earliest self-awareness** - diverges at frame 2 (10% severity)
6. **Model ensembles** reduce failure rates by 52% (33.9% to 51.6%)
