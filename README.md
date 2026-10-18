<div align="center">

# 🔬 AFS-VFM

### Anticipatory Failure Signals in Vision Foundation Models

*Systematically stress-testing state-of-the-art AI vision models to discover exactly when — and why — they break.*

> **Research into anticipatory failure signals in vision foundation models — DINOv2 (Base/Large), CLIP, SigLIP, DETR. Targeting BMVC/WACV.**

`computer-vision` • `vision-transformers` • `deep-learning` • `pytorch` • `huggingface` • `research` • `dinov2` • `clip` • `siglip` • `detr` • `robustness` • `early-warning`

<br/>

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Phase%203%20Early%20Warning%20Active-blue)]()

<br/>

**150,000+ model inferences** · **5 foundation models** · **5 core degradations + adversarial & semantic suite** · **1,500 benchmark images**

</div>

---

## 📌 Table of Contents

- [The Problem](#-the-problem)
- [Our Approach](#-our-approach)
- [Supported Foundation Models](#-supported-foundation-models)
- [System Architecture](#-system-architecture)
- [Degradation Engine (Track A)](#-degradation-engine--track-a)
- [Model Inference Pipeline (Track B)](#-model-inference-pipeline--track-b)
- [Phase 3 Early Warning System](#-phase-3-early-warning-system)
- [Key Results So Far](#-key-results-so-far)
- [Project Roadmap](#-project-roadmap)
- [Tech Stack](#-tech-stack)
- [Contributors](#-contributors)

---

## 🧠 The Problem

Vision Foundation Models (VFMs) like **DINOv2**, **CLIP**, **SigLIP**, and **DETR** achieve near-perfect accuracy on clean benchmark datasets. But the real world is messy — camera lenses get dirty, lighting drops, objects move, perspectives shift.

> **At what exact point does a state-of-the-art AI vision model stop *seeing* and start *guessing*?**

Current robustness benchmarks (ImageNet-C, COCO-C) test models at discrete corruption levels. They tell you *that* a model failed — but not *when* the internal signal crossed the tipping point. **AFS-VFM** fills that gap by isolating anticipatory signals that precede failure.

---

## 🎯 Our Approach

Instead of testing models on a single corrupted image, we generate a **20-frame temporal degradation sequence** where corruption intensity increases linearly from **0% to 100%**. This allows us to pinpoint the **exact frame** where each model's prediction transitions from correct to incorrect — the **failure boundary**.

We compare five architecturally distinct foundation models across diverse pre-training paradigms:

## 🤖 Supported Foundation Models

| Model | Architecture | Task | Paradigm | Dimensions / Head |
|-------|-------------|------|----------|-------------------|
| **DINOv2-Base** (`facebook/dinov2-base-imagenet1k-1-layer`) | ViT-B/14 | Image Classification | Self-Supervised Learning | 768-d |
| **DINOv2-Large** (`facebook/dinov2-large-imagenet1k-1-layer`) | ViT-L/14 | Image Classification | Self-Supervised Learning | 1024-d |
| **CLIP** (`openai/clip-vit-base-patch32`) | ViT-B/32 + Text Transformer | Zero-Shot Classification | Contrastive (Softmax) | 512-d |
| **SigLIP** (`google/siglip-base-patch16-224`) | ViT-B/16 + Text Transformer | Zero-Shot Classification | Pairwise Sigmoid BCE | 768-d |
| **DETR** (`facebook/detr-resnet-50`) | ResNet-50 + Transformer Decoder | Object Detection | Bipartite Matching Loss | 256-d |

---

## 🏗 System Architecture

The pipeline operates in two parallel tracks with a clean handshake protocol:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AFS-VFM BENCHMARK PIPELINE                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌───────────────────────────────────────────┐                         │
│   │       TRACK A: Degradation Engine         │                         │
│   │                                           │                         │
│   │   Input Image ──► DegradationPipeline     │                         │
│   │                     │                     │                         │
│   │       ┌─────────────┼─────────────┐       │                         │
│   │       ▼             ▼             ▼       │                         │
│   │   Motion Blur   Occlusion    Lighting     │                         │
│   │       ▼             ▼             ▼       │                         │
│   │     Scale      Viewpoint                  │                         │
│   │       │             │                     │                         │
│   │       └─────────────┼─────────────┘       │                         │
│   │                     ▼                     │                         │
│   │   Output: NumPy Array (20, H, W, 3)       │                         │
│   └───────────────────┬───────────────────────┘                         │
│                       │                                                 │
│              ─── Handshake Protocol ───                                  │
│                       │                                                 │
│   ┌───────────────────▼───────────────────────┐                         │
│   │       TRACK B: Inference Pipeline         │                         │
│   │                                           │                         │
│   │   Frame[0] ──► Establish Baseline         │                         │
│   │   Frame[N] ──► Run DINOv2 / CLIP / DETR   │                         │
│   │              ──► Compare to Baseline       │                         │
│   │              ──► Record is_failure         │                         │
│   │                     │                     │                         │
│   │   Output: CSV with failure annotations     │                         │
│   └───────────────────────────────────────────┘                         │
│                                                                         │
│   ┌───────────────────────────────────────────┐                         │
│   │       INFRASTRUCTURE                      │                         │
│   │   • Crash-safe CSV checkpointing          │                         │
│   │   • Auto-resume on restart                │                         │
│   │   • Batch splitting (3 × 500 images)      │                         │
│   │   • Kaggle/Colab T4 GPU compatible        │                         │
│   └───────────────────────────────────────────┘                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Degradation Engine — Track A

A **deterministic**, **reproducible** image corruption engine built from scratch. No randomness — the same input always produces the same output, which is essential for scientific reproducibility.

Each degradation type is designed to target a specific failure mode that vision models encounter in real-world deployment:

| # | Degradation | Real-World Scenario | Algorithm | Severity Range |
|---|-------------|-------------------|-----------|---------------|
| 1 | **Motion Blur** | Camera shake, moving objects | Horizontal convolution kernel (`cv2.filter2D`) | Kernel: 1 → 51 px |
| 2 | **Occlusion** | Objects blocking the field of view | Deterministic black rectangular patches | Coverage: 0% → 40% |
| 3 | **Lighting** | Nighttime, shadows, underexposure | RGB brightness + contrast scaling | Brightness: 100% → 5% |
| 4 | **Scale** | Low resolution, distance from camera | Downsample + nearest-neighbor upsample | Resolution: 100% → 2% |
| 5 | **Viewpoint** | Camera tilt, perspective distortion | Perspective warp (`cv2.warpPerspective`) | Pinch: 0% → 40% |

### Design Principles

- **Frame 0 Guarantee**: The first frame is always pixel-identical to the original image — this is the clean baseline for all model comparisons
- **Linear Severity**: Severity scales from `0.0` to `1.0` across 20 frames (5.26% per frame), making it straightforward to correlate signal changes with known corruption levels
- **Resolution Preservation**: All frames maintain the original image dimensions
- **Minimal Dependencies**: The engine uses only OpenCV and NumPy — no deep learning frameworks required

---

## 🤖 Model Inference Pipeline — Track B

Three vision foundation models are loaded via HuggingFace Transformers and evaluated in PyTorch inference mode:

```python
# DINOv2 — Classification via self-supervised features
outputs = dinov2_model(**processor(images=frame, return_tensors="pt"))
prediction = model.config.id2label[outputs.logits.argmax(-1).item()]

# CLIP — Zero-shot reasoning against 20 candidate labels
outputs = clip_model(**processor(text=labels, images=frame, return_tensors="pt"))
prediction = labels[outputs.logits_per_image.softmax(dim=1).argmax(-1).item()]

# DETR — Object detection with confidence scoring
outputs = detr_model(**processor(images=frame, return_tensors="pt"))
prediction = model.config.id2label[logits.softmax(-1).max(dim=1)[1][best_det].item()]
```

### Failure Detection Logic

1. **Frame 0** → Run all 3 models → Store each model's prediction as its **baseline**
2. **Frames 1–19** → Run all 3 models → Compare each prediction against its baseline
3. **`is_failure = True`** when the model's prediction **diverges** from the baseline

This captures the exact moment each model's internal reasoning breaks under increasing visual stress.

---

## 📊 Key Results So Far

### Phase 1 — Data Generation: ✅ Complete

We successfully processed **1,500 validation images** (500 from COCO val2017 + 1,000 from ImageNet-1k) across all 5 degradation types and generated exactly **150,000 model inferences** (1,500 × 5 × 20 frames captured at each of the 3 model outputs).

#### Early Observations (Week 2 — Motion Blur on DINOv2)

From our initial single-feature integration test:

| Frame | Severity | DINOv2 Prediction | Status |
|-------|----------|-------------------|--------|
| 1 | 0% | `analog clock` | ✅ Baseline |
| 2 | 5% | `chain mail` | ❌ **First failure** |
| 6 | 26% | `window screen` | ❌ Drifting |
| 9–20 | 42–100% | `ruler` | ❌ Locked on wrong class |

> **Key Insight**: DINOv2 failed at just **5% degradation intensity** under motion blur — indicating extreme sensitivity to high-frequency spatial features. The model didn't gradually lose confidence; it **snapped** to a completely wrong class almost immediately.

### Phase 2 — Statistical Analysis & Failure Mapping: ✅ Complete

We extracted **90,000 internal brain signals** (logits, embeddings, confidence scores) and calculated exact failure boundaries across all models.

#### Breakthrough Discoveries

1. **DINOv2 is the Most Robust:** Across all 5 degradation types, DINOv2 survived the longest (33.9% average survival), outperforming CLIP (23.4%) and DETR (16.5%).
2. **Anticipatory Signals Exist (Hypothesis Proven):** We mathematically proved that internal model signals drop **2 to 5 frames BEFORE** the model's final prediction actually changes.
3. **Failure is Predictable:** By analyzing specific internal metrics (like logit margins and cosine embedding shifts), we predict imminent AI failure with high statistical significance before output corruption occurs.

---

## ⚡ Phase 3 Early Warning System: ✅ Active

The Phase 3 early warning system continuously monitors internal foundation model health during visual degradation and raises alerts prior to task-level failure:

```
  Clean Visual Stream
           │
           ▼
  Internal Signal Extraction (Confidence, Logit Margins, Cosine Drift)
           │
           ▼
  Threshold Predictor (Learned via ROC/AUC on empirical failure trajectories)
           │
     ┌─────┴──────────────┐
     ▼                    ▼
  [OK]            [WARNING / CRITICAL]
  (Healthy)       (2–4 frames BEFORE prediction divergence)
```

### Running the Early Warning Demo

Run the interactive multi-model simulation across all 5 foundation models:

```bash
# Run demo for all 5 foundation models
python -m src.earlywarning.demo --models dinov2 clip detr siglip dinov2_large

# Run regression test suite
python verify_early_warning.py
```

### Running Benchmark with Flexible Model Selection

```bash
# Quick pilot test on a subset of images
python main.py --pilot --models dinov2 clip siglip

# Run specific batch with all 5 models
python main.py --batch 1 --models dinov2 dinov2_large clip siglip detr
```

---

## 🗺 Project Roadmap

```text
Phase 1 — Infrastructure & Data Generation ███████████████████████████████ 100%  ✅
Phase 2 — Statistical Analysis & Failure Mapping ███████████████████████████████ 100%  ✅
Phase 3 — Early Warning System & Paper ███████████████████████████████  85%   🚀
```

| Phase | Milestone | Status | Description |
|-------|-----------|--------|-------------|
| **1.1** | Degradation Engine | ✅ Complete | 5-type deterministic corruption engine + semantic/adversarial extensions |
| **1.2** | Model Loader | ✅ Complete | Dynamic loading for DINOv2 (Base/Large), CLIP, SigLIP, DETR |
| **1.3** | Integration Test | ✅ Complete | End-to-end pipeline validation (Track A ↔ Track B) |
| **1.4** | Full Benchmark Run | ✅ Complete | 150,000+ inferences across 1,500 images |
| **2.1** | Failure Boundary Analysis | ✅ Complete | Dynamic model discovery and per-degradation breaking points |
| **2.2** | Cross-Model Comparison | ✅ Complete | Statistical comparison across self-supervised, contrastive, and detection paradigms |
| **2.3** | Internal Signal Probing | ✅ Complete | Embedding drift, attention entropy, logit margin analysis |
| **3.1** | Visualization Dashboard | ✅ Complete | Publication-quality timeline figures and early warning overlays |
| **3.2** | Early Warning Prototype | ✅ Complete | Multi-model health predictor with calibrated threshold trigger engine |
| **3.3** | Research Paper | 📝 In Progress | Formal write-up for BMVC/WACV submission |

---

## 🛠 Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.10+ |
| **Deep Learning** | PyTorch 2.0+, HuggingFace Transformers |
| **Models** | Meta DINOv2, OpenAI CLIP (ViT-B/32), Meta DETR (ResNet-50) |
| **Computer Vision** | OpenCV, NumPy, Pillow |
| **Datasets** | COCO val2017, ImageNet-1k (via HuggingFace Datasets) |
| **Compute** | Kaggle T4 GPU (free tier), Google Colab compatible |
| **Version Control** | Git, GitHub (multi-contributor workflow with feature branches) |

---

## 👥 Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/AmanDevNet">
        <img src="https://github.com/AmanDevNet.png" width="120px;" alt="Aman Sharma"/><br />
        <b>Aman Sharma</b>
      </a><br />
      <a href="https://www.linkedin.com/in/aman-sharma-842b66318"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white" alt="LinkedIn"/></a><br />
      <sub>Lead Author</sub><br />
      <sub>
        Degradation Engine Architecture · Model Inference Integration ·<br/>
        Batch Processing Infrastructure · Dataset Pipeline · Orchestration
      </sub>
    </td>
    <td align="center">
      <a href="https://github.com/angeltaneja">
        <img src="https://github.com/angeltaneja.png" width="120px;" alt="Angel Taneja"/><br />
        <b>Angel Taneja</b>
      </a><br />
      <a href="https://www.linkedin.com/in/angel-taneja-140b45263"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white" alt="LinkedIn"/></a><br />
      <sub>Co-Author</sub><br />
      <sub>
        Research Paper Authoring · Visualization & Analytics ·<br/>
        Testing · DINOv2/CLIP/DETR Integration · Failure Detection Logic · Quality Assurance · Code Review
      </sub>
    </td>
  </tr>
</table>

---

## 📄 License

This project is open-sourced under the [MIT License](LICENSE).

---

## 🔗 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@misc{sharma2026afsvfm,
  title   = {AFS-VFM: Anticipatory Failure Signals in Vision Foundation Models},
  author  = {Sharma, Aman and Taneja, Angel},
  year    = {2026},
  url     = {https://github.com/AmanDevNet/AFS-VFM}
}
```

---

<div align="center">

**🔬 Research in Progress — Phase 3 (Visualization & Early Warning Prototype) currently underway.**

*Built with curiosity, broken models, and too many GPU hours.*

</div>
