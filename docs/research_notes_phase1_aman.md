# Research Notes — Phase 1: Image Degradation Engine
**Project:** Anticipatory Failure Signals in Vision Foundation Models (AFS-VFM)  
**Author:** Aman  
**Date:** 2026-03-18  
**Status:** Complete

---

## 1. Objective

Designed and implemented a deterministic image degradation engine that generates controlled sequences of progressively corrupted images. This module serves as the input stimulus generator for studying internal failure signals in vision foundation models (DINOv2, CLIP, DETR).

The core idea: instead of testing models on a single corrupted image, we produce a **20-frame temporal sequence** where degradation severity increases linearly from 0% to 100%. This allows us to pinpoint the exact frame where model predictions transition from correct to incorrect — the **failure boundary**.

---

## 2. Design Decisions

### 2.1 Why 20 Frames?
- Provides sufficient temporal resolution to observe gradual internal signal changes
- Balances granularity with computational efficiency for batch processing across large datasets
- Each frame represents a 5.26% increment in degradation severity (1/19 steps)

### 2.2 Why Deterministic Degradation?
- **Reproducibility** — essential for scientific claims; any researcher can regenerate identical sequences
- **Controlled variable isolation** — severity is the only changing variable; no stochastic noise
- **Comparability** — enables fair comparison across different model architectures on identical inputs

### 2.3 Why Linear Severity Scaling?
- Simplest monotonic mapping from frame index to corruption level
- Makes it straightforward to correlate internal signal changes with known severity values
- Future work can explore non-linear schedules if needed

---

## 3. Degradation Types Implemented

Five degradation types were chosen to cover distinct failure modes that vision models encounter in real-world deployment:

| # | Type | Real-World Analogue | Algorithm | Parameters |
|---|---|---|---|---|
| 1 | **Motion Blur** | Camera shake, moving objects | Horizontal convolution kernel via `cv2.filter2D` | Kernel length: 3 → 51 px |
| 2 | **Occlusion** | Objects blocking the view | Centred grey rectangle | Area coverage: 5% → 80% |
| 3 | **Lighting Reduction** | Nighttime, shadows, underexposure | HSV V-channel scaling | Brightness: 100% → 5% |
| 4 | **Scale Reduction** | Distance from camera, low resolution | `cv2.resize` + zero-padding | Dimensions: 100% → 10% |
| 5 | **Viewpoint Change** | Camera tilt, rotation | Affine warp via `cv2.warpAffine` | Rotation: 0° → 45° |

### Why These Five?
- **Motion blur** — tests robustness to spatial frequency loss
- **Occlusion** — tests whether models rely on local vs global features
- **Lighting** — tests sensitivity to luminance/contrast reduction
- **Scale** — tests multi-scale feature extraction capability
- **Viewpoint** — tests geometric invariance of learned representations

These correspond to common corruptions studied in robustness benchmarks (ImageNet-C, COCO-C) but applied as **gradual sequences** rather than discrete severity levels.

---

## 4. Technical Implementation

### 4.1 Architecture

```
src/degradation/
├── __init__.py           # Public API exports
├── degradation.py        # DegradationPipeline class + orchestration
├── transformations.py    # 5 degradation algorithm implementations
└── utils.py              # Image I/O utilities
```

- **Separation of concerns** — algorithms are isolated from orchestration logic
- **Modular design** — new degradation types can be added without modifying existing code
- **Clean API** — single function call produces the full sequence

### 4.2 Pipeline Flow

```
Input Image (H, W, 3) RGB
        │
        ▼
  ┌─────────────────────────────┐
  │  For i = 0 to 19:           │
  │    severity = i / 19        │
  │    frame[i] = transform(    │
  │      image, severity        │
  │    )                        │
  └─────────────────────────────┘
        │
        ▼
Output Array (20, H, W, 3) uint8
```

### 4.3 Key Properties
- **Input format:** RGB uint8 images of any resolution
- **Output format:** NumPy array, shape `(20, H, W, 3)`, dtype `uint8`
- **Resolution preservation:** all frames maintain original image dimensions
- **Dependencies:** OpenCV (`cv2`) and NumPy only — no deep learning frameworks needed

---

## 5. Verification

### 5.1 Automated Tests
All 5 degradation types verified with a synthetic 256×256 test image:

```
[PASS]  blur          shape=(20, 256, 256, 3)  dtype=uint8
[PASS]  occlusion     shape=(20, 256, 256, 3)  dtype=uint8
[PASS]  lighting      shape=(20, 256, 256, 3)  dtype=uint8
[PASS]  scale         shape=(20, 256, 256, 3)  dtype=uint8
[PASS]  viewpoint     shape=(20, 256, 256, 3)  dtype=uint8
```

### 5.2 Visual Verification
Generated 10-frame grid visualisations for each degradation type confirming:
- Frame 1 matches the original image (severity = 0%)
- Degradation increases monotonically across frames
- Frame 20 shows maximum degradation (severity = 100%)
- No artifacts, crashes, or dimension mismatches

---

## 6. Integration API

For Maria's Track B pipeline:

```python
from degradation import generate_degradation_sequence

# Returns shape (20, H, W, 3), dtype uint8, RGB
frames = generate_degradation_sequence("path/to/image.jpg", "blur")
```

Supported type aliases: `"blur"`, `"motion_blur"`, `"occlusion"`, `"lighting"`, `"light"`, `"dark"`, `"scale"`, `"zoom"`, `"viewpoint"`, `"rotate"`

---

## 7. Limitations & Future Work

- Current degradation types are **spatial/photometric only** — no semantic corruptions (e.g. adversarial perturbations, style changes)
- Severity scaling is strictly linear — non-linear schedules may better capture perceptual thresholds
- Occlusion uses a fixed grey rectangle — future versions could use more realistic occlusion patterns
- Combined/compound degradations (e.g. blur + lighting) are not yet supported

---

## 8. Relevance to Research Question

> *"Do internal model signals become unstable before the final prediction fails?"*

This module provides the **controlled stimulus** needed to answer this question. By feeding each 20-frame sequence through DINOv2/CLIP/DETR and recording internal signals (embedding drift, attention entropy, logit margins) at every frame, we can determine whether instability precedes the failure frame — the central hypothesis of AFS-VFM.
