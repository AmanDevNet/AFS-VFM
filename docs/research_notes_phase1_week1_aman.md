# Research Notes — Phase 1, Week 1: Image Degradation Engine
**Project:** Anticipatory Failure Signals in Vision Foundation Models (AFS-VFM)  
**Author:** Aman  
**Week:** 1 (March 18–19, 2026)  
**Status:** Complete — merged to main

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
| 1 | **Motion Blur** | Camera shake, moving objects | Horizontal convolution kernel via `cv2.filter2D` | Kernel length: 1 → 51 px |
| 2 | **Occlusion** | Objects blocking the view | Centred grey rectangle | Area coverage: 0% → 80% |
| 3 | **Lighting Reduction** | Nighttime, shadows, underexposure | Direct RGB→HSV V-channel scaling | Brightness: 100% → 5% |
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
- **Frame 0 guarantee:** always pixel-identical to the original input image
- **Resolution preservation:** all frames maintain original image dimensions
- **Edge case safety:** handles `num_frames=1` without errors
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

### 5.2 Frame 0 Integrity Verification
```
blur frame0 == original:      True
occlusion frame0 == original: True
num_frames=1 (no crash):      True
```

### 5.3 Visual Verification
Generated 10-frame grid visualisations for each degradation type confirming:
- Frame 1 is pixel-identical to the original image (severity = 0%)
- Degradation increases monotonically across frames
- Frame 20 shows maximum degradation (severity = 100%)
- No artifacts, crashes, or dimension mismatches

---

## 6. Code Review & Bug Fixes

After initial implementation, Maria reviewed the code and identified 3 issues. All were fixed same week.

| # | Issue Found | Root Cause | Fix Applied |
|---|---|---|---|
| 1 | Frame 0 slightly blurred in motion_blur | Kernel started at 3px instead of 1px | Changed kernel range from `3→51` to `1→51`; added early return for kernel ≤ 1 |
| 2 | Frame 0 had 5% occlusion patch | Occlusion started at 5% instead of 0% | Changed range from `5%→80%` to `0%→80%`; added severity ≤ 0 guard |
| 3 | Unnecessary BGR round-trip in lighting | Used RGB→BGR→HSV→BGR→RGB (4 conversions) | Switched to direct RGB→HSV→RGB (2 conversions) using `cv2.COLOR_RGB2HSV` |
| 4 | ZeroDivisionError if num_frames=1 | `severity = i / (num_frames - 1)` divides by zero | Added guard: when `num_frames == 1`, severity defaults to 0.0 |

**Takeaway:** Frame 0 must always be the pristine original — this is the clean baseline for all model comparisons. This constraint was implicit in the research design but not enforced in the initial code.

---

## 7. Integration API

For Maria's Track B pipeline:

```python
from degradation import generate_degradation_sequence

# Returns shape (20, H, W, 3), dtype uint8, RGB
frames = generate_degradation_sequence("path/to/image.jpg", "blur")
```

Supported type aliases: `"blur"`, `"motion_blur"`, `"occlusion"`, `"lighting"`, `"light"`, `"dark"`, `"scale"`, `"zoom"`, `"viewpoint"`, `"rotate"`

---

## 8. Git Activity

| Date | Commit | Description |
|---|---|---|
| March 18 | `d276bc9` | Initial implementation — full degradation engine (Track A Phase 1) |
| March 19 | `b19484c` | Bug fixes — frame 0 clean, direct RGB-HSV, ZeroDivisionError guard |
| March 19 | `b56ff38` | Merged `feature/degradation-engine` → `main` |

---

## 9. Limitations & Future Work

- Current degradation types are **spatial/photometric only** — no semantic corruptions (e.g. adversarial perturbations, style changes)
- Severity scaling is strictly linear — non-linear schedules may better capture perceptual thresholds
- Occlusion uses a fixed grey rectangle — future versions could use more realistic occlusion patterns
- Combined/compound degradations (e.g. blur + lighting) are not yet supported

---

## 10. Relevance to Research Question

> *"Do internal model signals become unstable before the final prediction fails?"*

This module provides the **controlled stimulus** needed to answer this question. By feeding each 20-frame sequence through DINOv2/CLIP/DETR and recording internal signals (embedding drift, attention entropy, logit margins) at every frame, we can determine whether instability precedes the failure frame — the central hypothesis of AFS-VFM.
