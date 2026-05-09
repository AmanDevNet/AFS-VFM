# AFS-VFM Phase 3: Implementation Plan
## Visualization, Practical System & Research Paper

---

## What is Phase 3?

Phase 1 collected the data (150K inferences). Phase 2 analyzed it (failure boundaries, internal signals, hypothesis testing). **Phase 3 turns everything into publishable outputs** - the figures, the practical tool, and the final paper.

```
Phase 1 (DONE): Collect data          --> full_benchmark.csv
Phase 2 (DONE): Analyze data          --> hypothesis tested, findings proven
Phase 3 (NOW):  Visualize + Publish   --> figures, paper, working prototype
```

---

## The 3 Parts of Phase 3

| Part | What It Produces | Time | GPU? |
|------|-----------------|:----:|:----:|
| **3A** | 9 publication-quality figures | ~2-3 hours | No |
| **3B** | Working failure early warning prototype | ~2 hours | No |
| **3C** | Research paper + polished GitHub repo | ~3-4 hours | No |

**Everything runs locally on your laptop. No Kaggle needed.**

Parts 3A and 3B can run in **parallel** (no dependencies between them).
Part 3C depends on 3A being done (needs the figures for the paper).

```
    3A (Figures)  ──┐
                    ├──> 3C (Paper + Repo)
    3B (Prototype) ─┘
```

---

## Part 3A - Publication-Quality Visualizations

### Goal
Generate every figure needed for the paper, LinkedIn posts, and GitHub README.

### What gets built
`src/visualization/generate_figures.py`

### All 9 Figures

#### Figure 1: Robustness Heatmap
- **Type:** Seaborn heatmap (3 models x 5 degradation types)
- **Cell values:** Survival rate (%)
- **Color:** Green = robust, Red = fragile
- **Purpose:** Shows at a glance which model breaks under which degradation
- **Data source:** `results/phase2a/failure_statistics.csv`

#### Figure 2: Failure Boundary Distributions
- **Type:** Violin/box plots grouped by model
- **X-axis:** Degradation type
- **Y-axis:** First failure frame (1-20)
- **Purpose:** Shows the FULL distribution of when models fail, not just the mean
- **Data source:** `results/phase2a/failure_boundaries.csv`

#### Figure 3: Signal Trajectory Around Failure (THE KEY FIGURE)
- **Type:** Line plot with shaded confidence bands
- **X-axis:** Relative frame (-5 to +5, where 0 = failure)
- **Y-axis:** Cosine similarity / confidence
- **Subplots:** One per model (DINOv2, CLIP)
- **Purpose:** Visually proves the core hypothesis - signals drop BEFORE failure
- **Data source:** `results/phase2c/failure_aligned_signals.csv`

#### Figure 4: AUROC Comparison Bar Chart
- **Type:** Grouped horizontal bar chart
- **Bars:** Each signal type per model
- **Length:** AUROC value (0 to 1)
- **Purpose:** Shows which signal best predicts failure
- **Data source:** `results/phase2c/predictability_metrics.csv`

#### Figure 5: ROC Curves
- **Type:** Multi-line plot with diagonal reference
- **Lines:** One curve per (model, signal) combination
- **Annotations:** AUC value per curve
- **Purpose:** Full ROC curves for the paper's methodology section
- **Data source:** Recomputed from `results/phase2c/merged_signals_boundaries.csv`

#### Figure 6: Survived vs Failed Divergence Timeline
- **Type:** Dual-line plot with shaded significance region
- **Lines:** Mean signal for survived images vs failed images
- **X-axis:** Frame (1-20)
- **Y-axis:** Signal value
- **Purpose:** Shows WHEN survived and failed groups diverge
- **Data source:** `results/phase2c/survived_vs_failed.csv`

#### Figure 7: Ensemble Survival Bar Chart
- **Type:** Horizontal bar chart
- **Bars:** Single models, 2-model ensembles, 3-model ensemble
- **Purpose:** Shows the practical benefit of ensembling
- **Data source:** `results/phase2d/ensemble_complementarity.csv`

#### Figure 8: Failure Correlation Matrix
- **Type:** Annotated heatmap or chord diagram
- **Values:** Jaccard similarity between model failure sets
- **Purpose:** Shows which models fail on the same images
- **Data source:** `results/phase2d/failure_correlation_matrix.csv`

#### Figure 9: Architecture Comparison Radar Chart
- **Type:** Radar/spider chart with 5 axes
- **Axes:** Robustness, Self-awareness, Best AUROC, Lead time, Survival rate
- **Polygons:** One per model, overlaid
- **Purpose:** Holistic visual comparison for the paper's conclusion
- **Data source:** `results/phase2d/architecture_comparison.csv`

### Output Location
```
results/figures/
├── fig1_robustness_heatmap.png
├── fig2_failure_distributions.png
├── fig3_signal_trajectory.png        <-- THE key figure
├── fig4_auroc_comparison.png
├── fig5_roc_curves.png
├── fig6_divergence_timeline.png
├── fig7_ensemble_benefit.png
├── fig8_failure_correlation.png
└── fig9_architecture_radar.png
```

---

## Part 3B - Failure Early Warning System

### Goal
Build a working prototype that monitors internal model signals during inference and **triggers a warning when failure is predicted**. This demonstrates the practical application of our research findings.

### What gets built

```
src/earlywarning/
├── __init__.py
├── learn_thresholds.py           <-- Learns optimal thresholds from Phase 2C data
├── failure_predictor.py          <-- The predictor class
└── thresholds.json               <-- Learned threshold config
```

### How It Works

```
Input Image
     |
     v
  [Model Inference]
     |
     v
  [Extract Internal Signals]
  - cosine_sim vs baseline
  - logit_margin
  - top1_confidence
     |
     v
  [Compare to Learned Thresholds]
     |
     +---> Signal above threshold --> STATUS: OK
     |
     +---> Signal below threshold --> STATUS: WARNING
     |     "Failure predicted in ~3 frames"
     |
     +---> Signal far below threshold --> STATUS: CRITICAL
           "Failure imminent"
```

### Step 1: Learn Thresholds
- Input: `results/phase2c/prefailure_window_stats.csv`
- Method: Find the signal value at each frame before failure that maximizes F1-score for binary prediction (will fail vs won't fail)
- Output: `thresholds.json` with optimal thresholds per model per signal

### Step 2: Build Predictor Class
```python
class FailurePredictor:
    def __init__(self, thresholds_path):
        """Load learned thresholds."""
        
    def assess(self, model_name, signals_dict) -> dict:
        """
        Given current signal values, returns:
        {
            "status": "OK" | "WARNING" | "CRITICAL",
            "failure_probability": 0.73,
            "estimated_frames_to_failure": 3,
            "triggering_signals": ["logit_margin", "cosine_sim"],
            "recommendation": "Consider fallback model"
        }
        """
```

### Step 3: Demo Script
A script that:
1. Takes a sample image
2. Progressively degrades it (20 frames)
3. At each frame, runs the predictor
4. Prints a timeline showing warnings appearing BEFORE failure
5. Generates a visualization

---

## Part 3C - Research Paper & Repository Polish

### Goal
Produce a professional research paper and clean up the GitHub repository.

### Paper Structure

```
Title: "Anticipatory Failure Signals in Vision Foundation Models:
        Predicting Model Breakdown Under Visual Degradation"

Authors: Aman Sharma, Angel Taneja

1. Abstract (~200 words)
   - Problem, method, key findings, conclusion

2. Introduction (~500 words)
   - Problem: VFMs deployed without understanding failure
   - Gap: No systematic study of internal signal behavior before failure
   - Contribution: First study showing anticipatory failure signals

3. Related Work (~400 words)
   - Vision Foundation Models
   - Model Robustness Studies
   - Failure Prediction in Neural Networks

4. Methodology (~800 words)
   - 4.1 Benchmark Design
   - 4.2 Degradation Pipeline (5 types, 20 severity levels)
   - 4.3 Signal Extraction (embeddings, logits, confidence)
   - 4.4 Anticipatory Signal Detection Method

5. Experimental Setup (~300 words)
   - Models: DINOv2, CLIP, DETR
   - Datasets: COCO val, ImageNet val (1,500 images)
   - Hardware: Kaggle T4 GPU
   - Scale: 150K Phase 1 + 90K Phase 2B inferences

6. Results (~1000 words)
   - 6.1 Robustness Rankings (Table + Fig 1)
   - 6.2 Anticipatory Signals - Hypothesis Test (Fig 3)
   - 6.3 Failure Predictability - AUROC (Fig 4, 5)
   - 6.4 Survived vs Failed Divergence (Fig 6)
   - 6.5 Architecture & Training Effects (Fig 9)
   - 6.6 Ensemble Analysis (Fig 7)

7. Discussion (~500 words)
   - Why self-supervised models are more robust
   - Why CLIP's logit margin is the best predictor
   - Practical implications
   - Limitations

8. Conclusion (~200 words)

References (~20-30 citations)
```

### Repository Polish Checklist

```
[ ] Remove temporary files (verify_phase2b.py, etc.)
[ ] Update README.md with Phase 2 results and figures
[ ] Add docs/PHASE2_REPORT.md (already created)
[ ] Add docs/PHASE3_PLAN.md (this file)
[ ] Create RESULTS.md with key findings summary
[ ] Update requirements.txt with exact versions (add scikit-learn)
[ ] Add proper docstrings to all analysis modules
[ ] Verify all scripts run from clean clone
[ ] Add the paper PDF to the repository
[ ] Update GitHub repo description and tags
```

---

## Execution Order

| Step | Part | What to Do | Time |
|:----:|------|------------|:----:|
| 1 | **3A** | Generate all 9 figures | ~2-3 hours |
| 2 | **3B** | Build early warning prototype | ~2 hours |
| 3 | **3C** | Write paper using figures from 3A | ~3-4 hours |
| 4 | **3C** | Polish GitHub repo | ~1 hour |

### Verification

- **3A:** All figures render, values match Phase 2 CSVs, consistent styling
- **3B:** Predictor correctly warns before failure on test images
- **3C:** Paper is complete, repo clones and runs cleanly

---

## Files Created After Phase 3

```
d:\Research Project\
├── src/
│   ├── visualization/
│   │   └── generate_figures.py          <-- NEW (Part 3A)
│   └── earlywarning/
│       ├── __init__.py                  <-- NEW (Part 3B)
│       ├── learn_thresholds.py          <-- NEW (Part 3B)
│       ├── failure_predictor.py         <-- NEW (Part 3B)
│       └── thresholds.json              <-- NEW (Part 3B)
├── results/
│   └── figures/
│       ├── fig1_robustness_heatmap.png  <-- NEW (Part 3A)
│       ├── fig2_failure_distributions.png
│       ├── fig3_signal_trajectory.png
│       ├── ... (9 figures total)
│       └── fig9_architecture_radar.png
├── docs/
│   ├── PHASE2_REPORT.md                 <-- Already created
│   ├── PHASE3_PLAN.md                   <-- This file
│   └── paper/
│       └── AFS_VFM_Paper.pdf            <-- NEW (Part 3C)
└── RESULTS.md                           <-- NEW (Part 3C)
```
