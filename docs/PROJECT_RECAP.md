# The Complete AFS-VFM Project Guide 
*(An Exhaustive, File-by-File Guide for Angel)*

Welcome back! This document contains literally everything about our research project, **AFS-VFM (Anticipatory Failure Signals in Vision Foundation Models)**. It explains our goals, the tools we used, every single file in our folder, why it exists, and the exact results we got in each phase. 

It is written simply so you can catch up immediately and understand the exact purpose of every single piece of code and data we have.

---

## 🎯 1. The Ultimate Goal of Our Research
Artificial Intelligence (AI) vision models look really smart in perfect conditions, but they break when they see messy, real-world images (blurry, dark, blocked views). We wanted to find out:
1. **When do they break?**
2. **Why do they break?**
3. **Can we look inside the AI's "brain" and catch a warning signal BEFORE it makes a mistake?** (Just like a "check engine" light in a car).

---

## 🛠️ 2. What We Used (The Tech & Tools)
Before we look at the phases, here is exactly what we used to build this project:

- **The 3 AI Models We Tested:**
  1. **DINOv2 (by Meta):** A self-taught AI model (Self-Supervised Learning).
  2. **CLIP (by OpenAI):** An AI trained on text and images together.
  3. **DETR (by Meta):** An AI trained specifically to draw boxes around objects.
- **The Images (Datasets):** We used 1,500 real-world images. 500 from a famous dataset called **COCO**, and 1,000 from **ImageNet**.
- **The Hardware (GPUs):** Running 150,000 AI tests on our laptops would take weeks. So, we used a **Kaggle T4 GPU** (a powerful free supercomputer online) for the heavy lifting.
- **The Code Libraries:** We used **PyTorch** and **HuggingFace** to load the AI models, **OpenCV** to edit and ruin the images mathematically, and **Pandas** to save all our data into spreadsheets (CSVs).

---

## ✅ 3. Phase 1: Data Generation (The "Breaking Things" Phase)
**Goal:** To run 1,500 images through 3 AI models, while slowly ruining the images across 20 stages (from 0% bad to 100% bad) using 5 different types of image corruption (Blur, Dark Lighting, Blocking the view, Shrinking the image, and Changing the camera angle).

**Where we ran this:** On the Kaggle T4 GPU because it required massive computing power.

### 📁 Python Scripts Created in Phase 1 & Why They Exist:
- `download_dataset.py`: We needed a fast way to get the 1,500 images to our computer or the Kaggle GPU. This script automatically downloads the COCO and ImageNet images.
- `main.py`: This is the "Boss" script. It tells the models to look at the images, ruin them step-by-step, and save their guesses.
- `test_script.py` & `run_combined_test.py`: These are "smoke tests." Before running 150,000 tests on Kaggle, we ran these on our laptops to make sure our code wasn't crashing.
- `demo_visual.py`: This script generates visual examples (pictures) of what the 20 stages of degradation look like. It’s for humans to look at and include in our paper.
- `AFS_VFM_Benchmark.ipynb`: This is a "Notebook" file. It is literally the file we uploaded to Kaggle to run `main.py` on their supercomputer. 
- `src/degradation/degradation.py` & `transformations.py`: These are the mathematical engines. They are responsible for accurately and perfectly ruining the images without any randomness.
- `src/models/model_loader.py`: The script that actually connects to the internet to download and activate the DINOv2, CLIP, and DETR brains.

### 📊 Data (CSVs) Created in Phase 1:
- `results/full_benchmark.csv`: The holy grail of Phase 1. It is a massive spreadsheet with **150,000 rows**. It recorded every single guess the AI models made for all 1,500 images across all 20 stages of corruption.
- `results/batch_1.csv`, `batch_2.csv`, `batch_3.csv`: Since `full_benchmark` was so massive, we saved it in smaller chunks (batches) just in case Kaggle crashed halfway through.
- `results/pilot_benchmark.csv`: A tiny test spreadsheet we generated first just to make sure the data looked right.

---

## ✅ 4. Phase 2: Statistical Analysis (The "Brain Scanning" Phase)
**Goal:** Take the massive `full_benchmark.csv` and use math to find out *when* the models failed, and more importantly, look *inside* their brains to see if they knew they were failing.

We broke this phase into 4 mini-parts:

### 🔬 Part 2A: Failure Boundaries (When exactly did they fail?)
We wanted to find the exact frame (out of 20) where the AI's answer changed from Right to Wrong.
- **Script Used:** `src/analysis/failure_boundaries.py` (Calculated the exact breaking point).
- **CSV Results Generated:**
  - `results/phase2a/failure_boundaries.csv`: Listed the exact frame (1 to 20) each model failed on for each image.
  - `results/phase2a/failure_statistics.csv`: Showed us the "survival rates." **Result:** DINOv2 was the strongest (survived the longest), DETR was the weakest.
  - `results/phase2a/transition_matrices/*.csv`: Tracked what the AI "hallucinated." When it failed, what did it wrongly guess instead?

### 🧠 Part 2B: Internal Signal Probing (Looking inside the AI's Brain)
We needed to extract the internal math (confidence scores, embeddings) of the AI as the image got worse. We only did this on 300 images because it's computationally expensive.
- **Where we ran this:** Kaggle T4 GPU.
- **Scripts & Files Used:** 
  - `src/analysis/select_subset.py`: Hand-picked a perfectly balanced set of 300 images.
  - `src/models/signal_extractor.py`: The medical "scanner" that plugged into the AI to read its internal confidence numbers.
  - `AFS_VFM_Phase2B_SignalProbing.ipynb`: The notebook we ran on Kaggle to do the heavy lifting.
- **CSV Results Generated:**
  - `results/phase2b/selected_images.csv`: A list of the 300 images we picked.
  - `results/phase2b/signals/frame_signals.csv`: A giant spreadsheet tracking the internal brain signals of the AI for every single frame.

### 🔮 Part 2C: Anticipatory Signals (Can we predict failure?)
**THIS IS OUR CORE DISCOVERY.** We aligned the internal brain signals with the exact moment of failure to see if the brain signals dropped *before* the failure happened.
- **Script Used:** `src/analysis/anticipatory_signals.py`
- **CSV Results Generated:**
  - `results/phase2c/failure_aligned_signals.csv`: Adjusted the data so "Frame 0" is the exact moment of failure. 
  - `results/phase2c/predictability_metrics.csv`: Scored how well we can predict a failure. **Result:** We found that CLIP's "logit margin" (a specific brain signal) is the best predictor of failure in the entire project!
  - `results/phase2c/prefailure_window_stats.csv`: **Huge Result:** Proved mathematically that the AI's brain signals drop 3 to 5 frames *before* it actually gives the wrong answer.

### 🥊 Part 2D: Cross-Model Comparison (Model vs Model)
We compared how the 3 models failed differently.
- **Script Used:** `src/analysis/cross_model_comparison.py`
- **CSV Results Generated:**
  - `results/phase2d/robustness_rankings.csv`: Official ranking (1. DINOv2, 2. CLIP, 3. DETR).
  - `results/phase2d/failure_correlation_matrix.csv`: Showed that CLIP and DETR fail on the exact same images, but DINOv2 fails on different images.
  - `results/phase2d/ensemble_complementarity.csv`: Proved that if we combine all 3 models together (an ensemble), the overall system becomes incredibly strong.

---

## 🚀 5. Phase 3: The Final Stretch (What we are doing RIGHT NOW)
**Goal:** Turn all those CSV files into beautiful graphs, build a working prototype that actually warns people of AI failures in real-time, and write our final Research Paper.

**Where we run this:** Right here on our laptops (No Kaggle needed anymore!).

### 📁 The Files We Are About to Create in Phase 3:
1. **The Visualization Maker:**
   - **File we will build:** `src/visualization/generate_figures.py`
   - **Why it's needed:** We can't put CSV spreadsheets into a research paper. This script will read our Phase 2 CSVs and draw 9 beautiful, publication-quality graphs (Heatmaps, Line charts, Radar charts) and save them into a `results/figures/` folder.

2. **The "Early Warning System" Prototype:**
   - **Files we will build:** `src/earlywarning/learn_thresholds.py` and `failure_predictor.py`.
   - **Why it's needed:** We want to show that our research is practically useful. This code will act like a live monitor. It will watch an AI process an image, read its brain signals, and pop up a warning on the screen: *"WARNING: The AI's brain signals are dropping. Failure predicted in 3 frames!"* just before the AI gets it wrong.

3. **The Final Research Paper:**
   - **File we will build:** `docs/paper/AFS_VFM_Paper.pdf`
   - **Why it's needed:** The culmination of our work. A professionally written paper (Abstract, Introduction, Methodology, Results, Conclusion) that explains everything we did, featuring the 9 graphs we made. This is what we will publish and show to the world.

### What else?
We also have to clean up our GitHub folder (removing messy test files) so that when professors or recruiters look at our code, it looks flawless and easy to read.

*(End of Guide!)*
