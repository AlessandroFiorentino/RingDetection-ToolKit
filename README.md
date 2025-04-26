# RingDetectionToolkit

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1454qnYQ4ZSxsiQox7dS8wo6We9gcrMfM?usp=sharing)

RingDetectionToolkit is a Python module for **end-to-end 2D ring analysis**, from **synthetic noisy-ring generation** through **adaptive clustering**, **circle fitting**, **best-ring extraction**, and **statistical validation**. It includes **core workflows** for both low-noise and heavy-noise scenarios, plus a series of exploratory extras.

## :triangular_ruler: Synthetic Data & Geometry 
- Generation of circular and ring-shaped synthetic datasets
- Flexible simulation of overlapping and nested ring structures
- Controlled radial scatter and positional noise.

## :bookmark_tabs: Adaptive Clustering & Fitting
- Adaptive DBSCAN clustering with parameter tuning
- Least-squares and fast algebraic circle fits per cluster 
- Core-ring selection and outlier filtering for the highest-quality fits  
- Cluster merging using σ-scaled spatial/radial uncertainty.

## :dart: Best-Ring Extraction & Validation
- Extracts ring points within combined center+radius error bounds  
- Outlier exclusion for final fit refinement  
- Error-tolerant comparison: matching each fitted circle to its corresponding original circle  
- Computation of normalized error ratios (x, y, r) and overall detection efficiency.

## :gear: Procedures
- Sequential ('main_procedure'): fast, deterministic one-pass—ideal for low-noise data  
- Adaptive ('main_procedure_adaptive'): self-tuning outer loop—recovers rings under heavy noise.

## :bar_chart: Performance & Visualization
- Interactive visualization and reporting to plot and print utilities for points, circles, clusters, histograms, and color‐coded summaries  
- Robust evaluation and diagnostics for normalized error ratios, fitting‐pair matching, efficiency metrics, and comparability reports.


## :heavy_plus_sign: Extras: Experimental Modules & Studies
A set of additional utilities and tries, such as:
- HyperKamiokande-specific geometry calculators
- CPU/GPU-accelerated point sampling (multiprocessing and PyCUDA)  
- Circle merging alternative functions
- A CNN-based classifier for ring count recognition
- Hyperparameter Tuning attempts for parameter optimization
- Ptolemy’s theorem tests for ring extraction.

---

## :notebook_with_decorative_cover: Interactive Colab Demo

A **view-only** Google Colab notebook is available here — no setup required, just click to explore all functions with live examples:

[Open the demo notebook (read-only)](https://colab.research.google.com/drive/1454qnYQ4ZSxsiQox7dS8wo6We9gcrMfM?usp=sharing)


---

> [!Note]
> All modules in RingDetectionToolkit_Extra are **exploratory** and may evolve. They are **not** required for standard ring detection procedures.

---
<summary> :wrench: Installation </summary>

```bash
git clone https://github.com/AlessandroFiorentino/RingDetectionToolkit.git
cd RingDetectionToolkit
pip install -r requirements.txt
```

<summary> :rocket: Quick Start </summary>

After installing the requirements:

```bash
cd RingDetectionToolkit
python main.py
```

- By default, this runs the **serial** workflow (`MULTIPROCESSING = False`) with **verbose output enabled** (`VERBOSE = True`).  
- If you want **faster** and **cleaner** runs (especially with many rings or multiprocessing), consider setting `VERBOSE = False` manually inside `main.py`.  
- See the very first docstring inside `main.py` for full usage recommendations.

---
