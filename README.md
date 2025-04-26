# RingDetectionToolkit

RingDetectionToolkit is a Python module for **end-to-end 2D ring analysis**: from **synthetic noisy-ring generation** through **adaptive clustering**, **circle fitting**, **best-ring extraction**, and **statistical validation**. Includes **core workflows** for both low-noise and heavy-noise scenarios, plus a series of exploratory extras.

## :triangular_ruler: Synthetic Data & Geometry 
- Generation of circular and ring-shaped synthetic datasets
- Flexible simulation of overlapping and nested ring structures
- Controlled radial scatter and positional noise

## :bookmark_tabs: Adaptive Clustering & Fitting
- Adaptive DBSCAN clustering with parameter tuning
- Least-squares & Fast algebraic circle fits per cluster 
- Core-ring selection & outlier filtering for highest-quality fits  
- Cluster merging using σ-scaled spatial/radial uncertainty

## :dart: Best-Ring Extraction & Validation
- Extracts “core” ring points within combined center+radius error bounds  
- Outlier exclusion for final fit refinement  
- Error-tolerant comparison: matching each fitted circle to its corresponding original circle  
- Computation of normalized error ratios (x, y, r) and overall detection efficiency

## :gear: Workflows
- Sequential ('main_procedure'): fast, deterministic one-pass—ideal for low-noise data  
- Adaptive ('main_procedure_adaptive'): self-tuning outer loop—recovers rings under heavy noise

- ## :bar_chart: Performance & Visualization
- Interactive visualization & reporting to plot and print utilities for points, circles, clusters, histograms, and color‐coded summaries  
- Robust evaluation & diagnostics for normalized error ratios, fitting‐pair matching, efficiency metrics, and comparability reports


## :rocket: Extras: Experimental Modules & Studies
A set of additional utilities and tries, such as:
- HyperKamiokande-specific geometry calculators
- CPU/GPU-accelerated point sampling (multiprocessing & PyCUDA)  
- Circle merging alternative functions
- A CNN-based classifier for ring count recognition
- Hyperparameter Tuning attempts for parameter optimization
- Ptolemy’s theorem tests for ring extraction. 

> [!Note]
> All modules in RingDetectionToolkit_Extra are **exploratory** and may evolve. They are **not** required for standard ring detection workflows

---
<details>
<summary>🔧 Installation</summary>

```bash
git clone https://github.com/AlessandroFiorentino/RingDetectionToolkit.git
cd RingDetectionToolkit
pip install -r requirements.txt
```

