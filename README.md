# RingDetectionToolkit :bookmark_tabs:
RingDetectionToolkit is a full-featured Python module designed to **simulate**, **detect**, and **analyze geometric ring structures in 2D data**.

This toolkit provides: a Python toolkit to generate, cluster, and classify noisy ring patterns.

## :triangular_ruler: Synthetic Data & Geometry 
- Generation of circular and ring-shaped synthetic datasets
- Flexible simulation of overlapping and nested ring structures
- Support for controlled noise and spatial randomness

## :bookmark_tabs: Adaptive Clustering & Fitting
- Custom clustering techniques optimized for ring detection
- Adaptive DBSCAN-based methods with parameter tuning
- Robust circle fitting algorithms for noisy and partial rings
- Error-tolerant comparison between syntetic data and found rings

## :bar_chart: Performance & Visualization
- Evaluation metrics and visualization tools
- Diagnostic plots for cluster quality, fit accuracy, and more
- Modular structure for use in notebooks or pipelines

## :gear: Extras: Experimental Modules & Studies
A set of additional utilities and tryes, such as:
- Fast generation and fitting algorithms
- Circle merging heuristics
- A CNN-based classifier for ring count recognition
- Fine-tuning tyes for parameter optimization
- Exploratory tools like Ptolemy's theorem validation on quadrilaterals

> [!Note]
> This repository is structured for modular experimentation and extension.
Main functionality lives in the core toolkit, while extra/ or experiments/ folders
host advanced studies and prototyping tools.

---
<details>
<summary>🔧 Installation</summary>

```bash
git clone https://github.com/AlessandroFiorentino/RingDetectionToolkit.git
cd RingDetectionToolkit
pip install -r requirements.txt
```
