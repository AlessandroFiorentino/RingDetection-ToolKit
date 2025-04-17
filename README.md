# RingDetectionToolkit
RingDetectionToolkit is a full-featured Python module designed to simulate, detect, and analyze geometric ring structures in 2D data.

This toolkit provides:
A Python toolkit for generating, clustering, and classifying synthetic ring patterns using DBSCAN and Convolutional Neural Networks (CNNs). Useful for benchmarking, ML training, and geometric analysis.

##Synthetic Data & Geometry
- Efficient generation of circular and ring-shaped synthetic datasets
- Flexible simulation of overlapping and nested ring structures
- Support for controlled noise and spatial randomness

## Adaptive Clustering & Fitting
- Custom clustering techniques optimized for ring detection
- Adaptive DBSCAN-based methods with local eps tuning
- Robust circle fitting algorithms for noisy or partial rings
- Error-tolerant comparison between ground truth and predictions

## Performance & Visualization
- Detailed evaluation metrics and visualization tools
- Diagnostic plots for cluster quality, fit accuracy, and more
- Modular structure for use in notebooks or pipelines

## Extras: Experimental Modules & Studies
A set of additional utilities and tryes, such as:
- Fast generation and fitting algorithms
- Circle merging heuristics
- A CNN-based classifier for ring count recognition
- Fine-tuning tyes for parameter optimization
- Exploratory tools like Ptolemy's theorem validation on quadrilaterals

#Note:
This repository is structured for modular experimentation and extension.
Main functionality lives in the core toolkit, while extra/ or experiments/ folders
host advanced studies and prototyping tools.

#Tip
Documentation is in progress — stay tuned for a detailed walkthrough of features, examples, and API references.

## Installation

```bash
git clone https://github.com/AlessandroFiorentino/ring-detection-toolkit_draft.git
cd ring-detection-toolkit
pip install -r requirements.txt
