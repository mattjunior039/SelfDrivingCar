# Self-Driving Car Vision Pipeline

This repository contains an end-to-end educational curriculum that tracks the evolution of computer vision in autonomous driving. Over three phases, you will move from classical heuristic pixel manipulation (rules and color spaces) up to modern deep learning architectures (CNNs and Real-Time Object Detection).

This project highlights the *brittleness of heuristic algorithms* and demonstrates exactly why modern self-driving cars rely almost entirely on learned features and deep neural networks.

## Phase 1: Rules, Color Spaces, & Geometric Primitives

Before using machine learning, early self-driving prototypes relied heavily on hard-coded rules. In this notebook, you will build a Stop Sign detector purely from math and logic.

Core ideas:
- Tensors, image channels, and matrix representations
- RGB vs HSV color spaces (and why Hue is robust to lighting)
- Binarization and dual-band thresholding
- Morphological cleaning (erosion and dilation)
- Contour extraction and polygon approximation (Douglas-Peucker)

You will see exactly how this heuristic approach works beautifully in perfect conditions, and how it completely falls apart when faced with sunsets, shadows, and red cars.

## Phase 2: Feature Learning with CNNs

This phase discards manual color rules and introduces Convolutional Neural Networks (CNNs). Instead of telling the computer *how* to find edges, we let it learn optimal spatial filters from data.

Core ideas:
- Sliding dot products and feature maps
- Automated feature extraction (learning kernel weights)
- `MiniRoadCNN` architecture in PyTorch (Conv2D, ReLU, MaxPool)
- Cross-Entropy loss and the optimization loop
- Evaluation using confusion matrices and tracking row-wise recall
- Understanding the **Spatial Bottleneck**: why a standard classifier loses positional information

This phase uses a dataset of 64x64 cropped patches containing cars, pedestrians, and empty roads.

## Phase 3: Real-Time Multi-Object Detection

A vehicle cannot just classify an image; it needs bounding boxes and coordinates to plan a trajectory. This phase introduces YOLO (You Only Look Once), showing how we preserve the spatial grid all the way to the output.

Core ideas:
- Single-Stage Object Detectors (YOLO)
- Bounding Box Regression
- Intersection over Union (IoU)
- Non-Maximum Suppression (NMS) to eliminate duplicate overlapping boxes
- Inference Latency (FPS) vs. Accuracy trade-offs
- Analyzing edge cases (temporal flickering, occlusion dropouts)

## Google Colab

These notebooks are designed to work perfectly in Google Colab. The first code cell in each notebook automatically checks for Colab, clones this repository, and copies the required `data/` assets (including image patches and videos) into the workspace.

This allows the notebooks to run in the cloud with real-world data without any manual uploading required!

### 📓 Interactive Colab Notebooks

Click any badge to launch the lab directly in your browser:

| Lab Phase | Direct Colab Link |
| :--- | :--- |
| **Phase 1: Heuristics & Color** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mattjunior039/SelfDrivingCar/blob/main/phase1_heuristic_vision_and_color_spaces.ipynb) |
| **Phase 2: Feature Learning CNNs** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mattjunior039/SelfDrivingCar/blob/main/phase2_feature_learning_with_cnns.ipynb) |
| **Phase 3: YOLO Object Detection** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mattjunior039/SelfDrivingCar/blob/main/phase3_realtime_object_detection_yolo.ipynb) |

## Local Setup

### Requirements
- Python 3.9+
- pip

### Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision opencv-python numpy matplotlib scikit-learn ipywidgets ultralytics
```

## Running the Notebooks

Open the notebooks in Jupyter or VS Code and run cells in order.

Typical flow:

```bash
jupyter notebook
```

Then open:
- `phase1_heuristic_vision_and_color_spaces.ipynb`
- `phase2_feature_learning_with_cnns.ipynb`
- `phase3_realtime_object_detection_yolo.ipynb`

## Summary

This repository is a compact end-to-end demonstration of how perception systems evolve from simple pixel-level rules into robust, real-time object detection systems capable of running on edge devices.
