# DeepBlue AI: Underwater Trash Detection

## Overview

DeepBlue AI is an advanced underwater trash detection and analysis system developed using Artificial Intelligence, Computer Vision, and Data Science techniques. The project is designed to automatically detect marine waste from underwater images and video streams to support environmental monitoring and ocean cleanup initiatives.

The system combines state-of-the-art deep learning architectures including YOLOv8 and RT-DETR for highly accurate object detection. It also integrates CLAHE image preprocessing to improve underwater visibility and ByteTrack object tracking for video analytics.

The project is deployed through an interactive Streamlit dashboard that allows users to upload images or videos and view real-time detection results.

---

# Features

* Underwater trash detection using AI
* Dual-model ensemble architecture (YOLOv8 + RT-DETR)
* CLAHE preprocessing for underwater image enhancement
* Weighted Boxes Fusion (WBF) for improved predictions
* ByteTrack integration for object tracking
* Real-time image and video analysis
* Interactive Streamlit dashboard
* Live analytics and confidence metrics
* Supports multiple marine waste categories

---

# Technologies Used

## Programming Language

* Python 3.10+

## Libraries & Frameworks

* Ultralytics YOLOv8
* RT-DETR
* OpenCV
* PyTorch
* Streamlit
* NumPy
* Pandas
* Matplotlib

## Development Environment

* Jupyter Notebook
* Google Colab (GPU Training)
* VS Code

---

# Dataset Information

## Dataset Source

* Roboflow Universe – Ocean Waste Dataset

## Dataset Size

* 3,628 annotated underwater images

## Classes (15)

* Mask
* Can
* Cellphone
* Electronics
* Glass Bottle
* Glove
* Metal
* Misc
* Net
* Plastic Bag
* Plastic Bottle
* General Plastic
* Fishing Rod
* Sunglasses
* Tire

---

# Project Architecture

The project follows a complete data science pipeline:

1. Data Collection
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing using CLAHE
4. Model Training (YOLOv8 + RT-DETR)
5. Ensemble Prediction using WBF
6. Object Tracking using ByteTrack
7. Streamlit Deployment

---

# Data Preprocessing

Underwater images usually suffer from:

* Low visibility
* Blue/green color dominance
* Haze and backscatter
* Low contrast

To overcome these challenges, CLAHE (Contrast Limited Adaptive Histogram Equalization) is applied before inference.

## Advantages of CLAHE

* Improves contrast
* Restores visibility
* Enhances object edges
* Improves model accuracy

---

# Machine Learning Models

## YOLOv8

YOLOv8 is used for fast and accurate object detection. It performs well on small and localized underwater objects.

### Advantages

* Real-time performance
* High detection accuracy
* Efficient bounding box prediction

---

## RT-DETR

RT-DETR is a transformer-based object detection model that provides global contextual understanding.

### Advantages

* Better contextual understanding
* Reduced false positives
* Improved complex scene analysis

---

# Ensemble Strategy

The project combines YOLOv8 and RT-DETR predictions using Weighted Boxes Fusion (WBF).

## Why WBF?

Instead of discarding overlapping predictions, WBF merges them using confidence scores.

### Benefits

* Higher accuracy
* Better localization
* Improved robustness

---

# Object Tracking with ByteTrack

ByteTrack is used for temporal analysis in videos.

## Purpose

* Assign unique IDs to detected objects
* Track movement across frames
* Avoid duplicate counting

## Benefits

* Accurate video analytics
* Real-time tracking
* Reliable object counting

---

# Streamlit Dashboard

The project includes an interactive web application built using Streamlit.

## Dashboard Features

* Image upload
* Video upload
* Real-time detection
* CLAHE toggle option
* Live metrics display
* Dataset visualization

---

# Installation Guide

## Step 1: Clone Repository

```bash
git clone https://github.com/Maaz89/Under_Water_Trash_Detection.git
cd Under_Water_Trash_Detection
```

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

If requirements.txt is not available:

```bash
pip install ultralytics streamlit opencv-python torch torchvision numpy pandas matplotlib
```

---

# Running the Project

## Run Jupyter Notebook

```bash
jupyter notebook
```

## Run Streamlit Dashboard

```bash
streamlit run app.py
```

---

# Model Training

## YOLOv8 Training

```bash
yolo train model=yolov8n.pt data=data.yaml epochs=150 imgsz=640
```

## RT-DETR Training

```bash
yolo train model=rtdetr-l.pt data=data.yaml epochs=150 imgsz=640
```

---

# Results

The system successfully detects multiple types of underwater trash from both images and videos.

## Key Achievements

* High-accuracy marine waste detection
* Improved underwater visibility using CLAHE
* Reduced false positives through ensemble learning
* Accurate object tracking using ByteTrack
* Real-time deployment through Streamlit

---

# Future Improvements

* GPS-based pollution heatmaps
* Real-time underwater drone integration
* TensorRT optimization for edge devices
* Trash volume and weight estimation
* Larger and more diverse datasets

---

# Applications

* Ocean cleanup monitoring
* Marine pollution analysis
* Environmental research
* Underwater robotics
* Smart surveillance systems
---

# License

This project is developed for educational and research purposes.

---

# Conclusion

DeepBlue AI demonstrates how Artificial Intelligence and Data Science can be used to solve real-world environmental challenges. By combining advanced deep learning architectures, preprocessing techniques, object tracking, and interactive deployment, the project provides a robust and scalable solution for underwater trash detection and marine pollution monitoring.
