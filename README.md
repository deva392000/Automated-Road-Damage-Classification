# 🚧 Automated Road Damage Classification Using Deep Learning and Cloud Deployment

## 📌 Project Overview

Road damage such as potholes, cracks, and damaged manholes can affect vehicle safety, road quality, and transportation efficiency. Manual road inspection is time-consuming and difficult to scale across large areas.

This project develops an **automated road damage classification system using Deep Learning and Computer Vision**. The system analyzes road images and classifies the detected road damage into one of three categories:

* 🕳️ **Pothole**
* 🛣️ **Crack**
* ⚠️ **Manhole**

The trained Deep Learning model is integrated with a **Streamlit web application**, allowing users to upload a road image and receive an AI-based prediction, confidence score, severity information, probability analysis, and **Grad-CAM explainability visualization**.

---

## 🎯 Objectives

The main objectives of this project are:

1. Build an automated road damage classification system.
2. Preprocess and augment real-world road images.
3. Train and compare multiple Deep Learning models.
4. Handle class imbalance during model training.
5. Evaluate model performance using multiple classification metrics.
6. Apply Transfer Learning for improved image classification.
7. Implement **Grad-CAM** for model explainability.
8. Develop an interactive Streamlit application.
9. Provide a practical AI solution for smart-city and infrastructure monitoring.

---

## 🧠 Problem Statement

Traditional road inspection depends heavily on manual surveys and physical inspections. These methods require significant time, manpower, and resources.

The proposed system uses Computer Vision and Deep Learning to automatically classify road damage from images, helping make road monitoring more efficient and scalable.

---

## 🗂️ Dataset

The project uses road-damage images belonging to three classes:

| Class   | Description                                                 |
| ------- | ----------------------------------------------------------- |
| Pothole | Damaged/depressed sections of the road surface              |
| Crack   | Visible cracks or fractures on the road surface             |
| Manhole | Road images containing manhole-related damage or structures |

### Class Mapping

```text
Class 0 → Pothole
Class 1 → Crack
Class 2 → Manhole
```

The classification pipeline organizes the processed images into training, validation, and testing datasets.

---

## 🔄 Project Workflow

```text
Road Damage Images
        ↓
Data Collection
        ↓
Image Preprocessing
        ↓
Image Resizing
        ↓
Data Augmentation
        ↓
Train / Validation / Test Split
        ↓
Class Imbalance Handling
        ↓
CNN + Transfer Learning Models
        ↓
Model Comparison
        ↓
Best Model Selection
        ↓
Fine-Tuning
        ↓
Model Evaluation
        ↓
Grad-CAM Explainability
        ↓
Streamlit Web Application
        ↓
Road Damage Prediction
```

---

## 🧹 Image Preprocessing

The images are processed before being supplied to the Deep Learning models.

Major preprocessing operations include:

* Image resizing to **224 × 224 pixels**
* Pixel normalization
* Training data augmentation
* Training/validation/testing separation
* Class balancing using
