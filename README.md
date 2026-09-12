# 🚧 Automated Road Damage Classification Using Deep Learning and Cloud Deployment

## 📌 Project Overview

Road damage such as potholes, cracks, and damaged manholes can affect vehicle safety, road quality, and transportation efficiency. Manual road inspection is time-consuming and difficult to scale across large areas.

This project develops an **automated road damage classification system using Deep Learning and Computer Vision**. The system analyzes road images and classifies them into one of three categories:

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
6. Apply Transfer Learning for image classification.
7. Implement **Grad-CAM** for model explainability.
8. Develop an interactive Streamlit application.
9. Demonstrate a practical AI solution for smart-city and infrastructure monitoring.

---

## 🧠 Problem Statement

Traditional road inspection depends heavily on manual surveys and physical inspections. These methods require significant time, manpower, and resources.

The proposed system uses Computer Vision and Deep Learning to automatically classify road damage from images, helping make road monitoring more efficient and scalable.

---

## 🗂️ Dataset

The project uses road-damage images belonging to three classes:

| Class   | Description                                                 |
| ------- | ----------------------------------------------------------- |
| Pothole | Damaged or depressed sections of the road surface           |
| Crack   | Visible cracks or fractures on the road surface             |
| Manhole | Road images containing manhole-related structures or damage |

### Class Mapping

```text
Class 0 → Pothole
Class 1 → Crack
Class 2 → Manhole
```

The processed images are organized into training, validation, and testing datasets.

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
* Training, validation, and testing separation
* Class imbalance handling using class weights

Data augmentation helps improve model generalization to road images with different visual conditions.

---

## 🤖 Deep Learning Models

Multiple Deep Learning architectures were trained and compared.

### 1. Baseline CNN

A Convolutional Neural Network was developed as the baseline model for road damage classification.

### 2. MobileNetV2

MobileNetV2 was evaluated as a lightweight Transfer Learning architecture suitable for efficient image classification.

### 3. ResNet50

ResNet50 was evaluated for deep feature extraction using residual learning.

### 4. EfficientNetB0

EfficientNetB0 was evaluated for balancing model complexity and classification performance.

---

## 🏆 Model Selection and Fine-Tuning

The trained models were compared using test-set performance.

The best-performing model was selected as the final model and subsequently fine-tuned to improve classification performance.

The final trained model is saved as:

```text
road_damage_final.keras
```

> **Note:** The trained model is not included in the normal GitHub repository because the file is approximately **162 MB**, which exceeds GitHub's standard individual file-size limit.

---

## 📊 Model Evaluation

The project evaluates the classification model using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Prediction Confidence
* Inference Latency

### Confusion Matrix

The confusion matrix helps evaluate how effectively the model distinguishes between:

```text
Crack
Manhole
Pothole
```

It also helps identify classification errors and class-level confusion.

---

## 🔍 Explainable AI — Grad-CAM

**Grad-CAM (Gradient-weighted Class Activation Mapping)** is implemented to provide visual explanations for model predictions.

Grad-CAM generates a heatmap highlighting image regions that contributed to the model's prediction.

### Grad-CAM Workflow

```text
Input Road Image
       ↓
Deep Learning Model
       ↓
Predicted Class
       ↓
Grad-CAM
       ↓
Important Image Regions
```

This helps make the classification system more interpretable by showing where the model is focusing when making its prediction.

---

## 🌐 Streamlit Application

The project includes an interactive **Streamlit web application**.

Users can:

1. Upload a road image.
2. Run AI-based classification.
3. View the predicted damage class.
4. View prediction confidence.
5. View severity information.
6. View recommended action.
7. Analyze class probabilities.
8. View Grad-CAM visualization.
9. View model and technical information.
10. Download the prediction report.

### Application Flow

```text
Upload Image
      ↓
Image Preprocessing
      ↓
Deep Learning Prediction
      ↓
Predicted Road Damage
      ↓
Confidence Score
      ↓
Severity / Recommendation
      ↓
Probability Analysis
      ↓
Grad-CAM Explanation
```

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Deep Learning

* TensorFlow
* Keras
* Convolutional Neural Networks
* Transfer Learning
* MobileNetV2
* ResNet50
* EfficientNetB0

### Computer Vision

* OpenCV
* PIL / Pillow

### Data Processing

* NumPy
* Pandas

### Machine Learning

* Scikit-learn

### Visualization

* Matplotlib

### Web Application

* Streamlit

### Explainable AI

* Grad-CAM

### Development Tools

* Jupyter Notebook
* VS Code
* Anaconda
* Git
* GitHub

---

## 🐍 Python Environment

The project was developed and tested using a dedicated **Anaconda TensorFlow environment**.

### Python

```text
Python 3.x
```

> The exact Python version should match the environment used for training and testing the project. For cloud deployment, the Python version will be pinned once the deployment environment is finalized.

### Environment

```text
tensorflow_env
```

---

## 📁 Project Structure

```text
Automated-Road-Damage-Classification/
│
├── app.py
├── road_damage.ipynb
├── requirements.txt
├── README.md
├── .gitignore
│
├── labels-YOLO/
│
└── road_damage_final.keras
```

> `road_damage_final.keras`, image datasets, and processed datasets are excluded from the Git repository where appropriate because of file-size and storage considerations.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/deva392000/Automated-Road-Damage-Classification.git
```

### 2. Navigate to the Project Folder

```bash
cd Automated-Road-Damage-Classification
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Environment

#### Windows

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Streamlit Application

After installing the dependencies and placing the trained model in the project directory, run:

```bash
streamlit run app.py
```

The Streamlit application will open in the browser.

---

## 🖥️ Application Features

### 🖼️ Image Upload

Users can upload a road image through the web interface.

### 🧠 AI Prediction

The trained Deep Learning model predicts the road damage category.

### 📈 Confidence Score

The application displays the model's confidence for the predicted class.

### ⚠️ Severity Information

The application provides an estimated severity level based on the predicted damage.

### 💡 Recommendation

The system provides an action-oriented recommendation related to the detected road damage.

### 📊 Probability Analysis

The probability distribution across the three classes is displayed to understand the model's prediction.

### 🔥 Grad-CAM

Grad-CAM provides a visual explanation of the regions influencing the model's prediction.

### 📥 Prediction Report

The application supports downloading prediction results as a report.

---

## 🏙️ Real-World Applications

This system can support:

* Smart-city infrastructure monitoring
* Municipal road inspection
* Road maintenance planning
* Transportation safety monitoring
* Automated road surveys
* Infrastructure analytics
* Public road-damage reporting systems
* AI-based civil infrastructure monitoring

---

## 🚀 Future Enhancements

Possible future improvements include:

* Deploying the application to a cloud platform.
* Using a larger and more diverse road-damage dataset.
* Adding additional road-damage categories.
* Implementing object detection for locating multiple damages in a single image.
* Adding GPS/location information.
* Integrating real-time camera or mobile-camera input.
* Developing an API for integration with municipal systems.
* Adding automated road-maintenance priority scoring.
* Improving model performance using advanced architectures.
* Adding continuous model monitoring after deployment.

---

## ☁️ Cloud Deployment

The project is designed with cloud deployment in mind.

The Streamlit application can be deployed to a suitable cloud hosting platform after providing:

* Application source code
* Python dependencies
* Trained model
* Required configuration

Because the trained model is approximately **162 MB**, the model file is intentionally excluded from the normal GitHub repository and requires an appropriate external model-storage strategy for cloud deployment.

---

## 📌 Project Highlights

| Area               | Implementation                                    |
| ------------------ | ------------------------------------------------- |
| Computer Vision    | Automated road-image analysis                     |
| Deep Learning      | CNN-based image classification                    |
| Transfer Learning  | MobileNetV2, ResNet50, EfficientNetB0             |
| Model Optimization | Best-model selection and fine-tuning              |
| Class Imbalance    | Class-weight handling                             |
| Evaluation         | Accuracy, Precision, Recall, F1, Confusion Matrix |
| Explainable AI     | Grad-CAM                                          |
| Web Application    | Streamlit                                         |
| Version Control    | Git & GitHub                                      |

---

## 🎓 Skills Demonstrated

* Python Programming
* Computer Vision
* Image Preprocessing
* Data Augmentation
* Convolutional Neural Networks
* Transfer Learning
* TensorFlow / Keras
* Model Training
* Class Imbalance Handling
* Model Evaluation
* Confusion Matrix Analysis
* Precision / Recall / F1-Score
* Explainable AI
* Grad-CAM
* Streamlit Development
* Git & GitHub
* AI Application Deployment

---

## 👨‍💻 Author

**Devendra**

GitHub: https://github.com/deva392000

---

## ⭐ Project Summary

**Automated Road Damage Classification Using Deep Learning and Cloud Deployment** demonstrates an end-to-end Computer Vision and Deep Learning solution for automated road-damage classification.

The project combines **image preprocessing, CNN-based classification, Transfer Learning, model comparison, fine-tuning, performance evaluation, Grad-CAM explainability, and Streamlit application development**.

The system demonstrates how Artificial Intelligence can assist in making road inspection more **automated, scalable, interpretable, and efficient**.
