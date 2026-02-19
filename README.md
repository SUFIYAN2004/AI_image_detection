# AI_image_detection
A Deep Learning project to distinguish between AI-generated images (CIFAKE) and real human photography.

# 🔍 AI vs. Human Image Detection System

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployment-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)

## 📌 Project Overview
This project is an AI-powered image classifier developed to distinguish between **human-captured photographs** and **AI-generated synthetic images**. It was created as a final-year project for my BCA degree to address the challenge of identifying synthetic visual content.



## 🛠️ Features
* **Real-time Detection:** Instantly classify any uploaded image.
* **Confidence Scoring:** Displays a percentage score indicating the model's certainty.
* **Deep Learning Backend:** Powered by a custom Convolutional Neural Network (CNN).
* **User-Friendly UI:** Built with Streamlit for a clean, responsive web experience.

## 🏗️ Technical Architecture
* **Dataset:** Trained on the **CIFAKE** dataset (100,000 images), featuring 50,000 real images and 50,000 AI-generated images.
* **Model:** Convolutional Neural Network (CNN) featuring:
  * **Batch Normalization** for stability.
  * **Dropout Layers** (0.25 - 0.5) to prevent overfitting.
  * **Data Augmentation** (Flip, Rotation, Zoom) to improve real-world accuracy.
* **Preprocessing:** Automatic 32x32 resizing and normalization.



---

## 🚀 How to Run Locally

### 1. Prerequisites
Ensure you have **Python 3.9+** and **Git LFS** installed on your system.
```bash
git lfs install
```


## 🚀 How to Run Locally

### 1. Prerequisites
Ensure you have **Python 3.9+** and **Git LFS** installed on your system.
```bash
git lfs install
2. Clone and Setup
Bash
# Clone the repository
git clone [https://github.com/SUFIYAN2004/AI_image_detection.git](https://github.com/SUFIYAN2004/AI_image_detection.git)
cd AI_image_detection

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate  # On Windows
# source venv/bin/activate # On Mac/Linux

# Install dependencies
pip install -r requirements.txt
3. Run the App
Bash
streamlit run main.py
# Performance
Validation Accuracy: Achieved a reliable accuracy of ~96%.

# Optimization: Used a low learning rate (10 ^−4) to ensure stable training and better generalization.

# Developer
V. Mohammed Sufiyan
