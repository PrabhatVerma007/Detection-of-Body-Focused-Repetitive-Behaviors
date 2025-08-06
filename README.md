# BFRB Detection with Sensor Data

This repository contains a machine learning pipeline for detecting **Body-Focused Repetitive Behaviors (BFRBs)** using sensor data. The pipeline processes **IMU, temperature, and time-of-flight (ToF) sensor data**, along with demographic information, to classify different behaviors.

---

## 🚀 Features

- **Data Preprocessing**: Handles missing values, outliers, and merges sensor data with demographic features  
- **Feature Engineering**: Extracts statistical features from sensor sequences  
- **Model Training**: Uses XGBoost for both binary (BFRB vs No BFRB) and multiclass classification  
- **Prediction Pipeline**: Generates predictions on new sensor data  
- **Demographic Integration**: Incorporates subject demographic information to improve model accuracy

---

## 📂 Data Structure

The pipeline works with two main data sources:

### 1. Sensor Data
- IMU measurements (acceleration, rotation)
- Temperature readings
- Time-of-Flight (ToF) sensor readings
- Behavior labels (for training)

### 2. Demographic Data
- Subject information (age, sex, handedness)
- Physical measurements (height, arm lengths)

---

## 🛠️ Pipeline Components

### 🔍 Data Inspection
- Examines file structure and identifies subject-related columns

### ⚙️ Preprocessing
- Handles missing values via forward/backward fill or median imputation
- Removes outliers or invalid sensor readings (e.g., ToF = -1)
- Merges sensor data with demographic attributes

### 🧠 Feature Extraction
- Extracts **mean, standard deviation, min, max, median** from each numeric column
- Supports feature creation for time-series data

### 📊 Model Training
- **Binary Classifier**: Detects presence or absence of BFRB
- **Multiclass Classifier**: Classifies specific types of behaviors when BFRB is detected

### 🔮 Prediction
- Preprocesses test data and generates behavior predictions using trained models

---

## 🧪 Usage

### Run the full pipeline:
```python
# Complete end-to-end execution
result = complete_end_to_end_pipeline()
```
## 🧪 Step-by-Step Debugging

To run the pipeline step-by-step for debugging or inspection:

```python
pipeline, train_processed, test_processed = debug_and_run_pipeline()
```
## 📦 Requirements

- **Python** 3.7 or higher

### Required Packages

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`

---

## ✅ Results

| Task                                        | F1 Score |
|---------------------------------------------|----------|
| Binary Classification (BFRB vs No BFRB)     | **0.857** |
| Multiclass Classification (Specific Behaviors) | **0.771** |

---

## 📁 Files

- `CMI.ipynb` — Main notebook containing the full pipeline  
- `bfrb_submission.csv` — Example output file with predictions  
- `binary_model.json` — Trained binary classifier  
- `multiclass_model.json` — Trained multiclass classifier  
- `scaler.pkl` — Fitted feature scaler used for inference

---

## 🔮 Future Improvements

- Incorporate more advanced time-series feature extraction techniques  
- Experiment with other model architectures (e.g., deep learning)  
- Add hyperparameter optimization (e.g., Optuna, GridSearchCV)  
- Implement k-fold cross-validation for better model robustness

---

