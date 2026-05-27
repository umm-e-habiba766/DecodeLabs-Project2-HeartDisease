# 🫀 Project 2 — Heart Disease Classification Using AI


## 📌 Project Overview

This project implements **Data Classification Using AI** — the predictive milestone of the DecodeLabs AI Engineering track. Using the **real Cleveland UCI Heart Disease dataset** (303 actual patient records), a **Random Forest Classifier** enhanced with a **Clinical Rules Engine** is trained to predict the presence of heart disease based on 13 clinical measurements.

> *"We do not write the rules. We provide history, and the machine derives the logic."*

---

## 🎯 Goal

Build a complete supervised learning pipeline that:
- Loads and cleans a real-world medical dataset
- Preprocesses and scales all 13 clinical features
- Trains a **Random Forest** classifier with balanced class weights
- Evaluates the model using Accuracy, F1, Precision, Recall, and ROC-AUC
- Applies a **Clinical Rules Engine** to correct ML bias for edge cases
- Provides an interactive **AI Clinical Assistant chatbot**

---

## 📂 Repository Structure

```
DecodeLabs-Project2-HeartDisease-KNN/
│
├── heart_disease_classification.py   ← Main script (full pipeline + chatbot)
├── requirements.txt                  ← Required Python packages
└── README.md                         ← Project documentation (this file)
```

---

## 🗃️ Dataset Information

| Property       | Value                                        |
|----------------|----------------------------------------------|
| Name           | Cleveland Heart Disease Dataset              |
| Source         | UCI Machine Learning Repository              |
| Samples        | 303 real patient records                     |
| Features       | 13 clinical measurements                     |
| Target Classes | 0 = No Disease  /  1 = Disease               |
| Missing Values | Handled via dropna() before training         |

### Feature Description

| # | Feature    | Description                                          | Type  |
|---|------------|------------------------------------------------------|-------|
| 1 | age        | Patient age in years                                 | Float |
| 2 | sex        | Biological sex (1=Male, 0=Female)                    | Int   |
| 3 | cp         | Chest pain type (0=Typical, 1=Atypical, 2=Non-anginal, 3=Asymptomatic) | Int |
| 4 | trestbps   | Resting blood pressure (mm/Hg)                       | Float |
| 5 | chol       | Serum cholesterol (mg/dl)                            | Float |
| 6 | fbs        | Fasting blood sugar > 120 mg/dl (1=Yes, 0=No)       | Int   |
| 7 | restecg    | Resting ECG results (0=Normal, 1=ST-T abnormality, 2=LVH) | Int |
| 8 | thalach    | Maximum heart rate achieved                          | Float |
| 9 | exang      | Exercise induced angina (1=Yes, 0=No)                | Int   |
|10 | oldpeak    | ST depression induced by exercise (Oldpeak)          | Float |
|11 | slope      | Slope of peak exercise ST segment (0=Up, 1=Flat, 2=Down) | Int |
|12 | ca         | Major vessels colored by fluoroscopy (0-4)           | Int   |
|13 | thal       | Thalassemia — Cleveland encoding (3=Normal, 6=Fixed Defect, 7=Reversible Defect) | Int |

> ⚠️ **Important:** This project uses the **original Cleveland UCI thalassemia encoding** (3, 6, 7) — not the simplified (0,1,2,3) encoding used in some Kaggle versions.

---

## 🔁 Pipeline — IPO Framework

```
INPUT                    PROCESS                        OUTPUT
──────────               ─────────────────────          ──────────────────────
Cleveland UCI    →       StandardScaler          →      Accuracy / F1 Score
heart.csv                Train-Test Split                Precision / Recall
13 Features              (80% / 20% stratified)         ROC-AUC
Feature Scaling          Random Forest (100 trees)      Confusion Matrix
                         Clinical Rules Engine           AI Clinical Chatbot
                         Clinical Validation
```

---

## ⚙️ Key Concepts Covered

| Concept                  | Description                                               |
|--------------------------|-----------------------------------------------------------|
| Feature Scaling          | StandardScaler — mean=0, variance=1                       |
| Train-Test Split         | 80/20 stratified shuffle split                            |
| Random Forest            | Ensemble of 100 decision trees, balanced class weights    |
| Confusion Matrix         | TP / TN / FP / FN breakdown                               |
| F1 Score                 | Harmonic mean of Precision and Recall                     |
| ROC-AUC                  | Model discrimination ability                              |
| Clinical Rules Engine    | 6 hand-crafted override rules for edge cases              |
| Clinical Validation      | Input plausibility checker before prediction              |
| AI Clinical Chatbot      | Real-time patient risk prediction with clinical reasoning  |

---

## 🌲 Random Forest — Model Configuration

```python
RandomForestClassifier(
    n_estimators    = 100,       # 100 decision trees
    max_depth       = 10,        # Prevents overfitting
    min_samples_split = 5,       # Minimum samples to split a node
    min_samples_leaf  = 2,       # Minimum samples at leaf node
    class_weight    = 'balanced',# Handles class imbalance automatically
    random_state    = 42         # Reproducibility
)
```

---

## 🧠 Clinical Rules Engine

A key feature of this project is the **6-rule Clinical Override System** that corrects ML bias for edge cases the dataset cannot represent well:

| Rule | Condition | Action |
|------|-----------|--------|
| 1 | Age < 35, oldpeak=0, thalach > 180, ca=0, thal=3 | Override → No Disease (Young athlete) |
| 2 | Age < 40, oldpeak < 0.5, ca=0, thal=3, normal BP & chol | Override → No Disease (Very low risk) |
| 3 | Age > 50, ca >= 2, abnormal thal, oldpeak > 1.5 | Override → Disease (Severe indicators) |
| 4 | Age < 40, atypical chest pain, no exercise angina | Override → No Disease (Non-cardiac) |
| 5 | oldpeak=0, thalach > 170, no angina, age < 50 | Override → No Disease (Normal stress test) |
| 6 | Age < 40, trestbps > 160, ML confidence < 75% | Reduce confidence (Hypertension, not CAD) |

> These rules address a known **dataset bias** — the Cleveland dataset contains very few healthy patients under 35, causing the ML model to overpredict disease in young, fit individuals.

---

## 📊 Model Results

| Metric      | Score        |
|-------------|--------------|
| Accuracy    | ~85-88 %     |
| F1 Score    | ~0.87        |
| Precision   | ~0.87        |
| Recall      | ~0.88        |
| ROC-AUC     | ~0.93        |
| Trees       | 100          |
| Max Depth   | 10           |

---

## 🚀 How to Run

### Prerequisites
```
Python 3.10+
pip
```

### Installation
```bash
# Clone the repository
git clone https://github.com/umm-e-habiba766/DecodeLabs-Project2-HeartDisease-KNN.git

# Navigate into the folder
cd DecodeLabs-Project2-HeartDisease-KNN

# Install dependencies
pip install -r requirements.txt
```

### Run the Project
```bash
python heart_disease_classification.py
```

The script will:
1. Load heart.csv (or fetch from GitHub mirror if file not found)
2. Clean missing values and scale all features
3. Train the Random Forest model and print evaluation metrics
4. Launch the interactive AI Clinical Assistant chatbot

---

## 📦 Requirements

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.4.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 💬 Clinical Chatbot Sample

```
══════════════════════════════════════════════════════════════════════
      AI CLINICAL ASSISTANT — RANDOM FOREST + RULES
══════════════════════════════════════════════════════════════════════
  📌 Thalassemia: 3=Normal  6=Fixed Defect  7=Reversible Defect

  Patient Assessment #1
  ──────────────────────────────────────────────────────────────────
   1. Patient Age (years): 28
   2. Biological Sex (1=Male, 0=Female): 1
   3. Chest Pain Type: 1
   ...

════════════════════════════════════════════════
       DIAGNOSTIC REPORT  —  Patient #1
════════════════════════════════════════════════
  🟢  FINDING  : NO HEART DISEASE DETECTED
       Risk Level   : LOW RISK
       Confidence   : 92.0%

  CLINICAL REASONING:
  • Very low risk profile (age < 40, no major risk factors)
  • ML model predicted: DISEASE (68.2%)
  • Clinical rules overrode ML prediction based on demographics

  🔄 CLINICAL OVERRIDE APPLIED:
     ML said        : DISEASE
     Clinical rules : NO DISEASE

  RECOMMENDATION:
  • Patient shows no immediate signs of heart disease.
  • Routine follow-up advised.
════════════════════════════════════════════════
  ⚠  DISCLAIMER: AI prediction — not a medical diagnosis.
```

---

## 📋 Project Evaluation Criteria

| Criteria                                    | Status |
|---------------------------------------------|--------|
| Real dataset loaded and cleaned              | ✅     |
| Feature scaling applied (StandardScaler)     | ✅     |
| Train-test split 80/20 stratified            | ✅     |
| Random Forest model trained                  | ✅     |
| Confusion matrix computed                    | ✅     |
| F1, Precision, Recall, ROC-AUC computed      | ✅     |
| Clinical Rules Engine (6 rules)              | ✅     |
| Clinical input validation                    | ✅     |
| AI chatbot with clinical reasoning           | ✅     |
| Clinical override detection and display      | ✅     |
| Clean, professional, commented code          | ✅     |
| README documentation                         | ✅     |

---


## ⚠️ Medical Disclaimer

This project is built **for educational and training purposes only**. The predictions made by this model should **never** be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified cardiologist or healthcare professional.

---

