# 🫀 Project 2 — Heart Disease Classification Using KNN

<div align="center">

![DecodeLabs](https://img.shields.io/badge/DecodeLabs-Industrial%20Training-2A9D8F?style=for-the-badge)
![Batch](https://img.shields.io/badge/Batch-2026-1B3A5C?style=for-the-badge)
![Project](https://img.shields.io/badge/Project-2%20of%205-E8632A?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=for-the-badge&logo=scikit-learn)

**Industrial Training Kit | Batch 2026 | Powered by DecodeLabs**

</div>

---

## 📌 Project Overview

This project implements **Data Classification Using AI** — the predictive milestone of the DecodeLabs AI Engineering track. Using the **real Cleveland UCI Heart Disease dataset** (303 actual patient records), a **K-Nearest Neighbors (KNN)** classifier is trained to predict the presence of heart disease based on 13 clinical features.

> *"We do not write the rules. We provide history, and the machine derives the logic."*

---

## 🎯 Goal

Build a complete supervised learning pipeline that:
- Loads and explores a real-world medical dataset
- Preprocesses and scales features correctly
- Finds the optimal K using the elbow method
- Trains and evaluates a KNN classifier
- Generates a 6-panel professional visual report
- Provides an interactive clinical chatbot predictor

---

## 📂 Repository Structure

```
DecodeLabs-Project2-HeartDisease-KNN/
│
├── heart_disease_classification.py   ← Main script (full pipeline + chatbot)
├── heart.csv                         ← Cleveland UCI dataset (303 records)
├── requirements.txt                  ← Required Python packages
├── README.md                         ← Project documentation (this file)
└── DecodeLabs_Project2_HeartDisease_KNN.png  ← Generated visual report
```

---

## 🗃️ Dataset Information

| Property       | Value                                      |
|----------------|--------------------------------------------|
| Name           | Cleveland Heart Disease Dataset            |
| Source         | UCI Machine Learning Repository            |
| Samples        | 303 real patient records                   |
| Features       | 13 clinical measurements                   |
| Target Classes | 0 = No Disease  /  1 = Disease             |
| Balance        | 138 No Disease  /  165 Disease             |

### Feature Description

| # | Feature    | Description                                    | Type  |
|---|------------|------------------------------------------------|-------|
| 1 | age        | Patient age in years                           | Float |
| 2 | sex        | Biological sex (1=Male, 0=Female)              | Int   |
| 3 | cp         | Chest pain type (0–3)                          | Int   |
| 4 | trestbps   | Resting blood pressure (mm/Hg)                 | Float |
| 5 | chol       | Serum cholesterol (mg/dl)                      | Float |
| 6 | fbs        | Fasting blood sugar > 120 mg/dl (1=Yes, 0=No) | Int   |
| 7 | restecg    | Resting ECG results (0–2)                      | Int   |
| 8 | thalach    | Maximum heart rate achieved                    | Float |
| 9 | exang      | Exercise induced angina (1=Yes, 0=No)          | Int   |
|10 | oldpeak    | ST depression induced by exercise              | Float |
|11 | slope      | Slope of peak exercise ST segment (0–2)        | Int   |
|12 | ca         | Major vessels colored by fluoroscopy (0–4)     | Int   |
|13 | thal       | Thalassemia type (0–3)                         | Int   |

---

## 🔁 Pipeline — IPO Framework

```
INPUT                  PROCESS                    OUTPUT
──────────             ──────────────             ──────────────
Cleveland UCI    →     StandardScaler      →      Confusion Matrix
heart.csv              Train-Test Split           F1 Score
13 Features            (80% / 20%)                ROC-AUC Curve
Feature Scaling        KNN Algorithm              6-Panel Report
                       Optimal K (Elbow)          Clinical Chatbot
```

---

## ⚙️ Key Concepts Covered

| Concept              | Description                                         |
|----------------------|-----------------------------------------------------|
| Feature Scaling      | StandardScaler — mean=0, variance=1                 |
| Train-Test Split     | 80/20 stratified shuffle split                      |
| KNN Algorithm        | K-Nearest Neighbors with Minkowski distance         |
| Elbow Method         | Finding optimal K by minimizing error rate          |
| Confusion Matrix     | TP / TN / FP / FN breakdown                        |
| F1 Score             | Harmonic mean of Precision and Recall               |
| ROC-AUC              | Model discrimination ability (area under curve)     |
| Clinical Chatbot     | Real-time patient risk prediction interface         |

---

## 📊 Model Results

| Metric      | Score       |
|-------------|-------------|
| Accuracy    | **86.89 %** |
| F1 Score    | **0.8857**  |
| Precision   | **0.8710**  |
| Recall      | **0.9375**  |
| ROC-AUC     | **0.9042**  |
| Optimal K   | **14**      |

> ⚠️ Note: High Recall (93.75%) is critical in medical diagnosis — the model correctly identifies 30 out of 32 actual disease cases, missing only 2.

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
git clone https://github.com/YOUR_USERNAME/DecodeLabs-Project2-HeartDisease-KNN.git

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
1. Load the dataset from `heart.csv`
2. Train the KNN model
3. Print all evaluation metrics
4. Save the 6-panel visual report as a PNG
5. Launch the interactive clinical chatbot

---

## 📦 Requirements

```
# requirements.txt
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.4.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🖥️ Visual Report Preview

The script generates a **6-panel professional report** including:

| Panel | Content                        |
|-------|--------------------------------|
| 1     | Class Distribution Bar Chart   |
| 2     | Feature Correlation with Target|
| 3     | Elbow Curve — Optimal K        |
| 4     | Confusion Matrix (TP/TN/FP/FN) |
| 5     | ROC Curve with AUC             |
| 6     | Model Summary Score Card       |

---

## 💬 Clinical Chatbot Sample

```
══════════════════════════════════════════════════════════════
     DECODELABS AI CLINICAL ASSISTANT — KNN ENGINE
══════════════════════════════════════════════════════════════

  DISCLAIMER: Educational use only. Not a medical diagnosis.

  Patient Assessment #1
  ──────────────────────────────────────────────────────────
   1. Patient Age (years): 58
   2. Biological Sex (1=Male, 0=Female): 1
   3. Chest Pain Type: 3
   ...

═════════════════════════════════════
      DIAGNOSTIC REPORT — Patient #1
═════════════════════════════════════
  🔴  FINDING  : HEART DISEASE DETECTED
      Risk Level   : HIGH RISK
      Confidence   : 78.6%

  RECOMMENDATION:
  Please refer patient to a cardiologist immediately.
═════════════════════════════════════
  ⚠  DISCLAIMER: AI prediction — not a medical diagnosis.
```

---

## 📋 Project Evaluation Criteria

| Criteria                            | Status |
|-------------------------------------|--------|
| Real dataset loaded correctly        | ✅     |
| Feature scaling applied              | ✅     |
| Train-test split (80/20)             | ✅     |
| Optimal K found via elbow method     | ✅     |
| KNN model trained and evaluated      | ✅     |
| Confusion matrix generated           | ✅     |
| F1 Score computed                    | ✅     |
| ROC-AUC curve plotted                | ✅     |
| 6-panel visual report saved          | ✅     |
| Clinical chatbot with validation     | ✅     |
| Clean, professional, commented code  | ✅     |
| README documentation                 | ✅     |

---

## 👤 Author

| Field        | Detail                                  |
|--------------|-----------------------------------------|
| Name         | *[Your Full Name]*                      |
| Batch        | DecodeLabs 2026                         |
| Track        | Artificial Intelligence                 |
| Project      | 2 — Data Classification Using AI (KNN)  |
| Submitted    | *[Submission Date]*                     |

---

## ⚠️ Medical Disclaimer

This project is built **for educational and training purposes only**. The predictions made by this model should **never** be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified cardiologist or healthcare professional.

---

## 🏢 About DecodeLabs

**DecodeLabs** is an industrial training platform empowering the next generation of AI engineers through hands-on project-based learning.

📞 +91 89330 06408 &nbsp;|&nbsp; ✉ decodelabs.tech@gmail.com &nbsp;|&nbsp; 🌐 www.decodelabs.tech

---

<div align="center">
<sub>DecodeLabs Industrial Training Kit | Batch 2026 | Project 2 of 5</sub>
</div>
