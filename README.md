# 💳 Credit Card Fraud Detection System

A production-ready fraud detection system trained on **284,807 real European credit card transactions**, achieving **AUC 0.9900** with full SHAP explainability and a live animated dashboard.

[![GitHub](https://img.shields.io/badge/GitHub-anshunigam--trigger-black?logo=github)](https://github.com/anshunigam-trigger/credit-card-fraud-detection)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-blueviolet)](https://shap.readthedocs.io)

---

## 🎯 Business Problem

Credit card fraud costs the global economy **$32 billion annually.** Traditional rule-based systems miss sophisticated fraud patterns. This project builds a machine learning system that:

- Detects fraud in real time with **AUC 0.9900**
- Explains **why** each transaction was flagged (SHAP)
- Makes confidence-based decisions: Block / Verify / Approve
- Serves predictions via a **FastAPI backend** with animated dashboard

---

## 🚀 Live Dashboard

A production-style animated web dashboard built with FastAPI + vanilla JS:

```
🔴 BLOCKED     → > 90% fraud probability
🟡 VERIFY      → 50–90% fraud probability  
🟢 APPROVED    → < 50% fraud probability
```

**Features:**
- Animated particle canvas background
- 3D tilt effect on scenario cards
- Circular gauge with glowing fraud score
- Real-time SHAP explainability bars
- CSV upload for custom transactions
- Session history tracking

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|-----|
| Logistic Regression | 97.46% | 0.0594 | 92.86% | 0.1117 | 0.9786 |
| **Random Forest** | **99.96%** | **89.25%** | **84.69%** | **86.91%** | **0.9900** |
| XGBoost | 99.92% | 74.11% | 84.69% | 79.05% | 0.9812 |
| XGBoost (tuned) | 99.73% | 38.16% | 88.78% | 53.37% | — |

**Best model → Random Forest (AUC 0.9900)**

---

## 🔍 Key Findings

**What drives fraud detection?**

| Rank | Feature | Importance | Insight |
|------|---------|------------|---------|
| 1 | V14 | 0.21 | Strongest fraud signal — extreme negative values |
| 2 | V12 | 0.19 | Second strongest indicator |
| 3 | V4 | 0.15 | Third strongest |
| 4 | V10 | 0.11 | Fourth strongest |
| 5 | V17 | 0.09 | Fifth strongest |

Amount and Time → near zero importance. Fraud doesn't depend on how much or when — it depends on hidden behavioral patterns captured in V1-V28.

---

## 🧠 SHAP Explainability

Every prediction comes with a SHAP explanation showing **why** the model flagged it:

```
Transaction #43428 — 94% fraud confidence

Starting point:    50.0%
V14 = -9.374  →   +14% (strongest signal)
V17 = -19.236 →   +7%
V10 = -14.110 →   +6%
V12 = -10.834 →   +6%
Other features →   -10%
─────────────────────────
Final score:       94% fraud
```

This satisfies regulatory requirements (GDPR, RBI) for explainable AI in financial systems.

---

## 🏗️ System Architecture

```
creditcard.csv (284,807 transactions)
        ↓
Data Preprocessing
  → StandardScaler on Amount & Time
  → Stratified train-test split (80/20)
        ↓
SMOTE Balancing (training only)
  → Before: 99.83% legit / 0.17% fraud
  → After:  50% / 50% balanced
        ↓
Model Training
  → Logistic Regression (baseline)
  → Random Forest (best model)
  → XGBoost (tuned with scale_pos_weight)
        ↓
Evaluation
  → ROC-AUC curve
  → Precision-Recall curve
  → Confusion matrix
  → SHAP explainability
        ↓
Deployment
  → fraud_model.pkl (saved model)
  → FastAPI backend (/predict endpoint)
  → Animated HTML dashboard
```

---

## 🛠️ Tech Stack

**ML & Data:**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-blueviolet?style=flat-square)

**Backend & Frontend:**

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Uvicorn](https://img.shields.io/badge/Uvicorn-purple?style=flat-square)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat-square&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat-square&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black)

---

## 📁 Project Structure

```
credit-card-fraud-detection/
├── credit_card_fraud.ipynb   # Full ML pipeline notebook
├── app.py                    # FastAPI backend
├── index.html                # Animated dashboard
├── fraud_model.pkl           # Saved Random Forest model
├── test_fraud.csv            # Sample fraud transaction
├── test_legit.csv            # Sample legit transaction
├── requirements.txt          # Dependencies
└── README.md
```

---

## ⚙️ Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/anshunigam-trigger/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download dataset**

Download from [Kaggle ULB Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` in the project folder.

**4. Run the notebook**
```bash
jupyter notebook credit_card_fraud.ipynb
```

**5. Start the dashboard**
```bash
python app.py
```

**6. Open browser**
```
http://localhost:8000
```

---

## 🧪 Test the System

Two sample CSV files are included:

| File | Expected Result |
|------|----------------|
| `test_fraud.csv` | 🔴 94% fraud — BLOCKED |
| `test_legit.csv` | 🟢 0% fraud — APPROVED |

Upload via the dashboard → Upload CSV tab.

---

## 💡 Technical Highlights

- **Extreme class imbalance** → 99.83% legitimate vs 0.17% fraud — handled with SMOTE + stratified split
- **SHAP explainability** → every prediction explained for regulatory compliance
- **Threshold tuning** → tested 0.2 to 0.5 thresholds, documented Precision-Recall tradeoffs
- **Synthetic dataset comparison** → identified and documented why synthetic fraud data fails (AUC 0.50) vs real data (AUC 0.99)
- **Production architecture** → FastAPI REST API serving saved model with real-time SHAP computation

---

## 📜 Requirements

```
fastapi
uvicorn
scikit-learn
xgboost
shap
pandas
numpy
matplotlib
seaborn
imbalanced-learn
joblib
pydantic
```

---

## 👨‍💻 Author

**Anshu Nigam**
- B.Tech CSE (IoT) — IEM Kolkata, Batch 2024–2028
- GitHub: [@anshunigam-trigger](https://github.com/anshunigam-trigger)
- Hugging Face: [@anshunigam](https://huggingface.co/anshunigam)

---

## 📄 License

MIT License — free to use and modify.
