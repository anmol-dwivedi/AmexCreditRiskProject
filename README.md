
# 🏦 Credit Risk Prediction using AmEx Dataset

Predicting credit card default risk using advanced machine learning models on a real-world Kaggle dataset. This project simulates real business strategies through threshold-based decision-making and model interpretability.

[![Kaggle](https://img.shields.io/badge/Data-Kaggle-blue)](https://www.kaggle.com/competitions/amex-default-prediction)
[![Model](https://img.shields.io/badge/Model-XGBoost%20%7C%20NeuralNet-green)](#modeling)
[![License](https://img.shields.io/badge/License-MIT-brightgreen)]()

---

## 📦 Dataset Overview

- **Source**: [AmEx Default Prediction](https://www.kaggle.com/competitions/amex-default-prediction)
- **Original Size**: 190+ features, ~5.5 million rows
- **Subset Used**: 1.1 million rows, 91149 unique customers
- **Target**: Binary classification - `default payment in next 90 days`

### Feature Categories:
- `D_*`: Delinquency
- `S_*`: Spend
- `P_*`: Payments
- `B_*`: Balances
- `R_*`: Risk-related metrics

---


## 📊 Exploratory Data Analysis (EDA) with PySpark

The `Exploration.ipynb` notebook leverages **PySpark** to handle and analyze the large-scale AmEx credit risk dataset efficiently.

### Key EDA Steps:

- **🔍 Schema Inspection**: Loaded data into PySpark DataFrame and examined structure using `printSchema()`  
- **📈 Descriptive Stats**: Used `describe()` to view mean, stddev, min, max for key features  
- **🧱 Missing Values**: Identified nulls across variables with `isNull()` and aggregation queries  
- **📊 Categorical Variable Distributions**: Analyzed class imbalances using `groupBy().count()`  
- **🕒 Time-Based Analysis**: Explored temporal patterns in customer behavior using time groupings

This scalable approach ensured rapid profiling of over a million customer records.


## 🔄 Project Workflow

1. **📥 Data Ingestion** → Load and preprocess sampled Kaggle dataset
2. **📊 Exploratory Data Analysis** → Distribution, missing value patterns, feature grouping
3. **🛠️ Feature Engineering** → Rolling averages, mins, maxes, ranges
4. **🏗️ One-Hot Encoding** → 45 encoded categorical variables
5. **🧠 Modeling**
    - XGBoost (with Grid Search)
    - Neural Network (with Keras + Grid Search)
6. **📈 Evaluation**
    - ROC-AUC Score
    - SHAP for model explainability
7. **💡 Business Strategy**
    - Conservative threshold: `0.3`
    - Aggressive threshold: `0.5`

---

## 📊 Exploratory Data Analysis

- Visualized missingness and feature patterns across customer timelines
- Sampled customers' profiles across time for sanity checks
- Found significant nulls in payment and balance features, imputed via median

---

## 🧠 Modeling

### XGBoost

- Tuned: `n_estimators`, `learning_rate`, `subsample`, `colsample_bytree`, `scale_pos_weight`
- Grid Search across 72+ configs × 3-fold CV
- SHAP analysis for interpretation:
  - `P_2`, `B_1`, `S_3`, and `D_45` heavily influenced predictions
- Best model selected based on:
  - Highest ROC-AUC on Test Set
  - Smallest gap between Train and Test ROC-AUC

### Neural Network

- Tuned: `#hidden layers`, `#nodes/layer`, `activation`, `dropout`, `batch size`
- 32 model variations tested using TensorFlow/Keras
- Final model outperformed XGBoost in AUC

---

## 🎯 Business Strategy

| Strategy      | Threshold | Objective                          |
|---------------|-----------|------------------------------------|
| Conservative  | 0.30      | Lower default rate, safer lending |
| Aggressive    | 0.50      | Higher revenue, more risk         |

---

## 🗂️ Repository Structure

```
AmexCreditRiskProject/
│
├── 1_Final_AML_Project_Code.ipynb      # Main Modeling Notebook
├── Exploration.ipynb                   # EDA Visualizations
├── DataPrepFull_Modelling.ipynb        # Data prep and pipeline
├── feature_importance_xgb_m1.xlsx      # Feature importance sheets
├── grid_search_results_*.csv           # Tuning logs
├── Credit Risk Project.pptx            # Class presentation
└── README.md                           # (This file)
```

---

## ▶️ How to Run

1. Clone the repo:
```bash
git clone https://github.com/anmol-dwivedi/AmexCreditRiskProject.git
cd AmexCreditRiskProject
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the notebooks in order:
   - `Exploration.ipynb`
   - `DataPrepFull_Modelling.ipynb`
   - `1_Final_AML_Project_Code.ipynb`

---

## 🚀 Future Work

- Incorporate AmEx competition metric for custom evaluation
- Deploy model using Streamlit/Flask for live scoring
- Experiment with LightGBM, CatBoost
- Integrate AutoML techniques (e.g., Optuna, FLAML)

---

## 🙌 Acknowledgments

Built by:
- [Anmol Dwivedi](https://github.com/anmol-dwivedi)

Inspired by the Kaggle AmEx Default Risk Competition

---
