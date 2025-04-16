
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

---

## 🔄 Project Workflow

1. ** Data Ingestion** → Load and preprocess sampled Kaggle dataset  
2. ** Exploratory Data Analysis** → Distribution, missing value patterns, feature grouping  
3. ** Feature Engineering** → Rolling averages, mins, maxes, ranges  
4. ** One-Hot Encoding** → 45 encoded categorical variables  
5. ** Modeling**
    - XGBoost (with Grid Search)
    - Neural Network (Keras)
6. ** Evaluation**
    - ROC-AUC Score
    - SHAP for interpretability
7. ** Business Strategy**
    - Conservative threshold: `0.3`
    - Aggressive threshold: `0.5`

---

## 🧠 Modeling Workflow (XGBoost & Neural Network)

The modeling pipeline follows a structured process with robust feature engineering, model tuning, and explainability:

- ** Data Sampling**: 70% random sample of the cleaned data  
- ** Null Handling**: Imputed numerical columns with median, categorical with mode  
- ** Encoding**: One-hot encoded 11 key categorical variables  
- ** Feature Engineering**: Aggregate and time-windowed averages for D, B, P, S, R features  
- ** Train/Test Split**: 70% Train, 15% Test1, 15% Test2  
- ** Models**:  
  - XGBoost (with parameter tuning)  
  - Neural Network (32 grid search configs in Keras)  
- ** Evaluation**: ROC-AUC used across train/test splits  
- ** SHAP Analysis**: Explained key features like `P_2`, `B_1`, `S_3`, `D_45` using SHAP  
- ** Business Strategy**: Thresholds set at 0.3 (conservative) and 0.5 (aggressive)

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
