# Telco Customer Churn Prediction (ML + Streamlit Dashboard)
## 🔍 Overview

This project builds a **production-style machine learning pipeline** to predict customer churn using the Telco Customer Churn dataset.

It demonstrates a **complete ML workflow**, including:

* Data Cleaning & Feature Engineering
* Exploratory Data Analysis (EDA)
* Pipeline-based Preprocessing
* Model Training & Hyperparameter Tuning
* Model Evaluation (CV + Test Set)
* Feature Importance Analysis
* Deployment using Streamlit

![Dashboard](images/dashboard.png)
---

## 🎯 Objective

Predict whether a customer will **churn (leave the service)**.

### Business Context:

* Churn prediction helps companies **retain customers**
* Focus is on **Recall** → minimizing missed churn cases (false negatives)

---

## ⚙️ Workflow

### 1. Data Cleaning

Performed using `src/cleaning.py`:

* Converted `TotalCharges` to numeric
* Dropped irrelevant column (`customerID`)
* Encoded target variable (`Churn`)

---

### 2. Feature Engineering

Implemented inside pipeline (`src/features.py`):

```python
AvgCharges = TotalCharges / (tenure + 1)
```

This ensures:

* No data leakage
* Consistent transformation during training and inference

---

### 3. Preprocessing Pipeline

Using `ColumnTransformer`:

* **Numerical Features**

  * Imputation (median)
  * StandardScaler

* **Categorical Features**

  * Imputation (most frequent)
  * OneHotEncoder

All transformations are done inside pipeline to ensure:
* ✅ Leakage-safe training
* ✅ Reproducibility
* ✅ Deployment consistency

---

### 4. Model Training & Tuning

Models used:

* Logistic Regression
* Decision Tree
* Random Forest
* KNN
* SVM
* XGBoost (final selected model)

Hyperparameter tuning performed using:

* `GridSearchCV`
* `StratifiedKFold (5-fold)`

---

### 5. Evaluation Metrics

Primary metric:

* **Recall** → prioritize detecting churn

Additional metrics:

* F1 Score
* ROC-AUC
* Accuracy

---

### 6. Model Selection

Models compared based on:

* Cross-validation mean score
* Standard deviation (stability)
* Test set performance

---

## 📊 Feature Importance

Feature importance extracted from XGBoost model.

### Top Features:

* **Contract Type**
* **Internet Service**
* **Tech Support**
* **Online Security**
* **Monthly Charges**

Grouped importance was used to aggregate OneHotEncoded features back to original variables for better interpretability.

![Dashboard](images/feature_importance.png)

---

## 📈 Results

| Metric    | Value |
| --------- |-------|
| Recall    | 0.997 |
| ROC-AUC   | 0.837 |
| Stability | 0.013 |

*(Exact values depend on training run)*

---

## 🚀 Deployment

A Streamlit app (`app.py`) is provided for interactive predictions.

![Dashboard](images/prediction.png)

### Run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧠 Key Learnings

* Pipelines are critical to prevent **data leakage**
* Recall is more important than accuracy in churn problems
* Hyperparameter tuning significantly improves performance
* Proper project structure is essential for real-world ML systems

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Imbalanced-learn
* Matplotlib, Seaborn
* Streamlit

---

## 📬 Author

Built as part of a **Data Science portfolio project** demonstrating end-to-end ML workflow.

Feel free to explore, fork, and improve 🚀
