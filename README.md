# 🏥 Hospital Readmission Risk Prediction

## 🔍 Problem Statement
Predict whether a diabetic patient will be readmitted to the hospital within 30 days using clinical and demographic features.

## 📊 Dataset
UCI Diabetes 130-US hospitals dataset (100k+ records, mixed categorical and numerical features).

## 🛠️ Approach
- Data cleaning & preprocessing using sklearn Pipelines and ColumnTransformer
- Baseline and ensemble models (Logistic Regression, Random Forest)
- Class imbalance handling and decision threshold tuning
- Hyperparameter tuning with RandomizedSearchCV
- Experiment tracking with MLflow
- Deployment using Streamlit with model artifacts hosted externally

## 🚀 Live Demo
👉 http://13.60.28.169:8501

## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
