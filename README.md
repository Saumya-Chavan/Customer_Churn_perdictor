# Telco Customer Churn Predictor

This repository contains an end-to-end Machine Learning mini-project to predict customer churn using the popular Telco Customer Churn dataset from Kaggle. 

## Project Workflow

This project is executed in five main phases:
1. **Data Collection**: Fetching the dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`).
2. **Exploratory Data Analysis (EDA)**: Understanding feature distributions, finding relationships, and identifying necessary pre-processing steps.
3. **Data Pre-processing**: Handling missing values, performing label encoding for categorical features, and addressing class imbalance using SMOTE.
4. **Model Training**: Training and comparing Decision Tree, Random Forest, and XGBoost classifiers using cross-validation.
5. **Model Evaluation**: Testing the best performing model on a held-out test set to gauge real-world performance.

## Tech Stack
* **Language:** Python 3
* **Libraries:** Pandas, NumPy, Scikit-Learn, Imbalanced-Learn (SMOTE), XGBoost, Matplotlib, Seaborn

## Project Structure
Customer-Churn-Project/
│
├── churn_model.ipynb   ← Training + EDA
├── churn_model.pkl     ← Saved model
├── app.py              ← Streamlit app
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   ← dataset
├── requirements.txt


## How to Run
1. Ensure the dataset `WA_Fn-UseC_-Telco-Customer-Churn.csv` is present in the root directory.
2. Run `python telco_churn_analysis.py` to begin executing the pipeline.
