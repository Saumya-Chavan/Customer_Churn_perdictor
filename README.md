# Telecom Customer Churn Predictor

This repository contains an end-to-end Machine Learning mini-project to predict customer churn using the popular Telco Customer Churn dataset from Kaggle, complete with a beautiful, modern Streamlit frontend.

## Project Workflow

This project is executed in five main phases:
1. **Data Collection**: Fetching the dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`).
2. **Exploratory Data Analysis (EDA)**: Understanding feature distributions, finding relationships, and identifying necessary pre-processing steps.
3. **Data Pre-processing**: Handling missing values, performing label encoding for categorical features, and addressing class imbalance using SMOTE.
4. **Model Training**: Training and comparing Decision Tree, Random Forest, and XGBoost classifiers using cross-validation.
5. **Interactive UI Deployment**: Deploying the model using a modern, glassmorphism-styled Streamlit web application.

## Tech Stack
* **Frontend:** Streamlit, Custom CSS (Glassmorphism design)
* **Backend Language:** Python 3
* **Libraries:** Pandas, NumPy, Scikit-Learn, Imbalanced-Learn (SMOTE), XGBoost, Matplotlib, Seaborn

## Project Branches
* **`churn_modeling`:** Contains the raw Machine Learning backend files (`churn_model.ipynb`, `.pkl` model files).
* **`streamlit_branch`:** Contains the frontend UI and web application codes.

## Project Structure (Streamlit UI Branch)
Customer-Churn-Project/
│
├── app.py                 ← Main Streamlit web application
├── background.jpg         ← UI background image
├── .streamlit/            
│   └── config.toml        ← Streamlit dark-mode theme configuration
├── customer_churn_model.pkl ← Saved XGBoost predictive model
├── encoders.pkl           ← Saved label encoders for feature mapping
├── README.md              ← Project documentation

## How to Run

1. Ensure the model files (`.pkl`) and images are present in the project folder.
2. Run the application locally using Streamlit:
   ```bash
   streamlit run app.py
   ```
