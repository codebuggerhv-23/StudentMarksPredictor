# 🎓 Student Marks Predictor — Machine Learning Model

A machine learning project to predict student exam scores based on their past performance, study habits, and related features.  
The goal is to help **educators, students, and institutions** identify learning patterns, predict outcomes, and take proactive steps to improve student success.

This project demonstrates an end-to-end ML workflow, including preprocessing, feature engineering, model training, evaluation, and prediction — built entirely in **Python** using `Pandas`, `NumPy`, `Seaborn`, and `Scikit-learn`.

---

## 📘 1. Project Overview

The project focuses on **predicting student marks** by analyzing academic and behavioral data.  
It highlights how machine learning can be used to gain actionable insights in education — identifying at-risk students and optimizing study strategies.

---

## 🎯 2. Objectives

- Perform complete data cleaning and exploratory analysis on the **Student Habits vs Performance** dataset.  
- Apply feature engineering and scaling to improve model accuracy.  
- Train, evaluate, and interpret models to identify the most influential factors affecting marks.  
- Demonstrate a reproducible end-to-end ML pipeline.

---

## 🧩 3. Dataset Description

**Source:** [Kaggle – Student Habits vs Performance Dataset](https://www.kaggle.com/)  
**Target Variable:** Exam Score  

| Property | Description |
|:--|:--|
| Rows | 909 |
| Columns | 16 |

The dataset includes variables such as study hours, attendance, previous marks, and other behavioral or demographic factors.

---

## ⚙️ 4. Workflow Summary

### 🧠 Step 1: Problem Definition & Objective
- **Goal:** Predict a student’s marks (regression) or grade (classification) using historical and auxiliary features.  
- **Stakeholders:** Students, teachers, and institutions — enabling early identification of at-risk students.  
- **Success Criteria:** Low prediction error (MAE, RMSE) or high accuracy/precision.

---

### 📦 Step 2: Data Collection
- Gathered data from **Kaggle Student Habits vs Performance** dataset.  
- Loaded data using Pandas, inspected shape, columns, and missing values.  
- Explored distributions and feature correlations.

---

### 🧹 Step 3: Data Pre-processing & Cleaning
- Handled missing values (imputation and removal).  
- Removed duplicates and outliers (e.g., unrealistic study hours).  
- Encoded categorical features using label or one-hot encoding.  
- Created derived features like:
  - **Attendance Rate**
  - **Study Hours per Week**
  - **Assignments Completed Ratio**
  - **Previous Average Marks**
- Applied **scaling/normalization** to numerical variables where necessary.

---

### 📊 Step 4: Exploratory Data Analysis (EDA)
- Visualized distributions (histograms, boxplots, scatterplots).  
- Analyzed correlations between study hours, attendance, and marks.  
- Identified strong predictors of exam performance.  

**Key Insights:**
- Students with < 60% attendance tended to score significantly lower.  
- Marks correlated positively with study hours and consistent performance across terms.

---

### ⚡ Step 5: Model Selection & Training
- Split data into **train (80%)** and **test (20%)** subsets.  
- Evaluated both regression and classification approaches:
  - **Regression:** Linear Regression, Decision Tree Regressor, Random Forest Regressor  
  - **Classification (optional):** Logistic Regression, Decision Tree Classifier, SVM  
- Tuned hyperparameters via **Grid Search** or **Random Search**.  
- Selected best model based on lowest RMSE and MAE.

---

### 📈 Step 6: Model Evaluation
- Measured model performance on the test set.  
- Used evaluation metrics:
  - **Regression:** MAE, RMSE, R²  
  - **Classification (if applicable):** Accuracy, Precision, Recall, F1-score  
- Visualized:
  - **Actual vs Predicted Marks (scatter plot)**
  - **Residual Errors (residual plot)**

**Best Model:** Linear Regression  
**Metric:** Lowest RMSE, highest predictive accuracy.

---

## 🧾 5. Key Takeaways

- Effective **data preprocessing**, **feature engineering**, and **scaling** significantly improved performance.  
- **Linear Regression** provided the most interpretable and accurate predictions based on RMSE.  
- The project demonstrates how simple ML techniques can make valuable predictions in educational analytics.

---

## 🧰 6. Tools and Technologies

**Language:** Python 3.10+  

**Libraries:**
- `Pandas`, `NumPy` → Data handling  
- `Matplotlib`, `Seaborn` → Visualization  
- `Scikit-Learn` → Model training, regularization, and evaluation  

---

## 🚀 7. Future Enhancements

- Experiment with **ensemble methods** (Random Forest, XGBoost, Gradient Boosting).  
- Implement **cross-validation** for better generalization.  
- Add **feature importance plots** and **error visualization**.  
- Deploy using **Flask** or **Streamlit** for interactive use.

---

## 👨‍💻 8. Author

**Name:** Harshit Vyas  
**Role:** Machine Learning Enthusiast / ML Intern Applicant  
**Objective:** To explore real-world ML workflows, build predictive systems, and develop interpretable models.

📬 **Contact:**
- Email: harshitvyashv23@gmail.com
- LinkedIn: www.linkedin.com/in/harshit-vyas-3ab537338
- GitHub: https://github.com/codebuggerhv-23

---

⭐ *If you found this project useful, don’t forget to star the repo!*

