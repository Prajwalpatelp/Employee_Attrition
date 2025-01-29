# 👨‍💼 Application of Data Science to Reduce Employee Attrition 👨‍💼

![Employee Attrition](image.jpeg)

Employee attrition is a major concern for organizations, affecting productivity and workforce stability. This project leverages **Data Science & Analytics** to analyze IBM datasets, uncover attrition patterns, and predict potential employee turnover. By implementing **data mining techniques**, this study aims to help businesses take proactive measures to retain valuable employees.

---

## 📊 Dataset Overview

This dataset, sourced from **IBM**, contains approximately **24,000 employee records** with attributes related to job satisfaction, salary, work-life balance, and more. The objective is to **identify attrition trends** and enhance business decision-making.

### 🔹 Target Feature:
- **Attrition** 

### 🔹 Categorical Features:
- Business Travel, Department, EducationField, Gender, JobRole, MaritalStatus, Over18, OverTime, Employee Source

### 🔹 Numerical Features:
- Age, DailyRate, DistanceFromHome, Education, EmployeeCount, EnvironmentSatisfaction, HourlyRate, JobInvolvement, JobLevel, JobSatisfaction, MonthlyIncome, MonthlyRate, NumCompaniesWorked, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction, StandardHours, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager

🔗 **Dataset Link**: [IBM HR Analytics Attrition Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

---

## 🔍 Data Preprocessing & Analysis

### 📌 Exploratory Data Analysis (EDA)
- **Data Visualization:** Used catplots, pie charts, and bar plots to analyze feature distributions.
- **Insights Extraction:** Identified key attrition drivers based on job roles, salary, and experience.

### 🛠️ Data Cleaning & Feature Engineering
- **Removed irrelevant features** to optimize model performance.
- **Handled missing values** with appropriate imputation techniques.
- **Applied label encoding & one-hot encoding** for categorical variables.

### 🏆 Model Evaluation
- **Performance Metrics:** ROC, AUC, Recall, Accuracy, Precision, Confusion matrix, Classification Report

---

## 🛑 Technologies & Libraries Used

### 🔹 Python Libraries:
- **Pandas** - Data manipulation
- **NumPy** - Numerical computations
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Machine learning & model evaluation
- **Statsmodels** - Statistical modeling
- **SciPy** - Scientific computing
- **Plotly** - Interactive visualizations

### 🔹 Machine Learning Models Implemented:

| Model                          | Description                         |
|--------------------------------|-------------------------------------|
| **Logistic Regression**        | Baseline classification model      |
| **GaussianNB**                 | Probabilistic classification       |
| **Decision Tree Classifier**   | Rule-based decision-making        |
| **Random Forest Classifier**   | Ensemble learning method          |
| **AdaBoost Classifier**        | Adaptive boosting approach        |
| **Gradient Boosting Classifier** | Sequential tree-based model      |
| **K-Nearest Neighbors**        | Distance-based classification      |
| **XGBoost Classifier**         | Optimized gradient boosting       |

### 🔹 Preprocessing & Feature Selection:
- **StandardScaler** - Feature scaling
- **LabelEncoder & OneHotEncoder** - Encoding categorical variables
- **SMOTE** - Handling class imbalance

---

## 🚀 End-to-End Machine Learning Workflow with DVC, MLflow & FastAPI  

### 📌 Project Overview  
This project demonstrates an **end-to-end machine learning pipeline** using **DVC (Data Version Control), MLflow, DagsHub, and FastAPI**. The workflow follows a structured approach for:  
👉 **Data ingestion & validation**  
👉 **Model training & evaluation**  
👉 **Experiment tracking with MLflow & DagsHub**  
👉 **Deployment with FastAPI & Docker**  
👉 **Continuous monitoring & version control using DVC**  

---

## 🔥 Workflow Overview  

```mermaid
graph TD
    A[Start] --> B[Data Ingestion]
    B --> C[Data Validation]
    C --> D[Data Transformation]
    D --> E[Model Training]
    E --> F[Experiment Tracking - MLflow with DagsHub]
    F --> G[Model Evaluation]
    G --> H[Training Pipeline]
    H --> I[Prediction Pipeline]
    I --> J[dvc.yaml - Workflow Automation]
    J --> K[FastAPI - app.py]
    K --> L[Docker - Containerization]
    L --> M[Monitoring & Maintenance]
    M --> A
```

### 🛠️ Run the Workflow

```bash
# Clone the repository
git clone https://github.com/Prajwalpatelp/Employee_Attrition.git

# Install dependencies
pip install -r requirements.txt

# Run DVC pipeline
dvc repro

# Run Training Pipeline
python src/employee/pipeline/Training_pipeline.py

# Run Prediction Pipeline
python src/employee/pipeline/Prediction_pipeline.py

# Run FastAPI app
uvicorn app:app --host 0.0.0.0 --port 8000

# Build Docker Image
docker build -t employee-attrition:latest .

# Run Docker Container
docker run -p 8000:8000 employee-attrition:latest
```

---

## 🌟 Model Performance: XGBoost Classifier

Using several models, the **XGBoost classifier** provided the best results:

### 🔍 Recall
- **Training Recall**: 99.91%
- **Test Recall**: 99.17%

### 🎯 Accuracy
- **Training Accuracy**: 99.96%
- **Test Accuracy**: 99.86%

### ✅ Precision
- **Training Precision**: 100.00%
- **Test Precision**: 99.91%

### 📈 AUC (Area Under the Curve)
- **Training AUC**: 100.00%
- **Test AUC**: 99.97%

### 📊 Confusion Matrix & AUC-ROC Curve

![Confusion Matrix](confusion_matrix.png)  
![ROC Curve](roc_curve.png)  

### 🏆 Summary
The **XGBoost Classifier** exhibits exceptional performance with near-perfect metrics:
👉 Strong generalization capabilities  
👉 High classification quality  
👉 Minimizes both false positives & false negatives  

---

## 💚 Contact Information  
👉 **For any questions or feedback, please contact:** [prajwalkumar2228@gmail.com](mailto:prajwalkumar2228@gmail.com) 💎

