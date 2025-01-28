👨‍💼👨‍💼 Application of Data Science to Reduce Employee Attrition 👨‍💼👨‍💼
Employee attrition is a critical issue that impacts an organization's growth and performance. This project aims to tackle the issue using Data Science and Analytics. By analyzing IBM datasets, we uncover employee profiles, identify patterns that lead to attrition, and apply predictive modeling to proactively address this challenge. Through this approach, we aim to reduce attrition and optimize workforce management.

🚀 Objective
The goal of this project is to apply advanced data science techniques to predict and mitigate employee attrition. By understanding the factors leading to attrition, businesses can take proactive measures to improve employee retention. Future expansions of this project could involve deeper segmentation and analysis of "at-risk" employee groups.

🛑 Dataset Description
This dataset includes employee survey data from IBM and indicates whether an employee left the company. With approximately 24,000 records, this dataset allows us to model attrition prediction.

Target Feature (Categorical):
Attrition: Target variable to predict employee turnover.
Categorical Features:
Business Travel, Department, EducationField, Gender, JobRole, MaritalStatus, Over18, OverTime, Employee Source
Numerical Features:
Age, DailyRate, DistanceFromHome, Education, EmployeeCount, EmployeeNumber, Application ID, EnvironmentSatisfaction, HourlyRate, JobInvolvement, JobLevel, JobSatisfaction, MonthlyIncome, MonthlyRate, NumCompaniesWorked, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction, StandardHours, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager
👉 Link to the dataset: IBM Attrition Dataset

🛑 Data Preprocessing
1. Exploratory Data Analysis (EDA)
Visualized feature distributions using bar plots, pie charts, and catplots.
Identified key relationships between features and the target variable.
2. Data Cleaning
Removed irrelevant features.
Handled missing values using suitable imputation techniques.
3. Feature Engineering
Applied Label Encoding and One-Hot Encoding to convert categorical variables into numerical representations.
4. Model Evaluation
Used ROC and AUC metrics to evaluate the model’s performance.
🛑 Libraries Used
Pandas: Data manipulation and analysis.
Numpy: Numerical operations and mathematical functions.
Matplotlib & Seaborn: Data visualization.
Plotly: Interactive plots for enhanced data visualization.
Scikit-learn: Machine learning models and evaluation metrics.
XGBoost: Gradient boosting for high-performance modeling.
Statsmodels: Statistical modeling and analysis.
Scipy: Scientific computing tools.
SMOTE: Handling class imbalance.
Warnings: Suppression of irrelevant warnings.
🛑 Model Evaluation: XGBoost Classifier
Leveraging the XGBoost Classifier, the following evaluation metrics were achieved:

🔍 Recall

Training Recall: 99.91%
The model successfully identified 99.91% of positive cases in the training set.
Test Recall: 99.17%
It performed remarkably on the test set, identifying 99.17% of positive cases.
🎯 Accuracy

Training Accuracy: 99.96%
Achieved an impressive 99.96% accuracy on the training set, with nearly perfect classification.
Test Accuracy: 99.86%
Demonstrated outstanding accuracy on the test set, with 99.86% of samples correctly classified.
✅ Precision

Training Precision: 100.00%
The model achieved flawless precision on the training set, with no false positives.
Test Precision: 99.91%
Maintained high precision on the test set, significantly minimizing false positives.
📈 AUC (Area Under the Curve)

Training AUC: 100.00%
A perfect AUC score on the training set, indicating flawless class separation.
Test AUC: 99.97%
Almost perfect class distinction on the test set, with an AUC of 99.97%.
🏆 Summary
The XGBoost Classifier exhibits exceptional performance with near-perfect metrics across all evaluation criteria:

Strong generalization capabilities, ensuring consistent performance on unseen data.
Maintains high classification quality, minimizing both false positives and false negatives.
Performance Visuals:
Below are the key evaluation metrics visualized for a better understanding of model performance.

Confusion Matrix:


ROC Curve:


🛑 Installation Instructions
Clone the Repository:
bash
Copy
Edit
git clone https://github.com/Prajwalpatelp/Employee_Attrition.git
Install the Required Libraries:
bash
Copy
Edit
pip install -r requirements.txt
🛑 Project Workflow
The project follows a structured pipeline to ensure smooth and efficient processing. Below is the project workflow:

Directory Structure:
plaintext
Copy
Edit
├── Notebook_Experiments
│   ├── Data
│   ├── Exploratory_Data_Analysis.ipynb
│   ├── Model_Training.ipynb
├── src
│   └── employee
│       ├── __init__.py
│       ├── exception.py
│       ├── logger.py
│       ├── utils
│       │   ├── __init__.py
│       │   └── utils.py
│       └── components
│           ├── __init__.py
│           ├── Data_ingestion.py
│           ├── Data_validation.py
│           ├── Data_transformation.py
│           ├── Model_trainer.py
│           └── Model_evaluation.py
│       ├── pipeline
│           ├── __init__.py
│           ├── Prediction_pipeline.py
│           └── Training_pipeline.py
├── static
│   └── styles.css
├── .gitignore
├── app.py
├── Dockerfile
├── README.md
├── .dvcignore
├── dvc.yaml
├── requirements.txt
└── setup.py
Pipeline Steps:
Data Ingestion: The dataset is loaded and pre-processed for further analysis.
Data Validation: Checks are performed to ensure the quality and integrity of the data.
Data Transformation: The dataset is transformed through encoding, normalization, and imputation.
Model Training: Multiple machine learning models are trained and evaluated.
Model Evaluation: Evaluation metrics (AUC, accuracy) are used to assess the model's performance.
Prediction Pipeline: The final model is integrated into a prediction pipeline for real-time use.
Run the Project:
Prediction Pipeline: src/employee/pipeline/Prediction_pipeline.py
Training Pipeline: src/employee/pipeline/Training_pipeline.py
🛑 Contact Information
For any inquiries or feedback, please reach out to me at: 📧 prajwalkumar2228@gmail.com