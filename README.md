# 👨‍💼 Application of Data Science to Reduce Employee Attrition 👨‍💼

Employee attrition is a critical issue that impacts an organization's growth and performance. This project utilizes Data Science and Analytics to address this challenge. By analyzing the IBM employee survey dataset, we uncover employee profiles, identify patterns that lead to attrition, and apply predictive modeling to mitigate this issue. Ultimately, this approach aims to reduce attrition and optimize workforce management.

## 🚀 Objective

The main goal of this project is to apply advanced data science techniques to predict and mitigate employee attrition. Understanding the factors that lead to attrition enables businesses to take proactive steps to improve employee retention. Future enhancements of this project could include deeper segmentation and analysis of "at-risk" employee groups.

## 🛑 Dataset Description

This dataset contains employee survey data from IBM, including information on whether an employee left the company. With approximately 24,000 records, this dataset allows us to build a predictive model for attrition.

### Target Feature (Categorical):
- **Attrition**: Target variable to predict employee turnover.

### Categorical Features:
- Business Travel
- Department
- EducationField
- Gender
- JobRole
- MaritalStatus
- Over18
- OverTime
- Employee Source

### Numerical Features:
- Age
- DailyRate
- DistanceFromHome
- Education
- EmployeeCount
- EmployeeNumber
- Application ID
- EnvironmentSatisfaction
- HourlyRate
- JobInvolvement
- JobLevel
- JobSatisfaction
- MonthlyIncome
- MonthlyRate
- NumCompaniesWorked
- PercentSalaryHike
- PerformanceRating
- RelationshipSatisfaction
- StandardHours
- StockOptionLevel
- TotalWorkingYears
- TrainingTimesLastYear
- WorkLifeBalance
- YearsAtCompany
- YearsInCurrentRole
- YearsSinceLastPromotion
- YearsWithCurrManager

👉 link  [ 🔗IBM_Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

## 🛑 Data Preprocessing

1. **Exploratory Data Analysis (EDA)**  
   - Visualized feature distributions using bar plots, pie charts, and catplots.  
   - Identified key relationships between features and the target variable.

2. **Data Cleaning**  
   - Removed irrelevant features.  
   - Handled missing values using suitable imputation techniques.

3. **Feature Engineering**  
   - Applied Label Encoding and One-Hot Encoding to convert categorical variables into numerical representations.

4. **Model Evaluation**  
   - Used ROC and AUC metrics to evaluate the model’s performance.

## 🛑 Libraries Used

- **Pandas**: Data manipulation and analysis.
- **Numpy**: Numerical operations and mathematical functions.
- **Matplotlib & Seaborn**: Data visualization.
- **Plotly**: Interactive plots for enhanced data visualization.
- **Scikit-learn**: Machine learning models and evaluation metrics.
- **XGBoost**: High-performance gradient boosting modeling.
- **Statsmodels**: Statistical modeling and analysis.
- **Scipy**: Scientific computing tools.
- **SMOTE**: Handling class imbalance.
- **Warnings**: Suppression of irrelevant warnings.

## 🛑 Model Evaluation: XGBoost Classifier

Leveraging the XGBoost Classifier, the following evaluation metrics were achieved:

### 🔍 Recall:
- **Training Recall**: 99.91%  
  The model successfully identified 99.91% of positive cases in the training set.
- **Test Recall**: 99.17%  
  The model performed remarkably on the test set, identifying 99.17% of positive cases.

### 🎯 Accuracy:
- **Training Accuracy**: 99.96%  
  Achieved an impressive 99.96% accuracy on the training set, with nearly perfect classification.
- **Test Accuracy**: 99.86%  
  Demonstrated outstanding accuracy on the test set, with 99.86% of samples correctly classified.

### ✅ Precision:
- **Training Precision**: 100.00%  
  The model achieved flawless precision on the training set, with no false positives.
- **Test Precision**: 99.91%  
  Maintained high precision on the test set, significantly minimizing false positives.

### 📈 AUC (Area Under the Curve):
- **Training AUC**: 100.00%  
  A perfect AUC score on the training set, indicating flawless class separation.
- **Test AUC**: 99.97%  
  Almost perfect class distinction on the test set, with an AUC of 99.97%.

### 🏆 Summary:
The XGBoost Classifier exhibits exceptional performance with near-perfect metrics across all evaluation criteria:
- Strong generalization capabilities, ensuring consistent performance on unseen data.
- Maintains high classification quality, minimizing both false positives and false negatives.

### Performance Visuals:
Below are key evaluation metrics visualized for a better understanding of the model's performance:

- **Confusion Matrix**:  
  ![Confusion Matrix](YOUR_IMAGE_PATH/confusion_matrix.png)

- **ROC Curve**:  
  ![ROC Curve](YOUR_IMAGE_PATH/roc_curve.png)

## 🛑 Installation Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Prajwalpatelp/Employee_Attrition.git
```

👉 To install required libraries:

```bash
  pip install -r requirements.txt
```