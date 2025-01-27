import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
from src.employee.logger import logging
from src.employee.exception import customexception
from src.employee.components.Data_ingestion import DataIngestionConfig  # Importing ingestion config

def DataTransformation():
    """
    Perform data transformation, including cleaning, feature engineering, scaling, and encoding.
    """
    try:
        logging.info("Starting data transformation.")

        # Use the ingestion configuration to get the raw data path
        ingestion_config = DataIngestionConfig()
        data_path = ingestion_config.raw_data_path

        # Check if the raw data file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
        logging.info(f"Loading data from {data_path}")
        data = pd.read_csv(data_path)

        # Dropping invalid and null values
        invalid_conditions = (
            (data['JobInvolvement'].isin([47.0, 54.0])) |
            (data['EmployeeCount'].astype(str).isin(['Marketing'])) |
            (data['TrainingTimesLastYear'].isin([22.0, 30.0])) |
            (data['StockOptionLevel'].isin([80.0])) |
            (data['PerformanceRating'].isin([11.0, 13.0])) |
            (data['PercentSalaryHike'].isin(['No', 'Yes'])) |
            (data['Over18'].isin(['1', '2'])) |
            (data['NumCompaniesWorked'].isin([4933.0, 23258.0])) |
            (data['MonthlyIncome'].isin(['Single', 'Married'])) |
            (data['MaritalStatus'].isin([4])) |
            (data['JobSatisfaction'].isin(['Manager'])) |
            (data['JobRole'].isin([5, 4])) |
            (data['HourlyRate'].isin(['Male', 'Female'])) |
            (data['Gender'].isin(['1', '2'])) |
            (data['EnvironmentSatisfaction'].isin([127249.0, 129588.0])) |
            (data['EducationField'].isin(['3', 'Test'])) |
            (data['DistanceFromHome'].isin(['Research & Development'])) |
            (data['Department'].isin(['1296'])) |
            (data['Application ID'].isin(['TESTING', 'Test', '???'])) |
            (data['EmployeeNumber'].isin(['Test', 'TESTING', 'TEST'])) |
            (data['Employee Source'].isin(['Test']))
        )
        to_drop = data.isna().any(axis=1) | invalid_conditions
        cleaned_data = data[~to_drop]

        # Save cleaning report
        cleaning_report_path = 'Artifacts/data_transformation/cleaning_report.txt'
        os.makedirs(os.path.dirname(cleaning_report_path), exist_ok=True)
        with open(cleaning_report_path, 'w') as f:
            f.write(f"Original dataset had {data.shape[0]} rows.\n")
            f.write(f"Cleaned dataset has {cleaned_data.shape[0]} rows.\n")

        # Converting columns to integer
        columns_to_convert = [
            'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeCount',
            'EmployeeNumber', 'Application ID', 'EnvironmentSatisfaction', 'HourlyRate',
            'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',
            'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
            'StandardHours', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
            'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
            'YearsWithCurrManager'
        ]
        for column in columns_to_convert:
            cleaned_data[column] = pd.to_numeric(cleaned_data[column], errors='coerce')
        cleaned_data.dropna(subset=columns_to_convert, inplace=True)
        for column in columns_to_convert:
            cleaned_data[column] = cleaned_data[column].astype(int)

        # Perform t-tests for numeric columns
        t_test_results = []
        for col in cleaned_data.select_dtypes(include=np.number).columns:
            if col != 'Attrition':
                current_employee = cleaned_data[cleaned_data['Attrition'] == 'Current employee'][col]
                voluntary_resignation = cleaned_data[cleaned_data['Attrition'] == 'Voluntary Resignation'][col]
                _, p_value = stats.ttest_ind(current_employee, voluntary_resignation, nan_policy='omit')
                t_test_results.append(f"P-value for '{col}' between 'Current employee' and 'Voluntary Resignation': {p_value:.4f}")
        t_test_results_path = 'Artifacts/data_transformation/t_test_results.txt'
        with open(t_test_results_path, 'w') as f:
            f.write("\n".join(t_test_results))

        # Drop unnecessary columns
        cleaned_data.drop(['EmployeeCount', 'EmployeeNumber', 'Application ID', 'StandardHours', 'Over18'], axis=1, inplace=True)

        # Encoding 'Attrition' column
        cleaned_data['Attrition'] = cleaned_data['Attrition'].apply(lambda x: 1 if x == 'Voluntary Resignation' else 0)

        # Identifying and saving outliers
        numeric_columns = cleaned_data.select_dtypes(include='int32').columns
        outliers_info = []
        for column in numeric_columns:
            Q1 = cleaned_data[column].quantile(0.25)
            Q3 = cleaned_data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = cleaned_data[(cleaned_data[column] < lower_bound) | (cleaned_data[column] > upper_bound)]
            if not outliers.empty:
                outliers_info.append(f"Outliers in {column}:\n{outliers[[column]].to_string(index=False)}\n")
        outlier_info_path = 'Artifacts/data_transformation/outlier_info.txt'
        with open(outlier_info_path, 'w') as f:
            f.write("\n".join(outliers_info))

        # Removing outliers using Z-score
        columns_to_check = [
            'Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome',
            'MonthlyRate', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole',
            'YearsSinceLastPromotion', 'YearsWithCurrManager'
        ]
        z_scores = stats.zscore(cleaned_data[columns_to_check], nan_policy='omit')
        cleaned_data = cleaned_data[(abs(z_scores) < 3).all(axis=1)]

        # Checking skewness
        continuous_features = [
            'Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome',
            'MonthlyRate', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole',
            'YearsSinceLastPromotion', 'YearsWithCurrManager'
        ]
        skewness = cleaned_data[continuous_features].skew()
        skewness_path = 'Artifacts/data_transformation/skewness_before_boxcox.txt'
        with open(skewness_path, 'w') as f:
            f.write("Skewness of continuous features:\n")
            f.write(skewness.to_string())

        # Apply Box-Cox transformation (handling negative values)
        cleaned_data[continuous_features] = cleaned_data[continuous_features].apply(
            lambda x: stats.boxcox(x + 1)[0] if (x + 1).min() > 0 else x
        )

        # Encoding categorical columns
        le = LabelEncoder()
        cleaned_data['OverTime'] = le.fit_transform(cleaned_data['OverTime'])
        cleaned_data['Gender'] = le.fit_transform(cleaned_data['Gender'])
        cleaned_data = pd.get_dummies(cleaned_data, drop_first=True)

        # Save the cleaned data
        cleaned_data_path = 'Artifacts/data_transformation/df_final.csv'
        cleaned_data.to_csv(cleaned_data_path, index=False)

        # Splitting the dataset
        X = cleaned_data.drop(['Attrition'], axis=1)
        y = cleaned_data['Attrition']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Applying SMOTE
        smote = SMOTE(sampling_strategy=0.7, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        # Standard scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_resampled)
        X_test_scaled = scaler.transform(X_test)

        # Save scaled data and preprocessor
        np.savetxt('Artifacts/data_transformation/X_train_scaled.csv', X_train_scaled, delimiter=',')
        np.savetxt('Artifacts/data_transformation/y_resampled.csv', y_resampled, delimiter=',')
        np.savetxt('Artifacts/data_transformation/X_test_scaled.csv', X_test_scaled, delimiter=',')
        np.savetxt('Artifacts/data_transformation/y_test.csv', y_test, delimiter=',')

        # Saving the preprocessor including dummy columns and encoders after scaling
        preprocessor = {
            'scaler': scaler,  # Using the scaler that was fit to the resampled data
            'columns': X.columns.tolist(),
        }

        # Save the preprocessor
        joblib.dump(preprocessor, 'Artifacts/data_transformation/Preprocessor.pkl')

        logging.info("Data transformation completed successfully.")

    except Exception as e:
        logging.error("An error occurred during data transformation.")
        raise customexception(e, sys)

if __name__ == "__main__":
    DataTransformation()
