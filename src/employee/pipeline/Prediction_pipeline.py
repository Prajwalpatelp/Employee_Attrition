import os
import sys
import joblib
import json
import pandas as pd
from src.employee.logger import logging
from src.employee.exception import customexception


class PredictionPipeline:
    """Pipeline for loading the trained model and making predictions."""

    def __init__(self):
        """Initialize the PredictionPipeline with the trained model and preprocessor."""
        try:
            model_path = os.path.join("Artifacts", "models", "Best_Model.pkl")
            preprocessor_path = os.path.join("Artifacts", "data_transformation", "Preprocessor.pkl")

            # Load the trained model
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Trained model not found at {model_path}")
            self.model = joblib.load(model_path)

            # Load the preprocessor
            if not os.path.exists(preprocessor_path):
                raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")
            self.preprocessor = joblib.load(preprocessor_path)

            logging.info("Prediction pipeline initialized successfully.")
        except Exception as e:
            logging.error("Error initializing the prediction pipeline.")
            raise customexception(e, sys)

    def predict(self, data, threshold=0.5):
        """
        Preprocess the input data and make predictions.

        Parameters:
        - data: Dictionary of input values (column_name: value)
        - threshold: Probability threshold for classifying as "Voluntary Resignation."

        Returns:
        - Prediction result as a label (e.g., 'Current Employee' or 'Voluntary Resignation')
        """
        try:
            # Convert input data to DataFrame
            input_df = pd.DataFrame([data])

            logging.info(f"Input DataFrame before preprocessing:\n{input_df}")

            # Align input DataFrame with expected columns
            expected_columns = self.preprocessor["columns"]
            input_df = pd.get_dummies(input_df)
            input_df = input_df.reindex(columns=expected_columns, fill_value=0)

            # Preprocess the data using the scaler
            scaler = self.preprocessor["scaler"]
            scaled_data = scaler.transform(input_df)

            logging.info(f"Scaled data (input to the model):\n{scaled_data}")

            # Make predictions using the trained model
            probabilities = self.model.predict_proba(scaled_data)[0]
            prediction_numeric = int(probabilities[1] >= threshold)

            # Log probabilities for debugging
            logging.info(f"Prediction probabilities: {probabilities}")
            logging.info(f"Applied threshold: {threshold}")
            logging.info(f"Prediction numeric result: {prediction_numeric}")

            # Map numeric prediction to human-readable labels
            label_mapping = {0: "Current Employee", 1: "Voluntary Resignation"}
            prediction_label = label_mapping.get(prediction_numeric, "Unknown")

            logging.info(f"Prediction label result: {prediction_label}")

            # Save the prediction result to a JSON file
            result_path = os.path.join("Artifacts", "predictions", "prediction_results.json")
            os.makedirs(os.path.dirname(result_path), exist_ok=True)

            prediction_result = {
                "input_data": data,
                "prediction": prediction_label
            }

            with open(result_path, 'w') as f:
                json.dump(prediction_result, f, indent=4)

            logging.info(f"Prediction results saved to {result_path}")

            return prediction_label
        except Exception as e:
            logging.error("Error during prediction.")
            raise customexception(e, sys)


# Entry point for standalone script execution
if __name__ == "__main__":
    try:
        logging.info("Testing the PredictionPipeline.")

        # Updated sample input to produce "Voluntary Resignation"
        sample_input = {
            'Age': 35,
            'BusinessTravel': 'Travel_Frequently',
            'DailyRate': 250,
            'Department': 'Sales',
            'DistanceFromHome': 40,
            'Education': 3,
            'EducationField': 'Marketing',
            'EnvironmentSatisfaction': 1,  # Low satisfaction
            'Gender': 'Male',
            'HourlyRate': 15,
            'JobInvolvement': 1,  # Low involvement
            'JobLevel': 1,
            'JobRole': 'Sales Executive',
            'JobSatisfaction': 1,  # Low satisfaction
            'MaritalStatus': 'Single',
            'MonthlyIncome': 1800,  # Low income
            'MonthlyRate': 8000,
            'NumCompaniesWorked': 6,  # High job-hopping
            'OverTime': 'Yes',  # Overtime workload
            'PercentSalaryHike': 3,  # Minimal hike
            'PerformanceRating': 2,
            'RelationshipSatisfaction': 1,  # Poor relationships
            'StockOptionLevel': 0,
            'TotalWorkingYears': 4,  # Low experience
            'TrainingTimesLastYear': 0,  # No training
            'WorkLifeBalance': 1,  # Poor balance
            'YearsAtCompany': 1,  # Short tenure
            'YearsInCurrentRole': 1,
            'YearsSinceLastPromotion': 0,  # No promotion
            'YearsWithCurrManager': 1,
            'EmployeeSource': 'Referral',
        }

        pipeline = PredictionPipeline()
        prediction = pipeline.predict(sample_input, threshold=0.3)  # Adjusted threshold
        print(f"Prediction result: {prediction}")
    except Exception as e:
        print(f"Error: {e}")
