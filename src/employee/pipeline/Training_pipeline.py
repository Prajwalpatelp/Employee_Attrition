import os
import sys
import mlflow
from src.employee.logger import logging
from src.employee.exception import customexception
from src.employee.components.Data_ingestion import DataIngestion
from src.employee.components.Data_validation import DataValidation
from src.employee.components.Data_transformation import DataTransformation
from src.employee.components.Model_trainer import ModelTrainer
from src.employee.components.Model_evaluation import evaluate_model

def start_training_pipeline():
    """
    Orchestrates the training pipeline by executing each step sequentially.
    """
    try:
        logging.info("Starting training pipeline.")

        # Ensure any active MLflow run is ended
        if mlflow.active_run():
            logging.warning("Ending the active MLflow run before starting a new one.")
            mlflow.end_run()

        # Step 1: Data Ingestion
        logging.info("Initiating data ingestion.")
        data_ingestion = DataIngestion()
        raw_data_path = data_ingestion.initiate_data_ingestion()

        # Step 2: Data Validation
        logging.info("Initiating data validation.")
        data_validation = DataValidation()
        data_validation.validate_data()

        # Step 3: Data Transformation
        logging.info("Initiating data transformation.")
        DataTransformation()  # Perform data transformation

        # Step 4: Model Training
        logging.info("Initiating model training.")
        data_path = os.path.join("Artifacts", "data_transformation")
        models_path = os.path.join("Artifacts", "models")
        os.makedirs(models_path, exist_ok=True)
        ModelTrainer(data_path, models_path, params={}, threshold=0.5)

        # Step 5: Model Evaluation
        logging.info("Initiating model evaluation.")
        model_path = os.path.join(models_path, "Best_Model.pkl")
        evaluate_model(data_path, model_path, threshold=0.5)

        logging.info("Training pipeline completed successfully.")

    except Exception as e:
        logging.error("An error occurred in the training pipeline.")
        raise customexception(e, sys)

if __name__ == "__main__":
    try:
        start_training_pipeline()
    except Exception as e:
        print(f"An error occurred: {e}")
