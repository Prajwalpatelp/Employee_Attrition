import os
import sys
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
import joblib
import mlflow
from src.employee.logger import logging
from src.employee.exception import customexception

# Set MLflow tracking environment variables
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/Prajwalpatelp/Employee_Attrition.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "Prajwalpatelp"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "bc25eaff1e11fab265c4d466c61af1acd9be33dd"


def evaluate_with_threshold(y_true, y_probs, threshold=0.5):
    """Evaluate model predictions based on a given threshold."""
    y_pred = (y_probs[:, 1] >= threshold).astype(int)  # Assuming binary classification
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "classification_report": classification_report(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return metrics, y_pred


def save_metrics_to_file(metrics, name, path):
    """Save metrics to a text file."""
    metrics_file = os.path.join(path, f"{name}_metrics.txt")
    with open(metrics_file, "w") as file:
        for key, value in metrics.items():
            if key in ["classification_report", "confusion_matrix"]:
                file.write(f"{key}:\n{value}\n")
            else:
                file.write(f"{key}: {value}\n")


def evaluate_model(data_path, model_path, threshold=0.5):
    """
    Evaluate the trained model on test data and log metrics and artifacts to MLflow.
    """
    try:
        logging.info("Starting model evaluation.")

        # End any active MLflow run to avoid conflicts
        if mlflow.active_run():
            logging.warning("Ending the active MLflow run before starting a new evaluation run.")
            mlflow.end_run()

        # Load the test data
        logging.info("Loading evaluation data.")
        X_test_scaled = np.loadtxt(os.path.join(data_path, "X_test_scaled.csv"), delimiter=",")
        y_test = np.loadtxt(os.path.join(data_path, "y_test.csv"), delimiter=",")

        # Ensure y_test is 1D if it isn't already
        if y_test.ndim > 1:
            y_test = np.argmax(y_test, axis=1)

        # Load the trained model
        logging.info("Loading trained model.")
        model = joblib.load(model_path)

        # Make predictions
        logging.info("Making predictions on test data.")
        y_probs = model.predict_proba(X_test_scaled)

        # Evaluate metrics using threshold
        metrics_test, y_pred = evaluate_with_threshold(y_test, y_probs, threshold)

        # Log metrics and artifacts to MLflow
        logging.info("Logging evaluation metrics and artifacts to MLflow.")
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        eval_dir = os.path.join("Artifacts", "evaluation")
        os.makedirs(eval_dir, exist_ok=True)

        # Save metrics and artifacts
        report_path = os.path.join(eval_dir, "classification_report.txt")
        conf_matrix_path = os.path.join(eval_dir, "confusion_matrix.npy")
        metrics_file_path = os.path.join(eval_dir, "evaluation_metrics.txt")

        with open(report_path, "w") as f:
            f.write(metrics_test["classification_report"])
        np.save(conf_matrix_path, metrics_test["confusion_matrix"])
        save_metrics_to_file(metrics_test, "evaluation", eval_dir)

        # Start a new MLflow run
        with mlflow.start_run(run_name="Model Evaluation"):
            # Log metrics
            mlflow.log_metrics({
                "test_accuracy": metrics_test["accuracy"],
                "test_precision": metrics_test["precision"],
                "test_recall": metrics_test["recall"],
            })

            # Log artifacts
            mlflow.log_artifact(report_path)
            mlflow.log_artifact(conf_matrix_path)
            mlflow.log_artifact(metrics_file_path)

        # Log results to the console and file
        logging.info(f"Test Accuracy: {metrics_test['accuracy'] * 100:.2f}%")
        logging.info(f"Test Precision: {metrics_test['precision']:.2f}")
        logging.info(f"Test Recall: {metrics_test['recall']:.2f}")
        logging.info("Classification Report:\n" + metrics_test["classification_report"])
        logging.info(f"Confusion Matrix:\n{metrics_test['confusion_matrix']}")

        print(f"Model Test Accuracy: {metrics_test['accuracy'] * 100:.2f}%")
        print(f"Classification Report:\n{metrics_test['classification_report']}")
        print(f"Confusion Matrix:\n{metrics_test['confusion_matrix']}")

    except Exception as e:
        logging.error("An error occurred during model evaluation.")
        raise customexception(e, sys)


if __name__ == "__main__":
    try:
        data_path = "Artifacts/data_transformation"
        model_path = "Artifacts/models/Best_Model.pkl"
        evaluate_model(data_path, model_path, threshold=0.5)
    except Exception as e:
        logging.error("An error occurred while running the evaluation script.")
        raise customexception(e, sys)
