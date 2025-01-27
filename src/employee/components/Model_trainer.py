import os
import sys
import numpy as np
import joblib
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score,
    precision_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import label_binarize
from src.employee.logger import logging
from src.employee.exception import customexception

# Set up MLflow environment
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/Prajwalpatelp/Employee_Attrition.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "Prajwalpatelp"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "bc25eaff1e11fab265c4d466c61af1acd9be33dd"


def save_metrics_to_file(metrics, name, path):
    """Save metrics to a text file."""
    metrics_file = os.path.join(path, f"{name}_metrics.txt")
    with open(metrics_file, "w") as file:
        for key, value in metrics.items():
            file.write(f"{key}: {value}\n")


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


def hyperparameter_tuning(X_train, y_train, X_test, y_test, models_path, threshold=0.5):
    try:
        logging.info("Starting hyperparameter tuning for Random Forest.")
        rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
        param_grid = {
            "n_estimators": [50, 100, 200, 300, 400, 500],
            "max_depth": [None, 10, 20, 30, 50, 100],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4, 6, 8],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False],
        }
        random_search = RandomizedSearchCV(
            estimator=rf_model,
            param_distributions=param_grid,
            n_iter=10,
            scoring="f1_weighted",
            cv=3,
            verbose=2,
            random_state=42,
            n_jobs=-1,
        )
        random_search.fit(X_train, y_train)

        logging.info(f"Best Parameters: {random_search.best_params_}")

        best_rf_model = random_search.best_estimator_
        y_probs_train = best_rf_model.predict_proba(X_train)
        y_probs_test = best_rf_model.predict_proba(X_test)

        metrics_train, y_pred_train = evaluate_with_threshold(y_train, y_probs_train, threshold)
        metrics_test, y_pred_test = evaluate_with_threshold(y_test, y_probs_test, threshold)

        metrics = {
            "train": metrics_train,
            "test": metrics_test,
        }

        logging.info(f"Metrics for Hyperparameter Tuning: {metrics}")

        # Save metrics to artifacts folder
        save_metrics_to_file(metrics_test, "Random_Forest_Tuned", models_path)

        # Log metrics to MLflow
        with mlflow.start_run(run_name="Random_Forest_Tuning"):
            mlflow.log_params(random_search.best_params_)
            mlflow.log_metrics({
                "train_accuracy": metrics_train["accuracy"],
                "test_accuracy": metrics_test["accuracy"],
                "train_recall": metrics_train["recall"],
                "test_recall": metrics_test["recall"],
            })
            mlflow.log_text(str(metrics_test["classification_report"]), "classification_report.txt")
            mlflow.log_text(str(metrics_test["confusion_matrix"]), "confusion_matrix.txt")

        # Save the best model
        model_path = os.path.join(models_path, "Random_Forest_Tuned_Model.pkl")
        joblib.dump(best_rf_model, model_path)
        mlflow.log_artifact(model_path)

        return best_rf_model, metrics_test["accuracy"]

    except Exception as e:
        logging.error("Error during hyperparameter tuning.")
        raise customexception(e, sys)


def ModelTrainer(data_path, models_path, params, threshold=0.5):
    try:
        logging.info("Starting model training process.")

        # Load data
        X_train_scaled = np.loadtxt(os.path.join(data_path, "X_train_scaled.csv"), delimiter=",")
        y_resampled = np.loadtxt(os.path.join(data_path, "y_resampled.csv"), delimiter=",")
        X_test_scaled = np.loadtxt(os.path.join(data_path, "X_test_scaled.csv"), delimiter=",")
        y_test = np.loadtxt(os.path.join(data_path, "y_test.csv"), delimiter=",")

        # Ensure y_test is 1D if it isn't already
        if y_test.ndim > 1:
            y_test = np.argmax(y_test, axis=1)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=200, random_state=42, class_weight='balanced'),
            "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
            "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
            "Naive Bayes": GaussianNB(),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
        }

        best_accuracy = 0
        best_model = None
        best_model_name = None

        for name, model in models.items():
            try:
                logging.info(f"Training model: {name}")
                with mlflow.start_run(run_name=name):
                    model.fit(X_train_scaled, y_resampled)
                    y_probs_train = model.predict_proba(X_train_scaled)
                    y_probs_test = model.predict_proba(X_test_scaled)

                    metrics_train, y_pred_train = evaluate_with_threshold(y_resampled, y_probs_train, threshold)
                    metrics_test, y_pred_test = evaluate_with_threshold(y_test, y_probs_test, threshold)

                    save_metrics_to_file(metrics_test, name, models_path)
                    mlflow.log_metrics({
                        "train_accuracy": metrics_train["accuracy"],
                        "test_accuracy": metrics_test["accuracy"],
                        "train_recall": metrics_train["recall"],
                        "test_recall": metrics_test["recall"],
                    })
                    mlflow.log_text(str(metrics_test["classification_report"]), "classification_report.txt")
                    mlflow.log_text(str(metrics_test["confusion_matrix"]), "confusion_matrix.txt")

                    # Update the best model selection
                    if metrics_test["accuracy"] > best_accuracy:
                        best_accuracy = metrics_test["accuracy"]
                        best_model = model
                        best_model_name = name

            except Exception as e:
                logging.error(f"Error training {name}: {e}")

        # Perform hyperparameter tuning
        logging.info("Starting hyperparameter tuning.")
        tuned_model, tuned_accuracy = hyperparameter_tuning(
            X_train_scaled, y_resampled, X_test_scaled, y_test, models_path, threshold
        )

        # Check if the tuned model performs better
        if tuned_accuracy > best_accuracy:
            best_model = tuned_model
            best_accuracy = tuned_accuracy
            best_model_name = "Random Forest (Tuned)"

        logging.info(f"Best model: {best_model_name} with Test Accuracy: {best_accuracy:.2f}")
        if best_model:
            model_path = os.path.join(models_path, "Best_Model.pkl")
            joblib.dump(best_model, model_path)
            mlflow.log_artifact(model_path)

        return best_model

    except Exception as e:
        logging.error("Error in ModelTrainer.")
        raise customexception(e, sys)


if __name__ == "__main__":
    data_path = "Artifacts/data_transformation"
    models_path = "Artifacts/models"
    os.makedirs(models_path, exist_ok=True)
    ModelTrainer(data_path, models_path, params={}, threshold=0.5)
