import os
import sys
import pandas as pd
from src.employee.logger import logging
from src.employee.exception import customexception
from dataclasses import dataclass
from src.employee.components.Data_ingestion import DataIngestionConfig


@dataclass
class DataValidationConfig:
    """Configuration for data validation."""
    raw_data_path: str = DataIngestionConfig().raw_data_path
    artifacts_dir: str = os.path.join("Artifacts", "Data_validation")


class DataValidation:
    """
    Class for validating and performing exploratory data analysis (EDA) 
    on the raw dataset.
    """
    def __init__(self):
        """Initialize the DataValidation class with configuration."""
        self.validation_config = DataValidationConfig()
        os.makedirs(self.validation_config.artifacts_dir, exist_ok=True)

    def validate_data(self):
        """
        Perform exploratory data analysis (EDA) and validate the dataset.
        Save the outputs in the specified artifacts directory.
        """
        try:
            # Check if the raw data file exists
            if not os.path.exists(self.validation_config.raw_data_path):
                raise FileNotFoundError(
                    f"Raw data file not found at {self.validation_config.raw_data_path}"
                )

            logging.info("Reading raw data for validation.")
            data = pd.read_csv(self.validation_config.raw_data_path)

            # Initialize artifact paths
            summary_txt_path = os.path.join(self.validation_config.artifacts_dir, "data_summary.txt")
            missing_csv_path = os.path.join(self.validation_config.artifacts_dir, "missing_summary.csv")
            unique_values_txt_path = os.path.join(self.validation_config.artifacts_dir, "unique_values.txt")
            duplicate_txt_path = os.path.join(self.validation_config.artifacts_dir, "duplicate_check.txt")

            # Start generating outputs
            with open(summary_txt_path, 'w') as summary_file:
                summary_file.write("Features/Columns:\n")
                summary_file.write(f"{data.columns.tolist()}\n")
                summary_file.write("*" * 90 + "\n\n")

                summary_file.write(f"Dataset Shape: Rows = {data.shape[0]}, Columns = {data.shape[1]}\n")
                summary_file.write("*" * 90 + "\n\n")

                summary_file.write("Data Types:\n")
                summary_file.write(f"{data.dtypes}\n")
                summary_file.write("*" * 90 + "\n\n")

                contains_nan = data.isna().values.any()
                summary_file.write(f"Contains NaN values in the dataset: {contains_nan}\n")
                summary_file.write("*" * 90 + "\n\n")

                empty_cells_by_column = data.isna().sum()
                summary_file.write("Total Empty Cells by Columns:\n")
                summary_file.write(f"{empty_cells_by_column}\n\n")
                summary_file.write("*" * 90 + "\n\n")

                total_missing = data.isnull().sum().sum()
                null_percentage = round((total_missing * 100) / data.size, 2)
                summary_file.write(f"The dataset contains {null_percentage}% null values.\n")
                summary_file.write("*" * 90 + "\n\n")

            # Calculate missing values and save to CSV
            missing_values = data.isna().sum()
            missing_percentage = round(100 * missing_values / data.shape[0], 2)
            missing_summary = (
                pd.DataFrame({
                    'Column': data.columns,
                    'No. Missing Values': missing_values,
                    'Percent': missing_percentage
                })
                .sort_values(by='Percent', ascending=False)
                .reset_index(drop=True)
            )
            missing_summary.to_csv(missing_csv_path, index=False)
            logging.info(f"Missing values summary saved to {missing_csv_path}")

            # Unique values and counts
            with open(unique_values_txt_path, 'w') as unique_file:
                for col in data.columns:
                    unique_file.write(f"Number of unique values in '{col}' are - {data[col].nunique()}\n")
                    unique_file.write(f"Counts of each value in '{col}':\n{data[col].value_counts()}\n")
                    unique_file.write("\n" + "-" * 90 + "\n\n")
            logging.info(f"Unique values summary saved to {unique_values_txt_path}")

            # Check for duplicate rows
            duplicates_exist = data.duplicated().any()
            with open(duplicate_txt_path, 'w') as duplicate_file:
                duplicate_file.write(f"Does the dataset contain duplicate rows? {duplicates_exist}\n")
            logging.info(f"Duplicate check saved to {duplicate_txt_path}")

            logging.info("Data validation completed successfully.")
            print("Data validation completed. Artifacts are saved.")

        except Exception as e:
            logging.error("An exception occurred during data validation.")
            raise customexception(e, sys)


if __name__ == "__main__":
    print("Data validation script started.")
    try:
        validation = DataValidation()
        validation.validate_data()
    except Exception as e:
        print(f"An error occurred: {e}")
