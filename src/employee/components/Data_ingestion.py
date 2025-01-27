import os
import sys
import pandas as pd
from src.employee.logger import logging
from src.employee.exception import customexception
from pathlib import Path
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion."""
    raw_data_path: str = os.path.join("Artifacts", "raw_data.csv")


class DataIngestion:
    def __init__(self):
        """Initialize the DataIngestion class with configuration."""
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Ingest data from a CSV file and save it to the specified path.
        """
        logging.info("Data ingestion process started.")
        try:
            # Reading data from the CSV file
            csv_file_path = "Notebook_Experiments\\Data\\employee_data.csv"
            if not os.path.exists(csv_file_path):
                raise FileNotFoundError(f"CSV file not found at {csv_file_path}")
            logging.info(f"Reading data from {csv_file_path}")
            data = pd.read_csv(csv_file_path)

            # Creating necessary directories
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f"Raw data saved at {self.ingestion_config.raw_data_path}")

            logging.info("Data ingestion process completed successfully.")

            # Returning the path for the saved raw data
            return self.ingestion_config.raw_data_path
        except Exception as e:
            logging.error("An exception occurred during data ingestion.")
            raise customexception(e, sys)


if __name__ == "__main__":
    print("Data ingestion script started.")
    try:
        ingestion = DataIngestion()
        raw_data_path = ingestion.initiate_data_ingestion()
        print(f"Data ingestion completed successfully. Raw data saved at: {raw_data_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
