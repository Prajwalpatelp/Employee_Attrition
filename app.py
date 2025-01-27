import logging
import pandas as pd
from io import StringIO
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import List
from src.employee.pipeline.Prediction_pipeline import PredictionPipeline

# Initialize FastAPI app
app = FastAPI()

# Setup logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the request model using Pydantic
class EmployeeData(BaseModel):
    Age: int
    BusinessTravel: str
    DailyRate: int
    Department: str
    DistanceFromHome: int
    Education: int
    EducationField: str
    EnvironmentSatisfaction: int
    Gender: str
    HourlyRate: int
    JobInvolvement: int
    JobLevel: int
    JobRole: str
    JobSatisfaction: int
    MaritalStatus: str
    MonthlyIncome: int
    MonthlyRate: int
    NumCompaniesWorked: int
    OverTime: str
    PercentSalaryHike: int
    PerformanceRating: int
    RelationshipSatisfaction: int
    StockOptionLevel: int
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int
    EmployeeSource: str

# Initialize the prediction pipeline
pipeline = PredictionPipeline()

@app.post("/predict/")
async def predict_employee_status(employee_data: EmployeeData, threshold: float = 0.3):
    try:
        input_data = employee_data.dict()
        logger.info(f"Received input data: {input_data}")
        prediction = pipeline.predict(input_data, threshold)
        logger.info(f"Prediction result: {prediction}")
        return {"prediction": prediction}
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")

@app.post("/predict-file/")
async def predict_from_file(file: UploadFile = File(...), threshold: float = 0.3):
    try:
        logger.info(f"Received file: {file.filename}")
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))

        logger.info(f"Input data:\n{df.head()}")

        expected_columns = list(EmployeeData.schema()["properties"].keys())
        if not all(col in df.columns for col in expected_columns):
            missing_cols = set(expected_columns) - set(df.columns)
            raise HTTPException(status_code=400, detail=f"Missing columns in uploaded file: {missing_cols}")

        results = []
        for _, row in df.iterrows():
            input_data = row.to_dict()
            prediction = pipeline.predict(input_data, threshold)
            results.append({"input": input_data, "prediction": prediction})

        return {"predictions": results}
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Employee Prediction API. Use the /predict and /predict-file endpoints to make predictions."}
