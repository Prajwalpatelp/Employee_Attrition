# 🚀 End-to-End Machine Learning Workflow with DVC, MLflow & FastAPI  

## 📌 Project Overview  

This project demonstrates an **end-to-end machine learning pipeline** using **DVC (Data Version Control), MLflow, DagsHub, and FastAPI**. The workflow follows a structured approach for:  
✅ **Data ingestion & validation**  
✅ **Model training & evaluation**  
✅ **Experiment tracking with MLflow & DagsHub**  
✅ **Deployment with FastAPI & Docker**  
✅ **Continuous monitoring & version control using DVC**  

The project is designed for a **warehouse inventory management system** and follows best practices in **MLOps**.  

---

## 🔥 Workflow Overview  

```mermaid
graph TD
    A[Start] --> B[Data Ingestion]
    B --> C[Data Validation]
    C --> D[Data Transformation]
    D --> E[Model Training]
    E --> F[Experiment Tracking - MLflow & DagsHub]
    F --> G[Model Evaluation]
    G --> H[Training Pipeline]
    H --> I[Prediction Pipeline]
    I --> J[dvc.yaml - Workflow Automation]
    J --> K[FastAPI - app.py]
    K --> L[Docker - Containerization]
    L --> M[Monitoring & Maintenance]
    M --> A
