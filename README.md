graph TD
    A[Start] --> B[Data Ingestion]
    B --> C[Data Validation]
    C --> D[Data Transformation]
    D --> E[Model Trainer]
    E --> F[Model Evaluation]
    F --> G[Training Pipeline (DVC)]
    G --> H[Prediction Pipeline]
    H --> I[Run dvc.yaml]
    I --> J[Run FastAPI (app.py)]
    J --> K[Dockerization (Dockerfile)]
    K --> L[Monitoring & Maintenance]
    L --> A[End]

    subgraph Components
        B1[src/employee/components/Data_ingestion.py]
        C1[src/employee/components/Data_validation.py]
        D1[src/employee/components/Data_transformation.py]
        E1[src/employee/components/Model_trainer.py]
        F1[src/employee/components/Model_evaluation.py]
        
        B --> B1
        C --> C1
        D --> D1
        E --> E1
        F --> F1
    end

    subgraph Pipelines
        G1[src/employee/pipeline/Training_pipeline.py]
        H1[src/employee/pipeline/Prediction_pipeline.py]

        G1 --> G
        H1 --> H
    end

    subgraph Utilities
        X1[src/employee/logger.py]
        X2[src/employee/exception.py]
        X3[src/employee/utils/utils.py]
    end

    X1 -.-> B
    X1 -.-> C
    X1 -.-> D
    X1 -.-> E
    X1 -.-> F

    X2 -.-> B
    X2 -.-> C
    X2 -.-> D
    X2 -.-> E
    X2 -.-> F

    X3 -.-> G
    X3 -.-> H
