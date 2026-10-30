# Task Title
Containerized Appointment No-Show ML Pipeline

# Description
Built a containerized ML training and inference pipeline for appointment no-show likelihood using automated preprocessing, Logistic Regression and Random Forest comparison, Precision/Recall/F1/ROC-AUC evaluation, automatic best-model selection, versioned model artifacts, and a separate Dockerized FastAPI prediction service with validation and automated tests.

# Key Deliverables
- Synthetic appointment dataset generator
- Missing-value preprocessing
- Numerical scaling
- Categorical one-hot encoding
- Logistic Regression
- Random Forest
- Precision, Recall, F1, ROC-AUC
- F1-based automatic best-model selection
- Saved preprocessing + model pipeline
- Joblib artifact
- JSON metadata
- UTC model versioning
- Separate training Docker image
- Separate FastAPI Docker image
- `/health`
- `/predict`
- Pydantic validation
- Probability output
- Model version in response
- Pytest automated tests
- Bind-mounted artifact persistence
- Responsible limitations documentation

# AI Assistance
ChatGPT assisted with architecture, preprocessing, metrics, model comparison, Docker separation, artifact persistence, FastAPI serving, input validation, model versioning, automated tests, documentation, and troubleshooting guidance. I execute, validate, and review the Docker builds, training metrics, API responses, and tests.

# AI Usage
Human Contribution: 70%
AI Assistance: 30%
