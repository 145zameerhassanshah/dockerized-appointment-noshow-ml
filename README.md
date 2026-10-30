# Smart Appointment No-Show Prediction Pipeline

Real-world-inspired CMIT AI Internship project that estimates appointment no-show likelihood so clinic staff can prioritize reminder and confirmation effort.

## Problem Solved
Missed appointments waste clinic capacity. This tool ranks appointment no-show likelihood using historical scheduling features. It does not make medical decisions.

## Training Pipeline
Synthetic appointment CSV → preprocessing → Logistic Regression + Random Forest → Precision/Recall/F1/ROC-AUC → best model selected automatically by F1 → saved model + preprocessing pipeline + metadata.

## Serving Pipeline
Saved pipeline → Dockerized FastAPI → `/predict` → validated request → probability + class + model name + model version.

## Quick Start

### 1. Build trainer
```bash
docker build -f Dockerfile.train -t appointment-noshow-trainer .
```

### 2. Train and persist artifacts
```bash
mkdir -p artifacts
docker run --rm -v "$(pwd)/artifacts:/app/artifacts" appointment-noshow-trainer
```

### 3. Inspect model metadata
```bash
cat artifacts/model_metadata.json
```

### 4. Build API
```bash
docker build -f Dockerfile.api -t appointment-noshow-api .
```

### 5. Run API
```bash
docker run --name appointment-noshow-api-container -p 8000:8000 appointment-noshow-api
```

### 6. Health
```bash
curl http://localhost:8000/health
```

### 7. Predict
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"age":42,"days_until_appointment":7,"previous_appointments":6,"previous_no_shows":2,"sms_reminder_sent":"yes","appointment_hour":15,"weekday":"Monday","department":"General Medicine"}'
```

### 8. Swagger
Open `http://localhost:8000/docs`

### 9. Tests
After artifacts exist:
```bash
pip install -r requirements-api.txt
pytest -v
```

## Important Limitation
Synthetic educational data only. No clinical use, no real patient data, no production-readiness claim.
