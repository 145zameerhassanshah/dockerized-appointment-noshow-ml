from pathlib import Path
import json, joblib, pandas as pd
from fastapi import FastAPI, HTTPException
from api.schemas import AppointmentFeatures

ART = Path("artifacts")
model = joblib.load(ART/"best_model.joblib")
meta = json.loads((ART/"model_metadata.json").read_text())

app = FastAPI(
    title="Appointment No-Show Prediction API",
    version="1.0.0",
    description="Educational operational ML service; not for clinical decision-making."
)

@app.get("/health")
def health():
    return {
        "status":"ok",
        "model_name":meta["selected_model"],
        "model_version":meta["model_version"],
        "selection_metric":meta["selection_metric"]
    }

@app.post("/predict")
def predict(x: AppointmentFeatures):
    if x.previous_no_shows > x.previous_appointments:
        raise HTTPException(400, "previous_no_shows cannot exceed previous_appointments")
    row = pd.DataFrame([x.model_dump()])
    pred = int(model.predict(row)[0])
    prob = float(model.predict_proba(row)[0,1])
    return {
        "prediction_class": pred,
        "risk_label": "higher_no_show_likelihood" if pred == 1 else "lower_no_show_likelihood",
        "probability_class_1": round(prob,4),
        "model_name": meta["selected_model"],
        "model_version": meta["model_version"],
        "selection_metric": meta["selection_metric"],
        "disclaimer": "Operational educational estimate only."
    }
