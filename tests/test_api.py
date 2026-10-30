from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)
VALID = {
    "age":42,
    "days_until_appointment":7,
    "previous_appointments":6,
    "previous_no_shows":2,
    "sms_reminder_sent":"yes",
    "appointment_hour":15,
    "weekday":"Monday",
    "department":"General Medicine",
}

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_version"]

def test_predict():
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    b = r.json()
    assert b["prediction_class"] in [0,1]
    assert 0 <= b["probability_class_1"] <= 1
    assert b["model_version"]

def test_invalid_age():
    bad = dict(VALID); bad["age"] = 150
    assert client.post("/predict", json=bad).status_code == 422

def test_impossible_history():
    bad = dict(VALID); bad["previous_appointments"] = 1; bad["previous_no_shows"] = 3
    assert client.post("/predict", json=bad).status_code == 400
