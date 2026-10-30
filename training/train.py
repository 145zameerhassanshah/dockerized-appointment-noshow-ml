from pathlib import Path
from datetime import datetime, timezone
import json, joblib, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

DATA = Path("data/appointments.csv")
ART = Path("artifacts")
ART.mkdir(exist_ok=True)
TARGET = "no_show"
SELECTION_METRIC = "f1"

NUM = ["age","days_until_appointment","previous_appointments","previous_no_shows","appointment_hour"]
CAT = ["sms_reminder_sent","weekday","department"]

def preprocessor():
    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([("num", num, NUM), ("cat", cat, CAT)])

def score(name, pipe, X, y):
    pred = pipe.predict(X)
    prob = pipe.predict_proba(X)[:,1]
    return {
        "model_name": name,
        "precision": round(precision_score(y,pred,zero_division=0),4),
        "recall": round(recall_score(y,pred,zero_division=0),4),
        "f1": round(f1_score(y,pred,zero_division=0),4),
        "roc_auc": round(roc_auc_score(y,prob),4),
    }

df = pd.read_csv(DATA)
X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000,class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=250,random_state=42,class_weight="balanced",min_samples_leaf=2),
}

trained, results = {}, []
for name, clf in models.items():
    pipe = Pipeline([("preprocessor", preprocessor()),("classifier",clf)])
    pipe.fit(Xtr,ytr)
    m = score(name,pipe,Xte,yte)
    trained[name] = pipe
    results.append(m)
    print(m)

best = max(results, key=lambda r: r[SELECTION_METRIC])
best_name = best["model_name"]
version = datetime.now(timezone.utc).strftime("v%Y%m%dT%H%M%SZ")

joblib.dump(trained[best_name], ART/"best_model.joblib")
meta = {
    "model_version": version,
    "selected_model": best_name,
    "selection_metric": SELECTION_METRIC,
    "selected_model_metrics": best,
    "all_model_metrics": results,
    "training_rows": len(Xtr),
    "test_rows": len(Xte),
    "note": "Synthetic operational no-show model for educational use only."
}
(ART/"model_metadata.json").write_text(json.dumps(meta, indent=2))
print(f"Selected: {best_name} by {SELECTION_METRIC}; version={version}")
