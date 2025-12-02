from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

MODEL_PATH = os.environ.get("MODEL_PATH", "models/rf_baseline.joblib")
SCALER_PATH = os.environ.get("SCALER_PATH", "models/scaler.joblib")

app = FastAPI(title="Manufacturing Defect Prediction API")

class PredictRequest(BaseModel):
    features: list[float]

@app.on_event("startup")
def load_artifacts():
    global model, scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

@app.post("/predict")
def predict(req: PredictRequest):
    X = np.array(req.features, dtype=float).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    return {"prediction": int(pred[0])}
