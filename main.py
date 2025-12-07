# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from catboost import CatBoostClassifier
import joblib
import pandas as pd

MODEL_PATH = "crop_type_catboost_optuna.cbm"
META_PATH = "crop_type_metadata.pkl"

app = FastAPI(title="Crop Type Prediction API")

# ---------- Load model & metadata at startup ----------
model = CatBoostClassifier()
model.load_model(MODEL_PATH)

meta = joblib.load(META_PATH)
FEATURE_COLUMNS = meta["feature_columns"]
CATEGORICAL_COLUMNS = meta["categorical_columns"]

# ---------- Request schema ----------
class PredictRequest(BaseModel):
    # JSON will be: { "data": { "Temparature": 26.0, "Humidity": 52.0, ... } }
    data: Dict[str, Any]


@app.get("/")
def root():
    return {"message": "Crop Type Prediction API is running"}


@app.post("/predict")
def predict(req: PredictRequest):
    # req.data is a dict with feature_name: value
    row = req.data

    # Check that all required columns are present
    missing = [col for col in FEATURE_COLUMNS if col not in row]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing features: {missing}. Required: {FEATURE_COLUMNS}",
        )

    # Build DataFrame in correct column order
    X = pd.DataFrame([row])[FEATURE_COLUMNS]

    # Ensure categorical columns are strings
    for col in CATEGORICAL_COLUMNS:
        X[col] = X[col].astype(str)

    # Predict
    pred = model.predict(X)
    # CatBoost returns array-like; take first element
    crop_type = pred[0]

    return {
        "crop_type": str(crop_type),
        "input": row,
    }

# For local debugging:  uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
