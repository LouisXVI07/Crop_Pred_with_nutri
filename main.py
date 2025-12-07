from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
from catboost import CatBoostClassifier
import joblib
import pandas as pd

MODEL_PATH = "crop_type_catboost_optuna.cbm"
META_PATH = "crop_type_metadata.pkl"

app = FastAPI(title="Crop Type Prediction API")

# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # for production, you can restrict to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load model & metadata ----
model = CatBoostClassifier()
model.load_model(MODEL_PATH)

meta = joblib.load(META_PATH)
FEATURE_COLUMNS = meta["feature_columns"]
CATEGORICAL_COLUMNS = meta["categorical_columns"]

class PredictRequest(BaseModel):
    data: Dict[str, Any]

@app.get("/")
def root():
    return {"message": "Crop Type Prediction API is running"}

@app.post("/predict")
def predict(req: PredictRequest):
    row = req.data

    missing = [col for col in FEATURE_COLUMNS if col not in row]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing features: {missing}. Required: {FEATURE_COLUMNS}",
        )

    X = pd.DataFrame([row])[FEATURE_COLUMNS]

    for col in CATEGORICAL_COLUMNS:
        X[col] = X[col].astype(str)

    pred = model.predict(X)
    crop_type = pred[0]

    return {
        "crop_type": str(crop_type),
        "input": row,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
