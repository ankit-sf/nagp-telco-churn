import json
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "churn_model.pkl"
METADATA_PATH = BASE_DIR / "model" / "model_metadata.json"


app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="Predicts whether a telecom customer is likely to churn.",
    version="1.0.0",
)


class CustomerData(BaseModel):
    """
    Flexible request body for customer data.

    I am accepting a dictionary instead of a very strict schema because the original
    Telco dataset has many categorical fields. The API still validates that all required
    model columns are present before making a prediction.
    """

    customer: Dict[str, Any]


def load_model_and_metadata():
    """Load the saved sklearn pipeline and metadata."""
    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Model file was not found. Please run `python train_model.py` first."
        )

    if not METADATA_PATH.exists():
        raise RuntimeError(
            "Model metadata was not found. Please run `python train_model.py` first."
        )

    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return model, metadata


model, metadata = load_model_and_metadata()


@app.get("/")
def health_check():
    return {
        "message": "Customer Churn Prediction API is running.",
        "predict_endpoint": "/predict",
    }


@app.post("/predict")
def predict_churn(payload: CustomerData):
    """
    Predict customer churn from JSON input.

    Expected request format:
    {
        "customer": {
            "gender": "Female",
            "SeniorCitizen": 0,
            ...
        }
    }
    """

    customer_dict = payload.customer
    required_columns = metadata["required_columns"]

    missing_columns = [
        column for column in required_columns if column not in customer_dict
    ]

    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Invalid input. Some required fields are missing.",
                "missing_columns": missing_columns,
            },
        )

    try:
        input_df = pd.DataFrame([customer_dict], columns=required_columns)

        prediction_numeric = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])

        prediction_label = "Yes" if prediction_numeric == 1 else "No"

        return {
            "prediction": prediction_label,
            "churn_probability": round(probability, 4),
        }

    except Exception as error:
        # This gives the user a helpful message while keeping the API from crashing.
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Prediction failed. Please check input values and data types.",
                "error": str(error),
            },
        )