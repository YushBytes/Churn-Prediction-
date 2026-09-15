"""
Production prediction utilities.

The Streamlit app (and any other consumer) should only ever import from
this module — it loads the ALREADY-TRAINED pipeline from
models/churn_model.pkl and never retrains anything. This guarantees the
exact preprocessing used at training time is reused at inference time.
"""
from __future__ import annotations

import joblib
import pandas as pd

from src.evaluate import risk_level
from src.feature_engineering import add_engineered_features

MODEL_PATH = "models/churn_model.pkl"

_RAW_INPUT_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


def load_model_bundle(path: str = MODEL_PATH) -> dict:
    """Load the saved pipeline + metadata bundle produced by src/train.py."""
    return joblib.load(path)


def predict_single(customer: dict, bundle: dict) -> dict:
    """Predict churn for a single customer.

    Parameters
    ----------
    customer : dict
        Raw customer attributes matching `_RAW_INPUT_COLUMNS`.
    bundle : dict
        The object returned by `load_model_bundle`.

    Returns
    -------
    dict with keys: prediction, churn_probability, risk_level
    """
    row = {col: customer.get(col) for col in _RAW_INPUT_COLUMNS}
    df = pd.DataFrame([row])
    df = add_engineered_features(df)
    df = df[bundle["feature_columns"]]

    pipeline = bundle["pipeline"]
    threshold = bundle["threshold"]

    probability = float(pipeline.predict_proba(df)[0, 1])
    prediction = "Likely to Churn" if probability >= threshold else "Likely to Stay"

    return {
        "prediction": prediction,
        "churn_probability": probability,
        "risk_level": risk_level(probability),
        "threshold_used": threshold,
    }


def predict_batch(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Predict churn for a batch of customers (raw, un-engineered columns)."""
    df = df.copy()
    df = add_engineered_features(df)
    df = df[bundle["feature_columns"]]

    pipeline = bundle["pipeline"]
    threshold = bundle["threshold"]

    probabilities = pipeline.predict_proba(df)[:, 1]
    predictions = ["Likely to Churn" if p >= threshold else "Likely to Stay" for p in probabilities]
    risk_levels = [risk_level(p) for p in probabilities]

    result = df.copy()
    result["churn_probability"] = probabilities
    result["prediction"] = predictions
    result["risk_level"] = risk_levels
    return result
