"""
Basic pipeline tests. Run from the repository root:

    pytest tests/test_pipeline.py -v

These tests verify the parts of the project that are easy to silently
break: data cleaning correctness, absence of data leakage, preprocessing
reproducibility, and that the saved production model can load and predict
independently of the training script.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessing import (
    ID_COLUMN,
    TARGET_COLUMN,
    build_preprocessor,
    get_feature_target_split,
    load_and_clean,
)
from src.feature_engineering import add_engineered_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "churn_model.pkl")


@pytest.fixture(scope="module")
def cleaned_df():
    return load_and_clean()


def test_data_loads_with_expected_shape(cleaned_df):
    assert cleaned_df.shape[0] == 7043
    assert ID_COLUMN not in cleaned_df.columns


def test_no_missing_values_after_cleaning(cleaned_df):
    assert cleaned_df.isnull().sum().sum() == 0


def test_target_is_binary(cleaned_df):
    assert set(cleaned_df[TARGET_COLUMN].unique()) == {0, 1}


def test_total_charges_is_numeric(cleaned_df):
    assert pd.api.types.is_numeric_dtype(cleaned_df["TotalCharges"])


def test_engineered_features_added(cleaned_df):
    df = add_engineered_features(cleaned_df)
    for col in [
        "tenure_group",
        "avg_monthly_spend",
        "num_add_on_services",
        "has_internet_service",
        "is_new_customer",
        "charge_per_service",
    ]:
        assert col in df.columns


def test_no_leakage_columns(cleaned_df):
    """None of the raw feature columns should be information only available
    after a churn decision (e.g. no 'churn_date', 'churn_reason', etc.)."""
    suspicious_keywords = ["churn_date", "churn_reason", "cancel", "exit_date"]
    feature_cols = [c for c in cleaned_df.columns if c != TARGET_COLUMN]
    for col in feature_cols:
        assert not any(k in col.lower() for k in suspicious_keywords)


def test_preprocessor_builds_and_transforms(cleaned_df):
    df = add_engineered_features(cleaned_df)
    X, y = get_feature_target_split(df)
    preprocessor = build_preprocessor(X)
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == X.shape[0]


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Model not trained yet; run `python -m src.train` first.")
def test_saved_model_predicts_independently():
    """The saved pipeline must load and predict without importing anything
    from the training script (i.e. it is truly production-ready)."""
    import joblib

    bundle = joblib.load(MODEL_PATH)
    assert "pipeline" in bundle
    assert "threshold" in bundle
    assert "feature_columns" in bundle

    from src.predict import predict_single

    sample = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": 170.0,
    }
    result = predict_single(sample, bundle)
    assert result["prediction"] in {"Likely to Churn", "Likely to Stay"}
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["risk_level"] in {"Low", "Medium", "High"}


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Model not trained yet; run `python -m src.train` first.")
def test_model_metrics_are_reasonable():
    """Sanity check: metrics must exist and not indicate an impossible
    (e.g. 100% accuracy) or leaking model."""
    import json

    metrics_path = os.path.join(os.path.dirname(__file__), "..", "artifacts", "metrics.json")
    with open(metrics_path) as f:
        metrics = json.load(f)

    roc_auc = metrics["metrics_at_optimized_threshold"]["roc_auc"]
    accuracy = metrics["metrics_at_optimized_threshold"]["accuracy"]
    assert 0.5 < roc_auc < 0.99, "ROC-AUC should be well above chance but not suspiciously perfect"
    assert 0.5 < accuracy < 0.99, "Accuracy should not be suspiciously perfect (possible leakage)"
