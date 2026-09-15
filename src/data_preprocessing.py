"""
Data loading and cleaning utilities for the Telco Customer Churn dataset.

Design notes
------------
- All cleaning here is deterministic and leakage-free: nothing depends on the
  target column, and nothing depends on train/test split statistics (that is
  handled later inside the sklearn ColumnTransformer, which is fit only on
  the training fold).
- `customerID` is a unique identifier with zero predictive value and is
  dropped before modeling to avoid the model memorizing IDs.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"

# Columns that are genuinely numeric in the raw file.
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Telco churn CSV exactly as downloaded from the source."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic, leakage-free cleaning steps.

    Steps performed:
    1. Drop exact duplicate rows (none exist in this dataset, but this keeps
       the pipeline robust if the source file changes).
    2. Fix `TotalCharges`: it is stored as text and contains 11 blank
       strings, all belonging to brand-new customers with `tenure == 0`.
       These customers have not been billed yet, so the correct value is 0,
       not a statistical imputation.
    3. Strip whitespace from string columns (defensive cleaning).
    4. Drop the `customerID` identifier column.
    """
    df = df.copy()

    df = df.drop_duplicates()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Rows with missing TotalCharges all have tenure == 0 (verified during EDA).
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    string_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in string_cols:
        df[col] = df[col].astype(str).str.strip()

    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the target column Yes/No -> 1/0."""
    df = df.copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0}).astype(int)
    return df


def load_and_clean(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Convenience wrapper: load raw data, clean it, and encode the target."""
    df = load_raw_data(path)
    df = clean_data(df)
    df = encode_target(df)
    return df


def get_feature_target_split(df: pd.DataFrame):
    """Split a cleaned dataframe into X (features) and y (target)."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a reproducible sklearn ColumnTransformer for the feature set.

    Numeric columns are scaled (StandardScaler) and categorical/text columns
    are one-hot encoded (unknown categories at inference time are ignored
    rather than raising, which keeps the deployed app robust). This
    transformer is meant to be the first step of a full sklearn Pipeline so
    that IDENTICAL preprocessing is applied during training and inference.
    """
    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
                categorical_features,
            ),
        ]
    )
    return preprocessor
