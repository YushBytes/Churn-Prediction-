"""
Domain-driven feature engineering for the Telco Customer Churn dataset.

Every engineered feature below is computed strictly from information a
telecom provider would know about a customer *before* any churn decision is
made (signup details, plan, usage, billing). None of them leak the target.

Engineered features
--------------------
1. tenure_group        : Bucketed tenure (0-12, 13-24, 25-48, 49-60, 61-72
                          months). Churn risk in this dataset is highly
                          non-linear in tenure (very high in month 1, then
                          drops sharply) — a linear model benefits from an
                          explicit bucket, and it also makes EDA/explanation
                          easier for a non-technical audience.
2. avg_monthly_spend    : TotalCharges / (tenure + 1). More robust than the
                          raw MonthlyCharges for customers whose current
                          monthly rate changed over their lifetime, and well
                          defined even for brand-new customers (tenure = 0).
3. num_add_on_services  : Count of subscribed add-on services (OnlineSecurity,
                          OnlineBackup, DeviceProtection, TechSupport,
                          StreamingTV, StreamingMovies, MultipleLines). A
                          simple proxy for how "locked in" / engaged a
                          customer is with the provider's ecosystem.
4. has_internet_service : Binary flag, 1 if the customer subscribes to any
                          internet service (DSL or Fiber optic), 0 otherwise.
                          Internet-only vs. phone-only customers churn very
                          differently in this dataset.
5. is_new_customer      : Binary flag, 1 if tenure <= 6 months. New customers
                          are known (from the EDA) to churn at a much higher
                          rate than tenured customers ("early-life churn").
6. charge_per_service    : MonthlyCharges / (num_add_on_services + 1). A rough
                          "value for money" ratio — customers paying a lot
                          for few services may perceive lower value.
"""
from __future__ import annotations

import pandas as pd

ADD_ON_SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "MultipleLines",
]

ENGINEERED_FEATURE_DESCRIPTIONS = {
    "tenure_group": "Bucketed customer tenure (0-12, 13-24, 25-48, 49-60, 61-72 months) "
    "to capture the strongly non-linear relationship between tenure and churn.",
    "avg_monthly_spend": "TotalCharges / (tenure + 1); a stable estimate of average monthly "
    "spend that also works for brand-new customers.",
    "num_add_on_services": "Count of subscribed add-on services; proxy for customer engagement "
    "/ lock-in with the provider's ecosystem.",
    "has_internet_service": "1 if the customer has any internet service (DSL/Fiber), 0 if not.",
    "is_new_customer": "1 if tenure <= 6 months; new customers show a much higher churn rate.",
    "charge_per_service": "MonthlyCharges / (num_add_on_services + 1); rough value-for-money ratio.",
}


def _tenure_group(tenure: pd.Series) -> pd.Series:
    bins = [-1, 12, 24, 48, 60, 72]
    labels = ["0-12", "13-24", "25-48", "49-60", "61-72"]
    return pd.cut(tenure, bins=bins, labels=labels)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with the engineered features appended."""
    df = df.copy()

    df["tenure_group"] = _tenure_group(df["tenure"]).astype(str)

    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)

    df["num_add_on_services"] = (
        df[ADD_ON_SERVICE_COLUMNS].eq("Yes").sum(axis=1)
    )

    df["has_internet_service"] = (df["InternetService"] != "No").astype(int)

    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)

    df["charge_per_service"] = df["MonthlyCharges"] / (df["num_add_on_services"] + 1)

    return df


def get_engineered_feature_names() -> list[str]:
    return list(ENGINEERED_FEATURE_DESCRIPTIONS.keys())
