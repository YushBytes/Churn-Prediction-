"""
Streamlit dashboard for the Customer Churn Prediction project.

Run from the repository root:

    streamlit run app/app.py

This app ONLY loads the pre-trained pipeline saved at models/churn_model.pkl
(via src.predict) -- it never retrains the model, so startup is instant and
behavior in production matches the notebook/training results exactly.
"""
import json
import os
import sys

import pandas as pd
import streamlit as st

# Make the repository root importable regardless of the working directory
# Streamlit is launched from, so `streamlit run app/app.py` always works
# from the repo root as instructed in the README.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import load_model_bundle, predict_single  # noqa: E402

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACTS_DIR = os.path.join(REPO_ROOT, "artifacts")
PLOTS_DIR = os.path.join(ARTIFACTS_DIR, "plots")
DATA_PATH = os.path.join(REPO_ROOT, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")


@st.cache_resource
def get_model_bundle():
    return load_model_bundle(os.path.join(REPO_ROOT, "models", "churn_model.pkl"))


@st.cache_data
def get_raw_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def get_metrics():
    with open(os.path.join(ARTIFACTS_DIR, "metrics.json")) as f:
        return json.load(f)


@st.cache_data
def get_feature_importance():
    return pd.read_csv(os.path.join(ARTIFACTS_DIR, "feature_importance.csv"))


def render_overview():
    st.title("📉 Customer Churn Prediction")
    st.markdown(
        """
        ### Business Problem
        Telecom providers lose significant revenue when customers **churn**
        (cancel their subscription). Acquiring a new customer is far more
        expensive than retaining an existing one, so being able to flag
        *likely-to-churn* customers early lets a retention team intervene
        (offers, outreach, plan changes) before the customer leaves.

        This dashboard uses a machine learning model trained on historical
        customer data to estimate each customer's probability of churning
        and to explain which factors matter most.
        """
    )

    df = get_raw_data()
    bundle = get_model_bundle()
    metrics = get_metrics()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers in dataset", f"{len(df):,}")
    col2.metric("Historical churn rate", f"{(df['Churn'] == 'Yes').mean():.1%}")
    col3.metric("Final model", bundle["model_name"])
    col4.metric("Test ROC-AUC", f"{metrics['metrics_at_optimized_threshold']['roc_auc']:.3f}")

    st.markdown("---")
    st.subheader("Dataset Summary")
    st.markdown(
        """
        - **Source:** [IBM Telco Customer Churn dataset]
          (https://github.com/IBM/telco-customer-churn-on-icp4d) —
          also published on Kaggle as *Telco Customer Churn* (blastchar).
        - **Records:** 7,043 customers, 20 raw features + target.
        - **Target:** `Churn` (Yes/No) — whether the customer left within
          the last month.
        - **Feature groups:** demographics (gender, senior citizen,
          partner, dependents), account info (tenure, contract, billing,
          payment method), and subscribed services (phone, internet,
          streaming, security add-ons).
        """
    )

    st.subheader("Model Performance Snapshot")
    m = metrics["metrics_at_optimized_threshold"]
    perf_cols = st.columns(5)
    perf_cols[0].metric("Accuracy", f"{m['accuracy']:.1%}")
    perf_cols[1].metric("Precision", f"{m['precision']:.1%}")
    perf_cols[2].metric("Recall", f"{m['recall']:.1%}")
    perf_cols[3].metric("F1-score", f"{m['f1_score']:.1%}")
    perf_cols[4].metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    st.caption(
        f"Evaluated on a held-out test set of {metrics['test_set_size']} customers "
        f"never seen during training, at the optimized decision threshold of "
        f"{metrics['threshold']:.2f}."
    )


def render_prediction_form():
    st.title("🔮 Churn Prediction")
    st.markdown("Enter a customer's details to estimate their churn risk.")

    bundle = get_model_bundle()

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Demographics**")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", ["No", "Yes"])
            dependents = st.selectbox("Has Dependents", ["No", "Yes"])

        with col2:
            st.markdown("**Account Info**")
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=1.0)
            total_charges = st.number_input(
                "Total Charges ($)", min_value=0.0, value=float(monthly_charges * max(tenure, 1)), step=1.0
            )

        with col3:
            st.markdown("**Services**")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        submitted = st.form_submit_button("Predict Churn Risk", type="primary")

    if submitted:
        customer = {
            "gender": gender,
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }
        result = predict_single(customer, bundle)

        st.markdown("---")
        st.subheader("Prediction Result")
        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            if result["prediction"] == "Likely to Churn":
                st.error(f"**{result['prediction']}**")
            else:
                st.success(f"**{result['prediction']}**")

        with res_col2:
            st.metric("Churn Probability", f"{result['churn_probability']:.1%}")

        with res_col3:
            risk_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
            st.metric("Risk Level", f"{risk_color[result['risk_level']]} {result['risk_level']}")

        st.progress(min(result["churn_probability"], 1.0))
        st.caption(
            f"Decision threshold used: {result['threshold_used']:.2f} "
            "(optimized via cross-validation, not the default 0.50 — see README for rationale)."
        )


def render_model_performance():
    st.title("📊 Model Performance")
    metrics = get_metrics()

    st.subheader("Final Model Metrics (held-out test set)")
    m = metrics["metrics_at_optimized_threshold"]
    df_metrics = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "PR-AUC"],
            "Value": [m["accuracy"], m["precision"], m["recall"], m["f1_score"], m["roc_auc"], m["pr_auc"]],
        }
    )
    df_metrics["Value"] = df_metrics["Value"].map(lambda v: f"{v:.4f}")
    st.table(df_metrics)

    comparison_path = os.path.join(ARTIFACTS_DIR, "model_comparison.csv")
    if os.path.exists(comparison_path):
        st.subheader("Model Comparison (5-fold cross-validation on training set)")
        st.dataframe(pd.read_csv(comparison_path), width="stretch")

    st.subheader("Diagnostic Plots")
    plot_col1, plot_col2 = st.columns(2)
    with plot_col1:
        st.image(os.path.join(PLOTS_DIR, "confusion_matrix.png"), caption="Confusion Matrix")
        st.image(os.path.join(PLOTS_DIR, "roc_curve.png"), caption="ROC Curve")
    with plot_col2:
        st.image(os.path.join(PLOTS_DIR, "precision_recall_curve.png"), caption="Precision-Recall Curve")


def render_churn_drivers():
    st.title("🧭 Churn Drivers")
    st.markdown(
        """
        The chart and table below rank input features by **permutation
        importance**: how much the model's F1-score drops on the test set
        when that feature's values are randomly shuffled. A larger drop
        means the model relies more heavily on that feature.

        > **Important:** this measures *predictive association*, not
        > proven causation. A feature can be a strong churn predictor
        > without directly causing customers to leave (e.g. it may
        > correlate with an underlying cause not captured in the data).
        """
    )

    importance_df = get_feature_importance()
    st.image(os.path.join(PLOTS_DIR, "feature_importance.png"), caption="Top Feature Importances")

    st.subheader("Top Churn Drivers")
    top10 = importance_df.head(10).reset_index(drop=True)
    top10.index = top10.index + 1
    st.table(top10[["feature", "importance"]])

    st.subheader("EDA: How These Factors Relate to Churn")
    eda_col1, eda_col2 = st.columns(2)
    with eda_col1:
        st.image(os.path.join(PLOTS_DIR, "churn_by_contract.png"))
        st.image(os.path.join(PLOTS_DIR, "tenure_vs_churn.png"))
    with eda_col2:
        st.image(os.path.join(PLOTS_DIR, "churn_by_internet_service.png"))
        st.image(os.path.join(PLOTS_DIR, "monthly_charges_vs_churn.png"))


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to", ["Overview", "Churn Prediction", "Model Performance", "Churn Drivers"]
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Academic ML project — Customer Churn Prediction. "
        "Model loaded from models/churn_model.pkl (no retraining at runtime)."
    )

    if page == "Overview":
        render_overview()
    elif page == "Churn Prediction":
        render_prediction_form()
    elif page == "Model Performance":
        render_model_performance()
    elif page == "Churn Drivers":
        render_churn_drivers()


if __name__ == "__main__":
    main()
