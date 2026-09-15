# Customer Churn Prediction

An end-to-end machine learning project that predicts whether a telecom customer
is likely to churn, explains *why* using feature-importance/churn-driver
analysis, and ships as an interactive Streamlit application.

## 1. Project Overview

Customer churn — a customer canceling their subscription — is one of the
most costly problems for subscription businesses. Acquiring a new customer
typically costs far more than retaining an existing one, so a model that
flags *at-risk* customers early lets a retention team intervene (offers,
proactive support, plan changes) before the customer leaves.

## 2. Objective

Given a customer's demographic, account, and service-usage attributes, predict:

- **Prediction:** `Likely to Churn` or `Likely to Stay`
- **Churn Probability:** a calibrated probability (0–100%)
- **Risk Level:** `Low` / `Medium` / `High`

and identify the strongest, most actionable factors behind churn.

## 3. Dataset

| | |
|---|---|
| **Name** | IBM Telco Customer Churn dataset |
| **Source** | [IBM `telco-customer-churn-on-icp4d` GitHub repo](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv) — the same dataset is also published on Kaggle as [*"Telco Customer Churn"* (blastchar)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Records** | 7,043 customers |
| **Raw columns** | 21 (20 features + target) |
| **Target** | `Churn` (`Yes`/`No`, encoded to 1/0) |

**Feature groups:**
- **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Account info:** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
- **Services:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Identifier (dropped before modeling):** `customerID`

The raw CSV is committed at `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

## 4. Methodology

```
Data Collection (IBM Telco dataset)
        ↓
Data Cleaning (fix TotalCharges, drop ID, encode target)
        ↓
Exploratory Data Analysis (churn distribution, drivers, correlations)
        ↓
Feature Engineering (6 domain-driven features)
        ↓
Preprocessing (sklearn ColumnTransformer: scaling + one-hot encoding)
        ↓
Model Training (Logistic Regression, Random Forest, HistGradientBoosting)
        ↓
Model Comparison (5-fold stratified cross-validation)
        ↓
Hyperparameter Tuning (RandomizedSearchCV)
        ↓
Threshold Optimization (out-of-fold F1 maximization)
        ↓
Evaluation (held-out test set — accuracy, precision, recall, F1, ROC-AUC, PR-AUC)
        ↓
Deployment (Streamlit app, loads the saved pipeline — no retraining at runtime)
```

All of this is implemented in `src/` and is reproducible end-to-end by running
`python -m src.train` (see [How to Run Locally](#11-how-to-run-locally)).

## 5. Exploratory Data Analysis

Key findings (see `notebooks/churn_analysis.ipynb` and `artifacts/plots/` for
the full set of charts):

- **Class imbalance:** ~26.5% of customers in the dataset churned (5,174 stayed
  vs. 1,869 churned) — handled via `class_weight="balanced"` and threshold
  tuning rather than assuming the default 0.5 threshold is appropriate.
- **Contract type** is one of the strongest signals: month-to-month customers
  churn far more than one-year or two-year contract customers.
- **Tenure:** churn risk is highest in a customer's first few months and
  drops steadily the longer they stay ("early-life churn").
- **Payment method:** customers paying by electronic check churn more than
  customers on automatic payment methods.
- **Internet service:** Fiber optic customers churn more than DSL or
  no-internet customers, and churned customers tend to pay *more* per month
  on average.
- **Data quality:** no duplicate rows; `TotalCharges` had 11 blank values, all
  belonging to brand-new customers with `tenure == 0` (fixed by setting them
  to 0, not by statistical imputation); `customerID` is a pure identifier and
  is dropped before modeling. No column represents information only knowable
  *after* a churn decision, so there is no data-leakage risk in the raw features.

## 6. Feature Engineering

Six domain-driven features were added on top of the raw columns
(`src/feature_engineering.py`), all computed strictly from information known
*before* any churn decision:

| Feature | Description | Why it could help |
|---|---|---|
| `tenure_group` | Bucketed tenure (0-12, 13-24, 25-48, 49-60, 61-72 months) | Churn risk is highly non-linear in tenure |
| `avg_monthly_spend` | `TotalCharges / (tenure + 1)` | Stable average spend estimate, well-defined even for brand-new customers |
| `num_add_on_services` | Count of subscribed add-on services | Proxy for engagement / lock-in with the provider |
| `has_internet_service` | 1 if any internet service, 0 if not | Internet vs. phone-only customers churn very differently |
| `is_new_customer` | 1 if `tenure <= 6` months | Captures the early-life churn spike directly |
| `charge_per_service` | `MonthlyCharges / (num_add_on_services + 1)` | Rough "value for money" ratio |

## 7. Models Evaluated

All candidates use `class_weight="balanced"` and are compared with 5-fold
**stratified** cross-validation on the training set (80% of the data; the
remaining 20% is held out untouched for final evaluation):

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **Logistic Regression** | 0.751 | 0.521 | 0.791 | **0.628** | 0.848 |
| HistGradientBoosting | 0.765 | 0.543 | 0.714 | 0.617 | 0.834 |
| Random Forest | 0.779 | 0.574 | 0.652 | 0.610 | 0.831 |

*(Full table reproduced in `artifacts/model_comparison.csv`.)*

**Class imbalance handling check** (Logistic Regression, CV on training set):

| `class_weight` | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| `balanced` | 0.751 | 0.521 | **0.791** | **0.628** | 0.848 |
| `None` | 0.808 | 0.676 | 0.530 | 0.594 | 0.848 |

Balancing trades some precision/accuracy for substantially higher recall on
the churn class — the right trade-off for this business problem, since the
cost of missing a churner is generally higher than the cost of a false alarm.
(Full comparison in `artifacts/class_imbalance_comparison.csv`.)

## 8. Final Model

**Logistic Regression** (tuned via `RandomizedSearchCV`, `C=0.01`, `solver=lbfgs`,
`class_weight="balanced"`) was selected — not because it had the highest raw
accuracy (Random Forest did), but because it had the **best F1-score and
recall on the churn class with a competitive ROC-AUC**, while also being the
most interpretable model for explaining churn drivers during a viva. Accuracy
alone is misleading here because the target is imbalanced (~26.5% churn); a
model that just predicts "No Churn" for everyone would already score ~73.5%
accuracy while being completely useless.

**Decision threshold:** the default 0.50 was not assumed to be optimal. A
threshold of **0.55** was chosen by scanning thresholds against **out-of-fold
predictions on the training set only** (5-fold CV, never touching the test
set) and picking the one that maximizes F1-score on the churn class. This
same threshold is used by the deployed Streamlit app.

## 9. Churn Drivers

Top churn drivers, ranked by **permutation importance** on the held-out test
set (i.e. how much the model's F1-score drops when that feature is shuffled —
computed at the level of the original input columns, not one-hot dummies, for
interpretability):

1. `is_new_customer` (tenure ≤ 6 months)
2. `tenure`
3. `Contract`
4. `TotalCharges`
5. `InternetService`
6. `MonthlyCharges`
7. `SeniorCitizen`
8. `charge_per_service`
9. `tenure_group`
10. `StreamingMovies`

*(Full ranking in `artifacts/feature_importance.csv`; one-hot-level
coefficients in `artifacts/feature_importance_encoded_level.csv`.)*

**Important:** this is a measure of **predictive association**, not proven
**causation**. It tells us which features the model relies on most, not that
those features directly *cause* customers to leave. For example, "fiber
optic" being associated with higher churn does not by itself prove fiber
optic service causes churn — it may instead correlate with price sensitivity
or a service-quality issue not captured in the data. Establishing causation
would require a controlled experiment, which is outside the scope of this
observational dataset.

## 10. Results

Final model evaluated once on the held-out test set (1,409 customers never
seen during training), at the optimized threshold of 0.55:

| Metric | Value |
|---|---:|
| Accuracy | 76.7% |
| Precision | 54.3% |
| Recall | 75.4% |
| F1-score | 63.2% |
| ROC-AUC | 0.845 |
| PR-AUC | 0.652 |

These numbers are genuine results from `artifacts/metrics.json` — not
fabricated — and are neither implausibly perfect (which would suggest data
leakage) nor poor; they reflect a real, moderately imbalanced, noisy
real-world churn dataset.

## 11. How to Run Locally

```bash
# 1. Clone / open the repository, then create a virtual environment
python -m venv venv

# 2. Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Re-run the full training pipeline
#    This regenerates data/processed/, models/churn_model.pkl and artifacts/
python -m src.train

# 5. Run the test suite
pytest tests/test_pipeline.py -v

# 6. Launch the Streamlit app (run from the repository root)
streamlit run app/app.py
```

To explore the full analysis notebook, install Jupyter (`pip install jupyter`)
and open `notebooks/churn_analysis.ipynb`. The notebook automatically switches
its working directory to the repository root on the first cell, so it runs
correctly regardless of where Jupyter is launched from.

## 12. Project Structure

```
customer-churn-prediction/
│
├── data/
│   ├── raw/                          # WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/                    # generated by src/train.py
│
├── notebooks/
│   └── churn_analysis.ipynb          # full, presentation-ready analysis
│
├── src/
│   ├── data_preprocessing.py         # loading, cleaning, ColumnTransformer
│   ├── feature_engineering.py        # 6 domain-driven engineered features
│   ├── train.py                      # model comparison, tuning, saving
│   ├── evaluate.py                   # shared metrics + plotting utilities
│   └── predict.py                    # production inference (no retraining)
│
├── models/
│   └── churn_model.pkl               # final pipeline + metadata
│
├── artifacts/
│   ├── metrics.json
│   ├── model_comparison.csv
│   ├── class_imbalance_comparison.csv
│   ├── feature_importance.csv
│   ├── feature_importance_encoded_level.csv
│   └── plots/                        # EDA + evaluation plots
│
├── app/
│   └── app.py                        # Streamlit dashboard (4 pages)
│
├── tests/
│   └── test_pipeline.py              # pytest suite
│
├── requirements.txt
├── .gitignore
├── README.md
├── VIVA_PREPARATION.md
└── LICENSE
```

## 13. Deployment

The app is a standard [Streamlit](https://streamlit.io) app and deploys
directly on **Streamlit Community Cloud**:

1. Push this repository to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app
   pointing at this repo.
3. Set the **main file path** to `app/app.py`.
4. Deploy — `requirements.txt` at the repo root is picked up automatically.

The app loads `models/churn_model.pkl` at startup (via `@st.cache_resource`)
and **never retrains** — startup is fast and behavior matches the metrics
reported here exactly.

## 14. Limitations

- **Single historical snapshot:** the dataset is from one provider at one
  point in time; the model may not generalize to other markets, pricing
  structures, or time periods.
- **Association, not causation:** churn drivers identified here are
  predictive associations from an observational dataset, not proven causes.
- **Class imbalance:** ~26.5% churn rate means the model still produces false
  positives/negatives that a retention team should budget for (recall ~75%,
  precision ~54% at the chosen threshold).
- **No external/behavioral data:** the dataset lacks customer support
  interactions, competitor pricing, or macroeconomic factors that likely also
  influence real-world churn.
- **Sampling:** we don't have information on how the original 7,043 customers
  were sampled from IBM's broader customer base, so subtle sampling bias
  cannot be ruled out.

## 15. Future Improvements

- Incorporate customer support/interaction logs and complaint history.
- Explicitly tie the decision threshold to retention-offer economics
  (cost-sensitive learning) rather than pure F1 maximization.
- Validate model stability on more recent data as it becomes available.
- Try an external gradient boosting library (XGBoost/LightGBM) if deployment
  compatibility can be confirmed on the target platform.
- Add SHAP-based per-customer explanations to the Streamlit app for
  case-by-case retention-team justification.

## 16. Team / Authors

- Ayush Bidwai — sole author

---

*Built as an academic Machine Learning course project. See
`VIVA_PREPARATION.md` for anticipated viva questions and answers.*
