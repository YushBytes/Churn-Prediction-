# Viva Preparation

Concise answers based on the *actual* implementation in this repository.

### 1. What is customer churn?
Customer churn is when a customer stops using a company's product or service
— here, a telecom customer canceling their subscription. It's captured by
the `Churn` column (`Yes`/`No`) in the dataset.

### 2. Why did you select this dataset?
The IBM Telco Customer Churn dataset is a well-known, legitimately sourced,
publicly available dataset (IBM's own GitHub, also on Kaggle) with a clear
binary churn target, a good mix of demographic/account/service features, and
enough rows (7,043) for genuine train/test evaluation — see README section 3.

### 3. What is the target variable?
`Churn`, originally `Yes`/`No`, encoded to `1`/`0` in
`src/data_preprocessing.py::encode_target`.

### 4. What preprocessing did you perform?
Dropping duplicates, converting `TotalCharges` from text to numeric, filling
its 11 blank values with 0 (all belong to brand-new customers with
`tenure == 0`), dropping the `customerID` identifier, and encoding the
target. Implemented in `src/data_preprocessing.py::clean_data`.

### 5. How did you handle missing values?
`TotalCharges` had 11 rows stored as blank strings. We verified they all had
`tenure == 0` (customers who hadn't been billed yet) and set them to `0.0`
— a domain-justified fix, not a generic statistical imputation.

### 6. What is feature engineering?
Creating new input features from existing raw data to expose patterns more
directly to the model — for example, turning raw tenure into a bucketed
`tenure_group` so a linear model can capture its non-linear relationship
with churn.

### 7. What features did you engineer and why?
Six features in `src/feature_engineering.py`: `tenure_group`,
`avg_monthly_spend`, `num_add_on_services`, `has_internet_service`,
`is_new_customer`, `charge_per_service` — see README section 6 for the
rationale behind each.

### 8. Why did you select these ML models?
Logistic Regression (interpretable, coefficients directly usable for churn
explanation), Random Forest (captures non-linear feature interactions), and
HistGradientBoosting (a strong sklearn-native boosting model that avoids the
deployment-compatibility risk of external boosting libraries like
XGBoost/LightGBM on a platform like Streamlit Cloud).

### 9. Why is accuracy alone insufficient?
The target is imbalanced (~26.5% churn). A model that always predicts "No
Churn" would score ~73.5% accuracy while being useless for the actual
business goal of catching churners. We optimized for F1-score/recall on the
churn class instead, and reported ROC-AUC/PR-AUC alongside accuracy.

### 10. What is precision?
Of all customers the model predicted would churn, the fraction that actually
did. Our final model: 54.3% at the chosen threshold.

### 11. What is recall?
Of all customers who actually churned, the fraction the model correctly
flagged. Our final model: 75.4% — prioritized because missing a real churner
is usually costlier than a false alarm.

### 12. What is F1-score?
The harmonic mean of precision and recall, balancing both in one number.
Used as the primary model-selection and threshold-tuning metric here (63.2%
on the held-out test set).

### 13. What is ROC-AUC?
The probability that the model ranks a randomly chosen churner higher than a
randomly chosen non-churner, across all thresholds. Our final model: 0.845 —
well above the 0.5 chance level and not suspiciously close to 1.0 (which
would suggest leakage).

### 14. What is overfitting?
When a model learns patterns specific to the training data (including noise)
rather than the general relationship, so it performs much better on training
data than on new, unseen data.

### 15. How did you prevent overfitting?
Stratified train/test split with a held-out test set touched only once at
the very end; 5-fold cross-validation during model comparison and tuning
instead of a single train/validation split; regularized models
(`LogisticRegression` with tuned `C`); and threshold selection done on
out-of-fold training predictions, never on the test set.

### 16. What is cross-validation?
Splitting the training data into k folds, training on k−1 folds and
validating on the remaining fold, k times, then averaging the results — gives
a more reliable performance estimate than a single split. We used 5-fold
**stratified** CV (`StratifiedKFold`) so each fold preserves the ~26.5% churn
rate.

### 17. What is class imbalance?
When one class is much rarer than another — here, churners (~26.5%) are the
minority class. Models trained naively on imbalanced data tend to favor the
majority class.

### 18. How did you handle class imbalance?
Used `class_weight="balanced"` (penalizes misclassifying the minority class
more), and separately verified in `artifacts/class_imbalance_comparison.csv`
that this actually improves recall/F1 versus no balancing (at some precision
cost) before keeping it in the final model. Also tuned the decision threshold
instead of assuming 0.5, further correcting for the imbalance.

### 19. How did you prevent data leakage?
(1) Dropped `customerID` (a pure identifier with no predictive meaning).
(2) Checked every remaining column is information known *before* a churn
decision — no post-churn fields exist in this dataset. (3) All preprocessing
statistics (scaling, one-hot categories) are fit only on the training fold,
inside a single sklearn `Pipeline`/`ColumnTransformer`, never on the test
set. (4) The decision threshold was chosen from training-set out-of-fold
predictions, not the test set. (5) `tests/test_pipeline.py` includes an
explicit `test_no_leakage_columns` check.

### 20. Why was the final model selected?
Logistic Regression had the best F1-score and recall on the churn class in
cross-validation with a competitive ROC-AUC (see README section 7), and is
also the most interpretable of the three candidates — valuable both for the
"churn driver" business requirement and for explaining the model in this
viva. It was not selected simply for having the highest accuracy (Random
Forest had higher CV accuracy but lower F1/recall).

### 21. What are the strongest churn drivers?
By permutation importance on the held-out test set: being a new customer
(`is_new_customer`), `tenure`, `Contract` type, `TotalCharges`,
`InternetService` type, and `MonthlyCharges` — consistent with the EDA
finding that month-to-month, low-tenure, higher-paying fiber customers churn
most. See README section 9 for the full list and the causation caveat.

### 22. How does the deployed application work?
`app/app.py` is a Streamlit app that loads the already-trained pipeline from
`models/churn_model.pkl` once at startup (`@st.cache_resource`) and never
retrains. A user fills in a customer's details on the "Churn Prediction"
page; `src/predict.py::predict_single` applies the exact same feature
engineering and preprocessing used during training, then returns a
prediction, probability, and risk level using the same tuned threshold
(0.55) reported in the README.

### 23. What are the limitations?
Single historical snapshot from one provider (limited generalizability),
predictive associations rather than proven causal drivers, moderate
precision/recall trade-off from class imbalance, and no external data such as
support interactions or competitor pricing. See README section 14 for the
full list.

### 24. How could this project be improved?
Incorporate customer support/interaction data, tie the decision threshold to
actual retention-offer costs (cost-sensitive learning), validate on more
recent data, evaluate an external boosting library if deployment
compatibility allows, and add per-customer SHAP explanations to the app. See
README section 15.
