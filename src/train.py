"""
End-to-end training script for the Customer Churn Prediction project.

Run from the repository root:

    python -m src.train

What this script does
----------------------
1. Loads and cleans the raw Telco churn dataset.
2. Engineers domain-relevant features.
3. Splits into train/test (stratified, held-out test set used ONLY for the
   final, unbiased evaluation reported in the README/artifacts).
4. Builds a leakage-safe preprocessing + model Pipeline for each candidate
   algorithm and compares them with stratified cross-validation on the
   training set only.
5. Checks whether class-imbalance handling (class_weight='balanced') helps.
6. Tunes the strongest candidate with RandomizedSearchCV.
7. Picks a decision threshold using out-of-fold training predictions
   (never the test set), fits the final pipeline on the full training set,
   and evaluates once on the untouched test set.
8. Saves the final pipeline + metadata to models/churn_model.pkl and writes
   all evaluation/EDA artifacts to artifacts/.
"""
from __future__ import annotations

import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.data_preprocessing import (
    TARGET_COLUMN,
    build_preprocessor,
    get_feature_target_split,
    load_and_clean,
)
from src.evaluate import (
    compute_metrics,
    find_best_threshold,
    plot_categorical_vs_churn,
    plot_churn_distribution,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_numeric_vs_churn,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from src.feature_engineering import ENGINEERED_FEATURE_DESCRIPTIONS, add_engineered_features

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
ARTIFACTS_DIR = "artifacts"
PLOTS_DIR = f"{ARTIFACTS_DIR}/plots"
MODELS_DIR = "models"
PROCESSED_DIR = "data/processed"
CV_FOLDS = 5
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def build_candidate_models() -> dict:
    """Candidate algorithms, all with class_weight='balanced' since the
    churn target is imbalanced (~26.5% positive class)."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
    }


def compare_models(X_train, y_train, preprocessor) -> pd.DataFrame:
    """5-fold stratified CV comparison of the candidate algorithms."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in build_candidate_models().items():
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", model)])
        results = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
        rows.append(
            {
                "model": name,
                "accuracy": results["test_accuracy"].mean(),
                "precision": results["test_precision"].mean(),
                "recall": results["test_recall"].mean(),
                "f1_score": results["test_f1"].mean(),
                "roc_auc": results["test_roc_auc"].mean(),
            }
        )
        print(f"  {name}: {rows[-1]}")
    return pd.DataFrame(rows).sort_values("f1_score", ascending=False).reset_index(drop=True)


def compare_imbalance_handling(X_train, y_train, preprocessor, model_name: str) -> pd.DataFrame:
    """Compare class_weight=None vs class_weight='balanced' for the best model."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    base_models = build_candidate_models()
    balanced_model = base_models[model_name]
    unbalanced_model = balanced_model.__class__(**{**balanced_model.get_params(), "class_weight": None})

    rows = []
    for label, model in [("balanced", balanced_model), ("none", unbalanced_model)]:
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", model)])
        results = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
        rows.append(
            {
                "class_weight": label,
                "accuracy": results["test_accuracy"].mean(),
                "precision": results["test_precision"].mean(),
                "recall": results["test_recall"].mean(),
                "f1_score": results["test_f1"].mean(),
                "roc_auc": results["test_roc_auc"].mean(),
            }
        )
    return pd.DataFrame(rows)


def get_param_grid(model_name: str) -> dict:
    if model_name == "LogisticRegression":
        return {
            "classifier__C": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            "classifier__solver": ["lbfgs", "liblinear"],
        }
    if model_name == "RandomForest":
        return {
            "classifier__n_estimators": [200, 300, 400, 500],
            "classifier__max_depth": [4, 6, 8, 10, None],
            "classifier__min_samples_leaf": [1, 2, 4, 8],
            "classifier__max_features": ["sqrt", "log2"],
        }
    if model_name == "HistGradientBoosting":
        return {
            "classifier__learning_rate": [0.03, 0.05, 0.1, 0.2],
            "classifier__max_iter": [100, 150, 200, 300],
            "classifier__max_depth": [3, 4, 6, None],
            "classifier__min_samples_leaf": [10, 20, 30],
            "classifier__l2_regularization": [0.0, 0.1, 0.5, 1.0],
        }
    raise ValueError(f"Unknown model: {model_name}")


def tune_model(X_train, y_train, preprocessor, model_name: str) -> Pipeline:
    model = build_candidate_models()[model_name]
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", model)])
    param_grid = get_param_grid(model_name)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_grid,
        n_iter=25,
        scoring="f1",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    print(f"  Best params for {model_name}: {search.best_params_}")
    print(f"  Best CV F1: {search.best_score_:.4f}")
    return search.best_estimator_


def get_feature_importance(pipeline: Pipeline, model_name: str, X_test, y_test) -> pd.DataFrame:
    """Churn-driver analysis at the level of the ORIGINAL (raw) input columns.

    Permutation importance is computed on the full pipeline using the raw
    test features (e.g. "Contract", "tenure"), which is what
    `permutation_importance` naturally measures (it shuffles one raw input
    column at a time and observes the drop in F1-score). This is far more
    interpretable for a business audience than one-hot encoded dummy
    columns like "Contract_Month-to-month", so it is used as the primary
    "Top Churn Drivers" ranking.

    As a secondary, model-native cross-check, we also report the strongest
    one-hot encoded coefficients/feature_importances_ (useful during the
    viva to explain *which category* of a feature drives churn).
    """
    perm_result = permutation_importance(
        pipeline, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, scoring="f1", n_jobs=-1
    )
    raw_importance_df = pd.DataFrame(
        {
            "feature": X_test.columns.tolist(),
            "importance": perm_result.importances_mean,
            "importance_std": perm_result.importances_std,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    # Secondary, model-native view at the one-hot encoded level.
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    encoded_feature_names = preprocessor.get_feature_names_out()

    native_importance = None
    if hasattr(classifier, "feature_importances_"):
        native_importance = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        native_importance = np.abs(classifier.coef_[0])

    if native_importance is not None:
        native_df = pd.DataFrame(
            {"encoded_feature": encoded_feature_names, "native_importance": native_importance}
        ).sort_values("native_importance", ascending=False).reset_index(drop=True)
        native_df.to_csv(f"{ARTIFACTS_DIR}/feature_importance_encoded_level.csv", index=False)

    return raw_importance_df


def main():
    import os

    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("=" * 70)
    print("STEP 1: Load and clean data")
    print("=" * 70)
    df = load_and_clean()
    print(f"Cleaned dataset shape: {df.shape}")

    print("\n" + "=" * 70)
    print("STEP 2: Feature engineering")
    print("=" * 70)
    df = add_engineered_features(df)
    for name, desc in ENGINEERED_FEATURE_DESCRIPTIONS.items():
        print(f"  + {name}: {desc}")

    df.to_csv(f"{PROCESSED_DIR}/telco_churn_processed.csv", index=False)

    print("\n" + "=" * 70)
    print("STEP 3: EDA plots -> artifacts/plots/")
    print("=" * 70)
    plot_churn_distribution(df, TARGET_COLUMN, f"{PLOTS_DIR}/churn_distribution.png")
    plot_categorical_vs_churn(df, "Contract", TARGET_COLUMN, f"{PLOTS_DIR}/churn_by_contract.png")
    plot_categorical_vs_churn(df, "PaymentMethod", TARGET_COLUMN, f"{PLOTS_DIR}/churn_by_payment_method.png")
    plot_categorical_vs_churn(df, "InternetService", TARGET_COLUMN, f"{PLOTS_DIR}/churn_by_internet_service.png")
    plot_numeric_vs_churn(df, "tenure", TARGET_COLUMN, f"{PLOTS_DIR}/tenure_vs_churn.png")
    plot_numeric_vs_churn(df, "MonthlyCharges", TARGET_COLUMN, f"{PLOTS_DIR}/monthly_charges_vs_churn.png")
    plot_correlation_heatmap(
        df,
        ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend", "num_add_on_services", TARGET_COLUMN],
        f"{PLOTS_DIR}/correlation_heatmap.png",
    )
    print("  Saved EDA plots.")

    print("\n" + "=" * 70)
    print("STEP 4: Train/test split (stratified, 80/20)")
    print("=" * 70)
    X, y = get_feature_target_split(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train churn rate: {y_train.mean():.3f}, Test churn rate: {y_test.mean():.3f}")

    preprocessor = build_preprocessor(X_train)

    print("\n" + "=" * 70)
    print("STEP 5: Model comparison (5-fold stratified CV on training set)")
    print("=" * 70)
    comparison_df = compare_models(X_train, y_train, preprocessor)
    print(comparison_df)
    comparison_df.to_csv(f"{ARTIFACTS_DIR}/model_comparison.csv", index=False)

    best_model_name = comparison_df.iloc[0]["model"]
    print(f"\nBest candidate by CV F1-score: {best_model_name}")

    print("\n" + "=" * 70)
    print("STEP 6: Class imbalance handling check")
    print("=" * 70)
    imbalance_df = compare_imbalance_handling(X_train, y_train, preprocessor, best_model_name)
    print(imbalance_df)
    imbalance_df.to_csv(f"{ARTIFACTS_DIR}/class_imbalance_comparison.csv", index=False)

    print("\n" + "=" * 70)
    print(f"STEP 7: Hyperparameter tuning for {best_model_name} (RandomizedSearchCV)")
    print("=" * 70)
    t0 = time.time()
    tuned_pipeline = tune_model(X_train, y_train, preprocessor, best_model_name)
    print(f"  Tuning took {time.time() - t0:.1f}s")

    print("\n" + "=" * 70)
    print("STEP 8: Threshold optimization (out-of-fold predictions, train set only)")
    print("=" * 70)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_probs = cross_val_predict(
        tuned_pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    best_threshold = find_best_threshold(y_train, oof_probs)
    print(f"  Selected decision threshold: {best_threshold:.2f} (chosen via CV, not the test set)")

    print("\n" + "=" * 70)
    print("STEP 9: Fit final pipeline on full training set, evaluate on held-out test set")
    print("=" * 70)
    tuned_pipeline.fit(X_train, y_train)
    test_probs = tuned_pipeline.predict_proba(X_test)[:, 1]

    metrics_default = compute_metrics(y_test, test_probs, threshold=0.5)
    metrics_optimized = compute_metrics(y_test, test_probs, threshold=best_threshold)
    print(f"  Metrics @ threshold=0.50: {metrics_default}")
    print(f"  Metrics @ threshold={best_threshold:.2f}: {metrics_optimized}")

    plot_confusion_matrix(y_test, test_probs, best_threshold, f"{PLOTS_DIR}/confusion_matrix.png")
    plot_roc_curve(y_test, test_probs, f"{PLOTS_DIR}/roc_curve.png")
    plot_precision_recall_curve(y_test, test_probs, f"{PLOTS_DIR}/precision_recall_curve.png")

    print("\n" + "=" * 70)
    print("STEP 10: Feature importance / churn driver analysis")
    print("=" * 70)
    importance_df = get_feature_importance(tuned_pipeline, best_model_name, X_test, y_test)
    importance_df.to_csv(f"{ARTIFACTS_DIR}/feature_importance.csv", index=False)
    plot_feature_importance(importance_df, f"{PLOTS_DIR}/feature_importance.png")
    print(importance_df.head(10))

    print("\n" + "=" * 70)
    print("STEP 11: Save final production pipeline + metadata")
    print("=" * 70)
    artifact_bundle = {
        "pipeline": tuned_pipeline,
        "model_name": best_model_name,
        "threshold": best_threshold,
        "feature_columns": X_train.columns.tolist(),
        "metrics_default_threshold": metrics_default,
        "metrics_optimized_threshold": metrics_optimized,
        "engineered_features": ENGINEERED_FEATURE_DESCRIPTIONS,
        "random_state": RANDOM_STATE,
    }
    joblib.dump(artifact_bundle, f"{MODELS_DIR}/churn_model.pkl")
    print(f"  Saved pipeline to {MODELS_DIR}/churn_model.pkl")

    metrics_output = {
        "model_name": best_model_name,
        "threshold": best_threshold,
        "test_set_size": int(len(y_test)),
        "train_set_size": int(len(y_train)),
        "metrics_at_default_threshold_0.5": metrics_default,
        "metrics_at_optimized_threshold": metrics_optimized,
    }
    with open(f"{ARTIFACTS_DIR}/metrics.json", "w") as f:
        json.dump(metrics_output, f, indent=2)
    print(f"  Saved metrics to {ARTIFACTS_DIR}/metrics.json")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Final model: {best_model_name}")
    print(f"Final threshold: {best_threshold:.2f}")
    print(f"Test ROC-AUC: {metrics_optimized['roc_auc']:.4f}")
    print(f"Test F1 (churn class): {metrics_optimized['f1_score']:.4f}")
    print(f"Test Recall (churn class): {metrics_optimized['recall']:.4f}")


if __name__ == "__main__":
    main()
