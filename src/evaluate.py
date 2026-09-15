"""
Evaluation utilities shared by the training script and the Streamlit app.

Keeping metric computation and plotting in one place guarantees that the
numbers shown in the notebook, the artifacts/ folder, and the deployed app
are always computed the same way.
"""
from __future__ import annotations

# NOTE: the matplotlib backend is intentionally NOT forced here. src/train.py
# (a headless script) sets it to "Agg" before importing this module; the
# notebook instead activates its own inline backend via `%matplotlib inline`.
# Forcing a backend in this shared module would silently override whichever
# backend the importing context (script vs. notebook) actually wants.
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Compute the standard churn-classification metric set at a given threshold."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def find_best_threshold(y_true, y_prob) -> float:
    """Pick the probability threshold that maximizes F1 on the churn class.

    We scan thresholds from 0.05 to 0.95 rather than trusting the default of
    0.50, since with an imbalanced target (~26.5% churn) the default
    threshold tends to under-predict the minority (churn) class.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    best_threshold, best_f1 = 0.5, -1.0
    for t in thresholds:
        y_pred = (np.asarray(y_prob) >= t).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1, best_threshold = score, t
    return float(best_threshold)


def risk_level(probability: float) -> str:
    """Bucket a churn probability into a human-readable risk level."""
    if probability >= 0.66:
        return "High"
    if probability >= 0.33:
        return "Medium"
    return "Low"


def plot_confusion_matrix(y_true, y_prob, threshold: float, save_path: str) -> None:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(f"Confusion Matrix (threshold={threshold:.2f})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_true, y_prob, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax, name="Final model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_precision_recall_curve(y_true, y_prob, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax, name="Final model")
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# EDA plots (used by both train.py, to populate artifacts/plots, and the
# notebook, so the story told in the notebook matches the saved artifacts).
# ---------------------------------------------------------------------------


def plot_churn_distribution(df, target_col: str, save_path: str) -> None:
    counts = df[target_col].map({0: "No Churn", 1: "Churn"}).value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(counts.index, counts.values, color=["#4C72B0", "#DD8452"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 30, f"{v} ({v / counts.sum():.1%})", ha="center")
    ax.set_title("Churn Distribution")
    ax.set_ylabel("Number of customers")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_categorical_vs_churn(df, column: str, target_col: str, save_path: str) -> None:
    import pandas as pd

    ct = pd.crosstab(df[column], df[target_col].map({0: "No Churn", 1: "Churn"}), normalize="index")
    fig, ax = plt.subplots(figsize=(6, 4))
    ct.plot(kind="bar", stacked=True, ax=ax, color=["#4C72B0", "#DD8452"])
    ax.set_title(f"Churn Rate by {column}")
    ax.set_ylabel("Proportion of customers")
    ax.legend(title="")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_numeric_vs_churn(df, column: str, target_col: str, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    data_no = df.loc[df[target_col] == 0, column]
    data_yes = df.loc[df[target_col] == 1, column]
    ax.boxplot([data_no, data_yes], tick_labels=["No Churn", "Churn"])
    ax.set_title(f"{column} vs Churn")
    ax.set_ylabel(column)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df, numeric_columns: list, save_path: str) -> None:
    import numpy as np

    corr = df[numeric_columns].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_columns)))
    ax.set_yticks(range(len(numeric_columns)))
    ax.set_xticklabels(numeric_columns, rotation=45, ha="right")
    ax.set_yticklabels(numeric_columns)
    for i in range(len(numeric_columns)):
        for j in range(len(numeric_columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation Heatmap (numeric features)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(importance_df, save_path: str, top_n: int = 15) -> None:
    top = importance_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top["feature"], top["importance"], color="#4C72B0")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
