from __future__ import annotations

from typing import Dict
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
import matplotlib.pyplot as plt


def evaluate_probabilities(y_true, y_prob, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def evaluate_model(name: str, y_true, y_prob, threshold: float) -> pd.DataFrame:
    metrics = evaluate_probabilities(y_true, y_prob, threshold)
    return pd.DataFrame([{ "model": name, "threshold": threshold, **metrics }])


def roc_optimal_threshold(y_true, y_prob) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx = (tpr - fpr).argmax()
    return float(thresholds[idx])


def plot_roc_curve(y_true, y_prob, output_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_precision_recall_threshold(y_true, y_prob, output_path):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, precision[:-1], linestyle="--", label="Precision")
    plt.plot(thresholds, recall[:-1], linestyle="--", label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision and Recall vs Threshold")
    plt.ylim([0, 1])
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_confusion(y_true, y_prob, threshold: float, output_path):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.set_title(f"Confusion Matrix @ threshold={threshold:.2f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], labels=["No", "Yes"])
    ax.set_yticks([0, 1], labels=["No", "Yes"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
