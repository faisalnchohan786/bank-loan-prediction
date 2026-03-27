from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import List, Dict

import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

from src.config import (
    DATA_PATH,
    FINAL_MODEL_NAME,
    FINAL_THRESHOLD,
    REDUCED_DROP_COLUMNS,
    MODEL5_SELECTED_FEATURES,
    MODELS_DIR,
    REPORTS_DIR,
    IMAGES_DIR,
)
from src.preprocess import load_data, build_model_frame, split_and_scale
from src.evaluate import (
    evaluate_model,
    roc_optimal_threshold,
    plot_roc_curve,
    plot_precision_recall_threshold,
    plot_confusion,
)
from src.utils import ensure_dir, save_json


@dataclass
class TrainingArtifacts:
    model_path: str
    scaler_path: str
    metadata_path: str


def fit_statsmodels_logit(X_train: pd.DataFrame, y_train: pd.Series):
    X_train_const = sm.add_constant(X_train, has_constant="add")
    model = sm.Logit(y_train, X_train_const.astype(float))
    result = model.fit(disp=False)
    return result


def train_all_configurations(data_path=DATA_PATH) -> pd.DataFrame:
    df = load_data(data_path)
    model_df = build_model_frame(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(model_df)

    results: List[pd.DataFrame] = []

    # Model 1
    m1 = LogisticRegression(
        solver="newton-cg",
        random_state=1,
        fit_intercept=False,
        class_weight={0: 0.15, 1: 0.85},
        max_iter=2000,
    )
    m1.fit(X_train, y_train)
    y_prob_m1 = m1.predict_proba(X_test)[:, 1]
    results.append(evaluate_model("Model 1 - Logistic Regression (Scikit-learn)", y_test, y_prob_m1, 0.5))

    # Model 2
    m2 = fit_statsmodels_logit(X_train, y_train)
    X_test_const = sm.add_constant(X_test, has_constant="add")
    y_prob_m2 = m2.predict(X_test_const.astype(float))
    results.append(evaluate_model("Model 2 - Logistic Regression (Statsmodels)", y_test, y_prob_m2, 0.5))

    # Reduced feature frame for Models 3 and 4
    X_train_const = sm.add_constant(X_train, has_constant="add")
    X_test_const = sm.add_constant(X_test, has_constant="add")

    drop_cols = [c for c in REDUCED_DROP_COLUMNS if c in X_train_const.columns]
    X_train1 = X_train_const.drop(columns=drop_cols)
    X_test1 = X_test_const.drop(columns=drop_cols)

    m34 = sm.Logit(y_train, X_train1.astype(float)).fit(disp=False)
    y_prob_m34 = m34.predict(X_test1.astype(float))

    # Model 3
    roc_threshold = roc_optimal_threshold(y_test, y_prob_m34)
    results.append(
        evaluate_model(
            f"Model 3 - Logistic Regression (ROC-Optimal Threshold = {roc_threshold:.3f})",
            y_test,
            y_prob_m34,
            roc_threshold,
        )
    )

    # Model 4
    results.append(
        evaluate_model(
            f"Model 4 - Logistic Regression (Precision-Recall Threshold = {FINAL_THRESHOLD:.2f})",
            y_test,
            y_prob_m34,
            FINAL_THRESHOLD,
        )
    )

    # Model 5
    selected = [c for c in MODEL5_SELECTED_FEATURES if c in X_train.columns]
    X_train5 = X_train[selected]
    X_test5 = X_test[selected]
    m5 = LogisticRegression(solver="newton-cg", random_state=1, fit_intercept=False, max_iter=2000)
    m5.fit(X_train5, y_train)
    y_prob_m5 = m5.predict_proba(X_test5)[:, 1]
    results.append(evaluate_model("Model 5 - Logistic Regression (Sequential Feature Selection)", y_test, y_prob_m5, 0.5))

    comparison = pd.concat(results, ignore_index=True)
    comparison = comparison.sort_values(["f1", "precision", "recall"], ascending=False).reset_index(drop=True)
    return comparison


def train_final_model(data_path=DATA_PATH) -> tuple[pd.DataFrame, TrainingArtifacts]:
    ensure_dir(MODELS_DIR)
    ensure_dir(REPORTS_DIR)
    ensure_dir(IMAGES_DIR)

    df = load_data(data_path)
    model_df = build_model_frame(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(model_df)

    X_train_const = sm.add_constant(X_train, has_constant="add")
    X_test_const = sm.add_constant(X_test, has_constant="add")

    drop_cols = [c for c in REDUCED_DROP_COLUMNS if c in X_train_const.columns]
    X_train_final = X_train_const.drop(columns=drop_cols)
    X_test_final = X_test_const.drop(columns=drop_cols)

    final_model = sm.Logit(y_train, X_train_final.astype(float)).fit(disp=False)
    y_prob_test = final_model.predict(X_test_final.astype(float))

    metrics = evaluate_model("Final Model - Notebook Model 4", y_test, y_prob_test, FINAL_THRESHOLD)
    metrics.to_csv(REPORTS_DIR / "final_model_metrics.csv", index=False)

    plot_roc_curve(y_test, y_prob_test, IMAGES_DIR / "roc_curve.png")
    plot_precision_recall_threshold(y_test, y_prob_test, IMAGES_DIR / "pr_threshold_curve.png")
    plot_confusion(y_test, y_prob_test, FINAL_THRESHOLD, IMAGES_DIR / "confusion_matrix.png")

    model_path = MODELS_DIR / f"{FINAL_MODEL_NAME}.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    metadata_path = MODELS_DIR / "model_metadata.json"

    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    metadata: Dict[str, object] = {
        "model_name": FINAL_MODEL_NAME,
        "threshold": FINAL_THRESHOLD,
        "feature_names_before_const": list(X_train.columns),
        "final_feature_names": list(X_train_final.columns),
        "drop_columns_for_final_model": drop_cols,
    }
    save_json(metadata, metadata_path)

    artifacts = TrainingArtifacts(
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        metadata_path=str(metadata_path),
    )
    return metrics, artifacts
