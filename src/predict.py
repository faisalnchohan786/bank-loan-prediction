from __future__ import annotations

import json
import pickle
import pandas as pd
import statsmodels.api as sm

from src.preprocess import load_data, build_model_frame, standardize_columns, engineer_features, align_features
from src.config import MODELS_DIR, FINAL_MODEL_NAME


def load_artifacts():
    with open(MODELS_DIR / f"{FINAL_MODEL_NAME}.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(MODELS_DIR / "model_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, scaler, metadata


def prepare_new_data(raw_df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    df = standardize_columns(raw_df)
    data = engineer_features(df)
    # keep target-less inference safe
    if "PersonalLoan" in data.columns:
        data = data.drop(columns=["PersonalLoan"])
    drop_candidates = ["Agebin", "ZIPCode", "County", "Experience", "Income_group", "Spending_group"]
    existing = [c for c in drop_candidates if c in data.columns]
    data = data.drop(columns=existing)
    data = pd.get_dummies(data, columns=["Regions", "Education"], drop_first=True, dtype="int8")
    X = align_features(data, metadata["feature_names_before_const"])
    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index,
    )
    X_scaled = sm.add_constant(X_scaled, has_constant="add")
    final_cols = metadata["final_feature_names"]
    for col in final_cols:
        if col not in X_scaled.columns:
            X_scaled[col] = 0
    X_scaled = X_scaled[final_cols]
    return X_scaled


def predict_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    global scaler
    model, scaler, metadata = load_artifacts()
    X_ready = prepare_new_data(raw_df, metadata)
    probs = model.predict(X_ready.astype(float))
    threshold = metadata["threshold"]
    preds = (probs >= threshold).astype(int)
    result = raw_df.copy()
    result["prediction_probability"] = probs
    result["prediction_label"] = preds
    result["threshold_used"] = threshold
    return result
