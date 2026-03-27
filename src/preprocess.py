from __future__ import annotations

from typing import Iterable
import pandas as pd
import zipcodes as zcode
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE, TEST_SIZE, RAW_REQUIRED_COLUMNS

COUNTY_TO_REGION = {
    'Los Angeles County': 'Los Angeles Region',
    'San Diego County': 'Southern',
    'Santa Clara County': 'Bay Area',
    'Alameda County': 'Bay Area',
    'Orange County': 'Southern',
    'Orange Country': 'Southern',
    'San Francisco County': 'Bay Area',
    'San Mateo County': 'Bay Area',
    'Sacramento County': 'Central',
    'Santa Barbara County': 'Southern',
    'Yolo County': 'Central',
    'Monterey County': 'Bay Area',
    'Ventura County': 'Southern',
    'San Bernardino County': 'Southern',
    'Contra Costa County': 'Bay Area',
    'Santa Cruz County': 'Bay Area',
    'Riverside County': 'Southern',
    'Kern County': 'Southern',
    'Marin County': 'Bay Area',
    'San Luis Obispo County': 'Southern',
    'Solano County': 'Bay Area',
    'Humboldt County': 'Superior',
    'Sonoma County': 'Bay Area',
    'Fresno County': 'Central',
    'Placer County': 'Central',
    'Butte County': 'Superior',
    'Shasta County': 'Superior',
    'El Dorado County': 'Central',
    'Stanislaus County': 'Central',
    'San Benito County': 'Bay Area',
    'San Joaquin County': 'Central',
    'Mendocino County': 'Superior',
    'Tuolumne County': 'Central',
    'Siskiyou County': 'Superior',
    'Trinity County': 'Superior',
    'Merced County': 'Central',
    'Lake County': 'Superior',
    'Napa County': 'Bay Area',
    'Imperial County': 'Southern',
    93077: 'Southern',
    96651: 'Bay Area',
}

SPECIAL_ZIP_FIXES = {
    92717: 'Orange County',
    92634: 'Orange County',
}

DROP_COLUMNS_AFTER_EDA = [
    "Agebin",
    "ZIPCode",
    "County",
    "Experience",
    "Income_group",
    "Spending_group",
]

ONE_HOT_COLUMNS = ["Regions", "Education"]


def load_data(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in RAW_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return standardize_columns(df)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={
        "ZIP Code": "ZIPCode",
        "Personal Loan": "PersonalLoan",
        "Securities Account": "SecuritiesAccount",
        "CD Account": "CDAccount",
    }).copy()


def map_zip_to_county(zip_code) -> str:
    if pd.isna(zip_code):
        return "Unknown"
    zip_int = int(zip_code)
    if zip_int in SPECIAL_ZIP_FIXES:
        return SPECIAL_ZIP_FIXES[zip_int]
    matches = zcode.matching(str(zip_int))
    if len(matches) == 1:
        return matches[0].get("county") or "Unknown"
    return "Unknown"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["County"] = data["ZIPCode"].apply(map_zip_to_county)
    data["Regions"] = data["County"].map(COUNTY_TO_REGION).fillna("Unknown")
    data["Agebin"] = pd.cut(
        data["Age"],
        bins=[0, 30, 40, 50, 60, 100],
        labels=["18-30", "31-40", "41-50", "51-60", "61-100"],
    )
    return data


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    data = engineer_features(df)

    drop_existing = [c for c in DROP_COLUMNS_AFTER_EDA if c in data.columns]
    data = data.drop(columns=drop_existing)

    if "PersonalLoan" not in data.columns:
        raise ValueError("Expected target column 'PersonalLoan' after standardization.")

    X = data.drop(columns=["PersonalLoan"])
    y = data["PersonalLoan"].astype(int)

    for col in ONE_HOT_COLUMNS:
        if col not in X.columns:
            raise ValueError(f"Expected feature column '{col}' before encoding.")

    X = pd.get_dummies(X, columns=ONE_HOT_COLUMNS, drop_first=True, dtype="int8")
    model_df = pd.concat([X, y], axis=1)
    return model_df


def split_and_scale(model_df: pd.DataFrame):
    X = model_df.drop(columns=["PersonalLoan"])
    y = model_df["PersonalLoan"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def align_features(df: pd.DataFrame, feature_names: Iterable[str]) -> pd.DataFrame:
    aligned = df.copy()
    for col in feature_names:
        if col not in aligned.columns:
            aligned[col] = 0
    aligned = aligned[list(feature_names)]
    return aligned
