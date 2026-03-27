from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "raw" / "Bank_Personal_Loan_Modelling.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
IMAGES_DIR = ROOT_DIR / "images"
SAMPLE_INPUT_PATH = ROOT_DIR / "sample_data" / "sample_input.csv"

RANDOM_STATE = 98
TEST_SIZE = 0.30

# Selected deployment model:
# Model 4 - Logistic Regression with reduced statsmodels feature set
# and a Precision-Recall operating threshold of 0.30
FINAL_MODEL_NAME = "model_4_pr_threshold_logistic"
FINAL_THRESHOLD = 0.30

RAW_REQUIRED_COLUMNS = [
    "Age",
    "Experience",
    "Income",
    "ZIP Code",
    "Family",
    "CCAvg",
    "Education",
    "Mortgage",
    "Personal Loan",
    "Securities Account",
    "CD Account",
    "Online",
    "CreditCard",
]

REDUCED_DROP_COLUMNS = [
    "Regions_Central",
    "Regions_Los Angeles Region",
    "Regions_Southern",
    "Regions_Superior",
    "Age",
    "Mortgage",
]

MODEL5_SELECTED_FEATURES = [
    "Age",
    "Income",
    "CCAvg",
    "Mortgage",
    "SecuritiesAccount",
    "Online",
    "CreditCard",
    "Regions_Central",
    "Regions_Los Angeles Region",
    "Regions_Superior",
    "Education_2",
]
