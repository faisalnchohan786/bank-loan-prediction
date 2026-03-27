import pandas as pd

from src.config import SAMPLE_INPUT_PATH
from src.predict import predict_dataframe


def main():
    # Load unseen sample data
    df = pd.read_csv(SAMPLE_INPUT_PATH)

    # Run inference
    predictions = predict_dataframe(df)

    # Sort by highest predicted probability
    predictions = predictions.sort_values(
        by="prediction_probability",
        ascending=False
    ).reset_index(drop=True)

    # Save full output
    output_path = "reports/sample_predictions.csv"
    predictions.to_csv(output_path, index=False)

    # Display a cleaner subset for quick review
    display_cols = [
        "Age",
        "Income",
        "Family",
        "CCAvg",
        "Education",
        "Mortgage",
        "Securities Account",
        "CD Account",
        "Online",
        "CreditCard",
        "prediction_probability",
        "prediction_label",
        "threshold_used",
    ]

    existing_display_cols = [c for c in display_cols if c in predictions.columns]

    print("\nTop scored customers (sorted by prediction probability):\n")
    print(predictions[existing_display_cols].to_string(index=False))

    print(f"\nSaved full predictions to: {output_path}")


if __name__ == "__main__":
    main()