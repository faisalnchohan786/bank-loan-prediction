from src.config import DATA_PATH, REPORTS_DIR
from src.train import train_all_configurations, train_final_model
from src.utils import ensure_dir


def main():
    ensure_dir(REPORTS_DIR)
    comparison = train_all_configurations(DATA_PATH)
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    metrics, artifacts = train_final_model(DATA_PATH)

    print("\nModel comparison")
    print(comparison.to_string(index=False))
    print("\nFinal model metrics")
    print(metrics.to_string(index=False))
    print("\nSaved artifacts")
    print(artifacts)


if __name__ == "__main__":
    main()
