# Deployment Guide

## Deployment status

This repository is structured in a deployment-ready format for portfolio demonstration purposes. The final selected model can be trained, saved, reloaded, and used to score unseen customer records without retraining.

## Final deployment-ready model

The deployment-ready model in this repository is:

**Model 4 - Logistic Regression with Precision-Recall Threshold = 0.30**

## Deployment-ready components

The project includes the following components to support deployment-style usage:

- modular preprocessing in `src/preprocess.py`
- training pipeline in `src/train.py`
- evaluation utilities in `src/evaluate.py`
- reusable prediction logic in `src/predict.py`
- training entry point in `scripts/run_training.py`
- inference entry point in `scripts/run_inference.py`
- lightweight demo interface in `app/app.py`

## Inference workflow

The inference process works as follows:

1. Load saved model artifacts from `models/`
2. Read new customer data from a CSV file
3. Apply the same preprocessing used during training
4. Generate predicted probabilities
5. Apply the final threshold of **0.30**
6. Return binary prediction labels and probabilities

## Required artifacts

The following files should exist before inference:

- `models/model_4_pr_threshold_logistic.pkl`
- `models/scaler.pkl`
- `models/model_metadata.json`

These are created by running:

```bash
py -m scripts.run_training
```

## Running local inference

Batch inference can be tested with:

```bash
py -m scripts.run_inference
```

This uses:

```
sample_data/sample_input.csv
```

and writes predictions to:

```
reports/sample_predictions.csv
```

## Demo app

A lightweight app is included to demonstrate user-facing inference.

Run it locally with:

```bash
python -m streamlit run app/app.py
```

This app can be used to:
- upload a CSV
- score unseen customer records
- display prediction probabilities and labels

## Why this is deployment-ready

This project demonstrates deployment readiness because:

- notebook experimentation has been separated from the final pipeline
- model training and inference are modularised
- trained artifacts are reusable
- threshold-based prediction logic is explicit
- a lightweight interface is included for demo use

## Possible deployment options

This project could be extended into a real deployment using:
- Streamlit Community Cloud
- Hugging Face Spaces
- FastAPI or Flask API
- Docker container
- AWS, Azure, or GCP hosting

## Deployment considerations

Before production deployment, consider:
- stricter input validation
- feature drift monitoring
- model versioning
- logging and observability
- authentication and access control
- regular threshold and performance review

## Recommended portfolio demo flow

To demonstrate the project end-to-end:

1. Add the training dataset to `data/raw/`
2. Run:
   ```bash
   py -m scripts.run_training
   ```
3. Confirm model artifacts are saved in `models/`
4. Run:
   ```bash
   py -m scripts.run_inference
   ```
5. Open `reports/sample_predictions.csv`
6. Launch the app:
   ```bash
   python -m streamlit run app/app.py
   ```

This shows:
- reproducible training
- saved model artifacts
- scoring of unseen records
- deployment-style usability
