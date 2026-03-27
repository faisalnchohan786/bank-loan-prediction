# Model Card

## Model name
Bank Loan Prediction - Model 4 Logistic Regression

## Model type
Binary classification

## Objective
Predict whether a customer will accept a personal loan offer.

## Final selected model
**Model 4 - Logistic Regression with Precision-Recall Threshold**

## Target variable
`PersonalLoan`

## Final threshold
`0.30`

## Why this model was selected
Five Logistic Regression-based model configurations were evaluated in the notebook. Model 4 was selected because it achieved the best overall balance between precision and recall and produced the highest F1-score among the evaluated models.

## Evaluation summary

| Metric | Value |
|---|---:|
| Accuracy | 0.9573 |
| Recall | 0.7917 |
| Precision | 0.7703 |
| F1-score | 0.7808 |
| ROC AUC | 0.9721 |

## Input features
- Age
- Experience
- Income
- ZIP Code
- Family
- CCAvg
- Education
- Mortgage
- Securities Account
- CD Account
- Online
- CreditCard

## Prediction output
The model returns:
- `prediction_probability`
- `prediction_label`
- `threshold_used`

## Intended use
This model is designed to support personal loan campaign targeting by identifying customers who are more likely to accept a loan offer.

## Limitations
- Performance depends on the quality and representativeness of the training data
- The selected threshold may need review if business priorities change
- Predictions should support decision-making, not replace it

## Repository artifacts
- `models/model_4_pr_threshold_logistic.pkl`
- `models/scaler.pkl`
- `models/model_metadata.json`
- `reports/final_model_metrics.csv`