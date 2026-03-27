# Project summary

The notebook evaluates five Logistic Regression configurations for personal loan prediction.

## Notebook conclusion
`Model 4` performs best overall, achieving the highest test F1-score and the strongest balance between recall and precision. `Models 1 and 5` achieve very high recall but at the cost of low precision, resulting in many false positives. `Model 2` is more conservative with lower recall, while `Model 3` improves recall through ROC-based threshold tuning but remains less balanced than `Model 4`.

## Final repository choice
This repository uses the notebook-selected final model for the scripted training and inference pipeline:
- reduced feature Logistic Regression
- precision-recall operating threshold = `0.30`
- deployable inference flow via saved artifacts and a small Streamlit app
