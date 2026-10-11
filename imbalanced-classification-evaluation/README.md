# Evaluation Metrics for Imbalanced Classification

## Business context
This project simulates a fraud-detection task where fraudulent transactions are rare. A classifier can achieve high accuracy by predicting nearly every transaction as legitimate, so accuracy alone is not sufficient. We compare ROC-AUC, precision-recall behavior, threshold trade-offs, and macro/micro/weighted F1 scores.

## Project structure
```text
README.md
requirements.txt
notebooks/imbalanced_evaluation.ipynb
docs/metric_suitability_report.md
src/custom_metrics.py
```

## 1. Set up
Use Python 3.10+ (recommended). From this directory:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Run the analysis
Run the Python script:
```bash
python src/custom_metrics.py
```
It creates `outputs/roc_curve.png`, `outputs/precision_recall_curve.png`, `outputs/threshold_tradeoffs.png`, and prints classification metrics.

Or launch the notebook:
```bash
jupyter notebook notebooks/imbalanced_evaluation.ipynb
```

## 3. Dataset
The script generates a reproducible synthetic binary dataset with a 95:5 class ratio. This avoids requiring a private dataset and makes the project easy to reproduce. For real deployment, replace the synthetic data with approved, properly split production data.

## 4. Interpretation
- **Accuracy:** can hide poor minority-class performance.
- **Precision:** among predicted fraud cases, the fraction that are actually fraud.
- **Recall:** among all actual fraud cases, the fraction detected.
- **F1:** harmonic mean of precision and recall.
- **ROC-AUC:** ranking ability across thresholds; can look optimistic when negatives dominate.
- **Precision-recall curve / average precision:** often more informative when the positive class is rare.
- **Threshold:** lower thresholds usually increase recall but may create more false alarms.

## 5. GitHub submission checklist
- Commit the script, notebook, report, README, and requirements.
- Include generated plots if your instructor expects evidence (they are created under `outputs/` after running).
- Add a short note explaining the business costs of false negatives (missed fraud) versus false positives (legitimate transactions flagged).
