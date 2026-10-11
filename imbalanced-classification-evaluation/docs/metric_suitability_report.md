# Metric Suitability Report: Imbalanced Fraud Detection

## Objective
Evaluate a binary classifier when positive cases (fraud) are rare. The demonstration uses a reproducible synthetic dataset with approximately 5% positive examples. Results from synthetic data illustrate the workflow; they are not evidence of real-world fraud performance.

## Why class imbalance matters
A model that predicts every case as legitimate could score about 95% accuracy while detecting no fraud. Therefore, accuracy is not a sufficient headline metric. Always inspect the confusion matrix and class-specific precision and recall.

## Metrics and their business meaning
- **Precision** = TP / (TP + FP). High precision means fewer legitimate transactions are incorrectly flagged. This matters when manual investigation is expensive or customers may be inconvenienced.
- **Recall** = TP / (TP + FN). High recall means more fraudulent transactions are detected. This matters when missed fraud is costly.
- **F1-score** = harmonic mean of precision and recall. Useful when both are important, but it weights precision and recall equally and does not encode business costs.
- **ROC-AUC** measures how well scores rank positives above negatives across thresholds. It is useful for comparing ranking performance, but may appear strong in highly imbalanced settings even when positive predictions are not very precise.
- **Precision-recall curve / average precision** focuses on positive-class retrieval and is often more revealing when positives are rare. Compare average precision with the positive-class prevalence.
- **Accuracy** is intuitive but can be misleading under strong imbalance.

## Threshold analysis
The default probability threshold of 0.50 is only a convention, not necessarily the best operating point. Lowering the threshold usually catches more fraud (higher recall) but also flags more legitimate transactions (lower precision and higher investigation workload). Raising it usually improves precision but misses more fraud.

Choose a threshold using validation data and an explicit cost model. For example:
`total_cost = false_positive_count * cost_of_review + false_negative_count * cost_of_missed_fraud`.
Do not choose a threshold using the final test set, because that leaks test information into model selection.

## Macro, micro, and weighted F1
- **Macro F1** computes F1 separately for each class and averages the class scores equally. It exposes poor performance on a rare class.
- **Weighted F1** averages class F1 scores weighted by class support. It reflects the class distribution, so the majority class has more influence.
- **Micro F1** aggregates TP, FP, and FN across classes before computing F1. In single-label classification, micro F1 equals accuracy; it can hide minority-class weaknesses.

The Python classification report prints per-class scores. For a multi-class extension, use `f1_score(y_true, y_pred, average="macro")`, `average="micro"`, and `average="weighted"` and compare them. For this binary demonstration, report positive-class precision/recall/F1 as well as macro and weighted F1 to keep minority-class performance visible.

## Recommendation
For rare-fraud screening, use **recall and precision-recall analysis as primary operational measures**, alongside average precision for ranking comparison. Select the threshold based on the relative costs of missed fraud and false alarms. Track precision, recall, false-positive volume, and false-negative volume after deployment. ROC-AUC is a useful secondary metric, not the only decision criterion.

## Limitations
The data are synthetic, and the baseline model is intentionally simple. Real deployment requires time-aware splits where appropriate, leakage checks, calibration checks, subgroup/fairness assessment, drift monitoring, and a threshold selected on validation data with business stakeholders.
