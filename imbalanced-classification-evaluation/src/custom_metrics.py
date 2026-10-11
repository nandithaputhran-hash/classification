"""Custom evaluation utilities and reproducible imbalanced-classification demo."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, classification_report
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

def confusion_counts(y_true, y_pred):
    """Return (tn, fp, fn, tp) for binary labels encoded as 0/1."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)

def custom_precision_recall_f1(y_true, y_pred):
    """Compute binary precision, recall and F1 directly from confusion counts."""
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}

def threshold_table(y_true, probabilities, thresholds=(0.10, 0.20, 0.30, 0.40, 0.50, 0.60)):
    """Return precision/recall/F1 and confusion counts at each threshold."""
    rows = []
    for threshold in thresholds:
        pred = (probabilities >= threshold).astype(int)
        values = custom_precision_recall_f1(y_true, pred)
        rows.append({"threshold": threshold, **values})
    return rows

def main():
    X, y = make_classification(
        n_samples=8000, n_features=20, n_informative=6, n_redundant=4,
        n_clusters_per_class=2, weights=[0.95, 0.05], class_sep=1.0,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    model = LogisticRegression(max_iter=2000, class_weight=None, random_state=42)
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    custom = custom_precision_recall_f1(y_test, predictions)
    print("=== Baseline at threshold 0.50 ===")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {custom['precision']:.4f}")
    print(f"Recall: {custom['recall']:.4f}")
    print(f"F1 (custom): {custom['f1']:.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")
    print(f"Average precision: {average_precision_score(y_test, probabilities):.4f}")
    print("\\nConfusion counts (TN, FP, FN, TP):",
          confusion_counts(y_test, predictions))
    print("\\nClassification report:")
    print(classification_report(y_test, predictions, digits=4, zero_division=0))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_test, probabilities):.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "roc_curve.png", dpi=160)
    plt.close()

    # Precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    ap = average_precision_score(y_test, probabilities)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"Average precision = {ap:.3f}")
    plt.axhline(y=np.mean(y_test), linestyle="--",
                label=f"Positive prevalence = {np.mean(y_test):.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "precision_recall_curve.png", dpi=160)
    plt.close()

    # Threshold trade-offs
    rows = threshold_table(y_test, probabilities)
    thresholds = [r["threshold"] for r in rows]
    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, [r["precision"] for r in rows], marker="o", label="Precision")
    plt.plot(thresholds, [r["recall"] for r in rows], marker="o", label="Recall")
    plt.plot(thresholds, [r["f1"] for r in rows], marker="o", label="F1")
    plt.xlabel("Classification threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title("Threshold Trade-offs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "threshold_tradeoffs.png", dpi=160)
    plt.close()

    print("\\n=== Threshold trade-offs ===")
    print("threshold | precision | recall | F1 | FP | FN")
    for r in rows:
        print(f"{r['threshold']:.2f}      | {r['precision']:.3f}     | "
              f"{r['recall']:.3f} | {r['f1']:.3f} | {r['fp']} | {r['fn']}")
    print(f"\\nPlots saved in: {OUT}")

if __name__ == "__main__":
    main()
