import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, f1_score, precision_score, recall_score
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from kneed import KneeLocator

def find_optimal_threshold(tpr, fpr, thresholds):
    j_statistic = tpr - fpr
    optimal_idx = np.argmax(j_statistic)
    return thresholds[optimal_idx]

def find_optimal_threshold_f1(precision, recall, thresholds):
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores[np.isnan(f1_scores)] = 0
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx]

def evaluate_model(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }

def plot_curve(fpr, tpr, roc_auc, curve_type="ROC"):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{curve_type} curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate' if curve_type == "ROC" else 'Recall')
    plt.ylabel('True Positive Rate' if curve_type == "ROC" else 'Precision')
    plt.title(f'{curve_type} Curve')
    plt.legend(loc="lower right")
    plt.show()

def main():
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]

    fpr_train, tpr_train, roc_thresholds_train = roc_curve(y_train, y_train_prob)
    fpr_test, tpr_test, roc_thresholds_test = roc_curve(y_test, y_test_prob)
    roc_auc_train = auc(fpr_train, tpr_train)
    roc_auc_test = auc(fpr_test, tpr_test)

    precision_train, recall_train, pr_thresholds_train = precision_recall_curve(y_train, y_train_prob)
    precision_test, recall_test, pr_thresholds_test = precision_recall_curve(y_test, y_test_prob)
    pr_auc_train = auc(recall_train, precision_train)
    pr_auc_test = auc(recall_test, precision_test)

    roc_threshold_train = find_optimal_threshold(tpr_train, fpr_train, roc_thresholds_train)
    f1_threshold_train = find_optimal_threshold_f1(precision_train, recall_train, pr_thresholds_train)

    pr_knee_locator = KneeLocator(recall_train, precision_train, curve="concave", direction="decreasing")
    pr_knee_threshold_train = pr_knee_locator.knee or 0.5

    roc_knee_locator = KneeLocator(fpr_train, tpr_train, curve="convex", direction="increasing")
    roc_knee_threshold_train = roc_knee_locator.knee or 0.5

    print(f"Optimal ROC Threshold: {roc_threshold_train}")
    print(f"Optimal F1 Threshold: {f1_threshold_train}")
    print(f"Knee Point Threshold (ROC): {roc_knee_threshold_train}")
    print(f"Knee Point Threshold (PR): {pr_knee_threshold_train}")

    roc_train_metrics = evaluate_model(y_train, y_train_prob, roc_knee_threshold_train)
    pr_train_metrics = evaluate_model(y_train, y_train_prob, pr_knee_threshold_train)

    print("\nModel Performance with ROC Knee Threshold:")
    print(roc_train_metrics)

    print("\nModel Performance with PR Knee Threshold:")
    print(pr_train_metrics)

    plot_curve(fpr_train, tpr_train, roc_auc_train, "ROC")
    plot_curve(recall_train, precision_train, pr_auc_train, "PR")

if __name__ == "__main__":
    main()
