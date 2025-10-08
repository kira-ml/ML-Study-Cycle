"""
Enhanced, optimized, and PEP8-compliant script implementing synthetic
regression/classification data generation, common loss functions, and
evaluation metrics. The code is intentionally kept minimal and self-contained
for educational/demonstration purposes with improved performance and clarity.
"""

from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------
# Synthetic data generation (regression + classification)
# ---------------------------------------------------------------------

np.random.seed(42)

# Regression dataset: 1000 samples, 5 features
X_reg = np.random.randn(1000, 5)
true_coeffs = np.array([2.5, -1.3, 0.8, -0.5, 1.2])
y_reg_true = X_reg @ true_coeffs + 0.5 * np.random.randn(1000)
y_reg_pred = y_reg_true + np.random.randn(1000) * 1.0

# Classification dataset
X_clf, y_clf_true = make_classification(
    n_samples=2000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    random_state=42,
)

X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf_true, test_size=0.3, random_state=42
)

logits = X_clf_test @ np.random.randn(X_clf_test.shape[1]) + np.random.randn(
    X_clf_test.shape[0]
) * 0.5
y_clf_probs = 1.0 / (1.0 + np.exp(-logits))
y_clf_pred = (y_clf_probs > 0.5).astype(int)


# ---------------------------------------------------------------------
# Optimized Loss Functions
# ---------------------------------------------------------------------

def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Optimized Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


def mae_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Optimized Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def binary_cross_entropy(
    y_true: np.ndarray, y_pred_probs: np.ndarray, epsilon: float = 1e-12
) -> float:
    """Numerically-stable Binary Cross-Entropy."""
    y_pred_probs = np.clip(y_pred_probs, epsilon, 1.0 - epsilon)
    return float(
        -np.mean(y_true * np.log(y_pred_probs) + (1 - y_true) * np.log(1 - y_pred_probs))
    )


def categorical_cross_entropy(y_true: np.ndarray, logits: np.ndarray) -> float:
    """Multi-class Categorical Cross-Entropy with numerical stability."""
    logits_stable = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits_stable)
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    if y_true.ndim == 1:
        one_hot = np.eye(logits.shape[1], dtype=float)[y_true]
    else:
        one_hot = y_true.astype(float)

    return float(-np.mean(np.sum(one_hot * np.log(softmax_probs + 1e-12), axis=1)))


def hinge_loss(y_true: np.ndarray, pred_decisions: np.ndarray) -> float:
    """Optimized Hinge loss for binary classifiers."""
    y_signed = 2 * y_true - 1
    return float(np.mean(np.maximum(0.0, 1.0 - y_signed * pred_decisions)))


# ---------------------------------------------------------------------
# Optimized Classification Metrics
# ---------------------------------------------------------------------

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Optimized accuracy calculation."""
    return float(np.mean(y_true == y_pred))


def precision_recall_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """Efficient confusion matrix computation."""
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fp, fn, tn


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Optimized precision calculation."""
    tp, fp, _, _ = precision_recall_counts(y_true, y_pred)
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Optimized recall calculation."""
    tp, _, fn, _ = precision_recall_counts(y_true, y_pred)
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Optimized F1 score calculation."""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return float(2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0


def auc_roc(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, List[float], List[float]]:
    """Optimized ROC AUC calculation with vectorized operations."""
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)

    # Sort indices by scores in descending order
    sorted_idx = np.argsort(y_scores)[::-1]
    y_sorted = y_true[sorted_idx]

    total_pos = int(np.sum(y_true == 1))
    total_neg = int(np.sum(y_true == 0))

    if total_pos == 0 or total_neg == 0:
        return 0.5, [0.0, 1.0], [0.0, 1.0]

    # Compute cumulative true positives and false positives
    cumsum = np.cumsum(y_sorted)
    tps = np.concatenate([[0], cumsum])
    fps = np.concatenate([[0], np.cumsum(1 - y_sorted)])

    # Calculate TPR and FPR
    tpr = tps / total_pos
    fpr = fps / total_neg

    # Calculate AUC using trapezoidal rule
    auc = np.trapz(tpr, fpr)
    return float(auc), fpr.tolist(), tpr.tolist()


# ---------------------------------------------------------------------
# Performance and Visualization
# ---------------------------------------------------------------------

# Regression metrics
mse_value = mse_loss(y_reg_true, y_reg_pred)
mae_value = mae_loss(y_reg_true, y_reg_pred)

# Classification metrics
bce_value = binary_cross_entropy(y_clf_test, y_clf_probs)
acc_value = accuracy(y_clf_test, y_clf_pred)
prec_value = precision(y_clf_test, y_clf_pred)
rec_value = recall(y_clf_test, y_clf_pred)
f1_value = f1_score(y_clf_test, y_clf_pred)
auc_value, fpr_curve, tpr_curve = auc_roc(y_clf_test, y_clf_probs)

# Print results
print(f"MSE: {mse_value:.4f}")
print(f"MAE: {mae_value:.4f}")
print(f"Binary Cross-Entropy: {bce_value:.4f}")
print("Classification metrics")
print(f"Accuracy: {acc_value:.4f}")
print(f"Precision: {prec_value:.4f}")
print(f"Recall: {rec_value:.4f}")
print(f"F1 score: {f1_value:.4f}")
print(f"AUC: {auc_value:.4f}")

# Visualize ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_curve, tpr_curve, label=f"ROC curve (AUC = {auc_value:.4f})")
plt.plot([0.0, 1.0], [0.0, 1.0], "k--", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.grid(True)
plt.show()

# Noise sensitivity analysis
noise_levels = [0.1, 0.5, 1.0, 2.0, 5.0]
print("\nNoise Level\tMSE\t\tMAE")
for noise in noise_levels:
    y_noisy = y_reg_true + np.random.randn(len(y_reg_true)) * noise
    mse = mse_loss(y_reg_true, y_noisy)
    mae = mae_loss(y_reg_true, y_noisy)
    print(f"{noise:.1f}\t\t{mse:.4f}\t{mae:.4f}")