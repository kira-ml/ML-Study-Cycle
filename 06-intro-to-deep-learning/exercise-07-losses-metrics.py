"""
Corrected, cleaned, and PEP8-compliant script implementing simple
synthetic regression/classification data generation, common loss
functions, and evaluation metrics. The code is intentionally kept
minimal and self-contained for educational/demonstration purposes.
"""

from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------
# Synthetic data generation (regression + classification)
# ---------------------------------------------------------------------

np.random.seed(42)  # Fix the RNG seed for reproducibility — crucial for experiments

# Regression dataset: 1000 samples, 5 features
X_reg = np.random.randn(1000, 5)

# Ground-truth linear coefficients used to synthesize regression targets
# Defining explicit coefficients ensures interpretability of downstream metrics
true_coeffs = np.array([2.5, -1.3, 0.8, -0.5, 1.2])

# Generate regression targets via linear model plus Gaussian noise
# Noise reflects real-world measurement uncertainty, preventing trivial learning
y_reg_true = X_reg @ true_coeffs + 0.5 * np.random.randn(1000)

# Simulated regression predictions: introduce additional noise
# This mimics imperfect model outputs for loss/metric evaluation
y_reg_pred = y_reg_true + np.random.randn(1000) * 1.0

# Classification dataset: binary, with informative + redundant features
# Using sklearn’s utility ensures realistic feature redundancy and correlations
X_clf, y_clf_true = make_classification(
    n_samples=2000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    random_state=42,
)

# Split into train/test — standard practice to evaluate generalization
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf_true, test_size=0.3, random_state=42
)

# Generate synthetic logits with a random linear probe
# Serves as a proxy for uncalibrated model scores before sigmoid/softmax
logits = X_clf_test @ np.random.randn(X_clf_test.shape[1]) + np.random.randn(
    X_clf_test.shape[0]
) * 0.5

# Map logits to probabilities using the sigmoid function
# Critical step in binary classification pipelines
y_clf_probs = 1.0 / (1.0 + np.exp(-logits))

# Convert probabilities to hard predictions using the canonical 0.5 threshold
y_clf_pred = (y_clf_probs > 0.5).astype(int)


# ---------------------------------------------------------------------
# Loss functions (regression + classification)
# ---------------------------------------------------------------------


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error (MSE).
    Penalizes large errors quadratically — sensitive to outliers.
    Canonical choice in regression tasks with Gaussian noise assumptions.
    """
    return float(np.mean((y_true - y_pred) ** 2))


def mae_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error (MAE).
    Linear penalty on deviations; robust against outliers compared to MSE.
    Particularly interpretable since it shares target variable units.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def binary_cross_entropy(
    y_true: np.ndarray, y_pred_probs: np.ndarray, epsilon: float = 1e-12
) -> float:
    """
    Numerically-stable Binary Cross-Entropy (BCE).
    Measures the negative log-likelihood of true labels under predicted probabilities.
    Epsilon clipping guards against log(0) errors.
    """
    y_pred_probs = np.clip(y_pred_probs, epsilon, 1.0 - epsilon)
    return float(
        -np.mean(y_true * np.log(y_pred_probs) + (1 - y_true) * np.log(1 - y_pred_probs))
    )


def categorical_cross_entropy(y_true: np.ndarray, logits: np.ndarray) -> float:
    """
    Multi-class Categorical Cross-Entropy (CCE).
    Operates on raw logits, applying a numerically-stable softmax.
    Accepts integer labels or one-hot encodings, returning mean NLL across classes.
    """
    # Subtract row-wise max for numerical stability in softmax
    logits_stable = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits_stable)
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Convert labels to one-hot if not already
    if y_true.ndim == 1:
        one_hot = np.eye(logits.shape[1], dtype=float)[y_true]
    else:
        one_hot = y_true.astype(float)

    return float(-np.mean(np.sum(one_hot * np.log(softmax_probs + 1e-12), axis=1)))


def hinge_loss(y_true: np.ndarray, pred_decisions: np.ndarray) -> float:
    """
    Hinge loss for margin-based binary classifiers (e.g., SVM).
    Encourages correct class scores to exceed margin boundary of 1.0.
    Input labels are mapped {0,1} → {-1,+1} to align with margin formulation.
    """
    y_signed = 2 * y_true - 1  # map {0,1} -> {-1,+1}
    return float(np.mean(np.maximum(0.0, 1.0 - y_signed * pred_decisions)))


# Demonstration of regression losses
mse_value = mse_loss(y_reg_true, y_reg_pred)
mae_value = mae_loss(y_reg_true, y_reg_pred)
print(f"MSE: {mse_value:.4f}")
print(f"MAE: {mae_value:.4f}")

# Evaluate binary classification with BCE
bce_value = binary_cross_entropy(y_clf_test, y_clf_probs)
print(f"Binary Cross-Entropy: {bce_value:.4f}")

# Example categorical cross-entropy on toy logits
# Serves as sanity check independent of synthetic data
y_true_cat = np.array([0, 1, 2])
logits_example = np.array(
    [
        [2.0, 1.0, 0.1],
        [0.5, 2.0, 0.3],
        [0.2, 0.1, 3.0],
    ]
)
cce_value = categorical_cross_entropy(y_true_cat, logits_example)
print(f"Categorical Cross-Entropy (toy): {cce_value:.4f}")

# Example hinge loss with random decision scores
hinge_preds = np.random.randn(len(y_clf_test)) * 2.0
hinge_val = hinge_loss(y_clf_test, hinge_preds)
print(f"Hinge loss: {hinge_val:.4f}")


# ---------------------------------------------------------------------
# Classification metrics (accuracy, precision, recall, F1, AUC)
# ---------------------------------------------------------------------


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Accuracy: ratio of correctly predicted instances.
    Suitable for balanced datasets, but can obscure minority-class performance.
    """
    return float(np.mean(y_true == y_pred))


def precision_recall_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Compute confusion matrix counts: TP, FP, FN, TN.
    Provides building blocks for precision, recall, and F1 metrics.
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fp, fn, tn


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Precision: TP / (TP + FP).
    High precision implies few false positives — critical in domains like medical diagnosis.
    """
    tp, fp, _, _ = precision_recall_counts(y_true, y_pred)
    denom = tp + fp
    return float(tp / denom) if denom > 0 else 0.0


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Recall: TP / (TP + FN).
    High recall implies few false negatives — crucial for safety-critical detection tasks.
    """
    tp, _, fn, _ = precision_recall_counts(y_true, y_pred)
    denom = tp + fn
    return float(tp / denom) if denom > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    F1 score: harmonic mean of precision and recall.
    Useful for imbalanced datasets where accuracy is misleading.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    denom = p + r
    return float(2.0 * p * r / denom) if denom > 0 else 0.0


def auc_roc(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, List[float], List[float]]:
    """
    Compute ROC AUC via threshold sweep.
    - Sorts predicted scores descending and accumulates TPR/FPR rates.
    - Trapezoidal rule integration yields AUC.
    O(N log N) implementation, sufficient for moderate dataset sizes.
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)

    sorted_idx = np.argsort(y_scores)[::-1]
    y_sorted = y_true[sorted_idx]

    total_pos = int(np.sum(y_true == 1))
    total_neg = int(np.sum(y_true == 0))

    # Edge case: if one class absent, AUC is undefined; fallback to baseline 0.5
    if total_pos == 0 or total_neg == 0:
        fpr = [0.0, 1.0]
        tpr = [0.0, 1.0]
        return 0.5, fpr, tpr

    tpr = [0.0]
    fpr = [0.0]
    tp = 0
    fp = 0

    # Incrementally build ROC points by thresholding sorted scores
    for label in y_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / total_pos)
        fpr.append(fp / total_neg)

    # Integrate under curve using trapezoidal rule
    auc = 0.0
    for i in range(1, len(fpr)):
        auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2.0

    return float(auc), fpr, tpr


# Compute standard classification metrics
acc_value = accuracy(y_clf_test, y_clf_pred)
prec_value = precision(y_clf_test, y_clf_pred)
rec_value = recall(y_clf_test, y_clf_pred)
f1_value = f1_score(y_clf_test, y_clf_pred)
auc_value, fpr_curve, tpr_curve = auc_roc(y_clf_test, y_clf_probs)

print("Classification metrics")
print(f"Accuracy: {acc_value:.4f}")
print(f"Precision: {prec_value:.4f}")
print(f"Recall: {rec_value:.4f}")
print(f"F1 score: {f1_value:.4f}")
print(f"AUC: {auc_value:.4f}")
print(f"Binary Cross-entropy: {bce_value:.4f}")

# ---------------------------------------------------------------------
# Visualization: ROC Curve
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(fpr_curve, tpr_curve, label=f"ROC curve (AUC = {auc_value:.4f})")
plt.plot([0.0, 1.0], [0.0, 1.0], "k--", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------------------------------------------------
# Sensitivity analysis: effect of additive Gaussian noise on regression.
# Demonstrates robustness of MSE/MAE under varying noise magnitudes.
# ---------------------------------------------------------------------

noise_levels = [0.1, 0.5, 1.0, 2.0, 5.0]
results = []

for noise in noise_levels:
    # Add controlled Gaussian noise to regression targets
    y_noisy = y_reg_true + np.random.randn(len(y_reg_true)) * noise
    results.append(
        {
            "noise": float(noise),
            "mse": mse_loss(y_reg_true, y_noisy),
            "mae": mae_loss(y_reg_true, y_noisy),
        }
    )

# Tabulated summary of noise impact on regression metrics
print("\nNoise Level\tMSE\t\tMAE")
for res in results:
    print(f"{res['noise']:.1f}\t\t{res['mse']:.4f}\t{res['mae']:.4f}")
