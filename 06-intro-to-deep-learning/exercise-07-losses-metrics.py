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
# Synthetic Data Generation: Understanding Model Evaluation Contexts
# ---------------------------------------------------------------------

# Setting a fixed seed ensures reproducible results for educational consistency
np.random.seed(42)

# REGRESSION DATASET: Simulates a linear relationship with noise
# - 1000 samples: Represents a typical training dataset size
# - 5 features: Independent variables that influence the target
# - true_coeffs: Actual weights the model should learn (ground truth)
# - y_reg_true: Target values generated using the true linear model + Gaussian noise
# - y_reg_pred: Simulated model predictions (e.g., from a trained regressor)
X_reg = np.random.randn(1000, 5)  # Feature matrix: 1000 samples x 5 features
true_coeffs = np.array([2.5, -1.3, 0.8, -0.5, 1.2])  # Ground truth coefficients
y_reg_true = X_reg @ true_coeffs + 0.5 * np.random.randn(1000)  # True targets with noise
y_reg_pred = y_reg_true + np.random.randn(1000) * 1.0  # Predicted targets with additional noise

# CLASSIFICATION DATASET: Generates a more complex binary classification problem
# - make_classification: Creates synthetic data with controlled properties
# - n_informative=5: 5 features carry predictive information
# - n_redundant=2: 2 features are linear combinations of informative ones
# This mimics real-world scenarios where features are often correlated
X_clf, y_clf_true = make_classification(
    n_samples=2000,      # Total number of data points
    n_features=10,       # Total number of input features
    n_informative=5,     # Number of truly informative features
    n_redundant=2,       # Number of redundant (correlated) features
    random_state=42      # For reproducible results
)

# Split data into training and testing sets (70% train, 30% test)
# This simulates a common practice in ML to evaluate generalization
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf_true, test_size=0.3, random_state=42
)

# Generate synthetic model predictions for the test set
# - logits: Raw model outputs (before sigmoid activation)
# - y_clf_probs: Probabilities for the positive class (0-1 range)
# - y_clf_pred: Binary predictions (thresholded at 0.5)
logits = X_clf_test @ np.random.randn(X_clf_test.shape[1]) + np.random.randn(
    X_clf_test.shape[0]
) * 0.5
y_clf_probs = 1.0 / (1.0 + np.exp(-logits))  # Sigmoid transformation
y_clf_pred = (y_clf_probs > 0.5).astype(int)  # Thresholding for binary prediction


# ---------------------------------------------------------------------
# Optimized Loss Functions: Measuring Model Performance
# ---------------------------------------------------------------------

def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Squared Error (MSE) - Common for Regression.

    Educational Insight:
    - Squares the errors, heavily penalizing large deviations
    - Differentiable everywhere, making it suitable for gradient-based optimization
    - Units are squared original units (e.g., if target is in meters, MSE is in meters^2)
    Formula: MSE = (1/n) * Σ(y_true - y_pred)^2
    """
    return float(np.mean((y_true - y_pred) ** 2))


def mae_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE) - Robust Regression Metric.

    Educational Insight:
    - Takes absolute value of errors, less sensitive to outliers than MSE
    - More interpretable: average magnitude of error in original units
    - Not differentiable at zero, which can complicate some optimization algorithms
    Formula: MAE = (1/n) * Σ|y_true - y_pred|
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def binary_cross_entropy(
    y_true: np.ndarray, y_pred_probs: np.ndarray, epsilon: float = 1e-12
) -> float:
    """
    Calculate Binary Cross-Entropy - Standard for Binary Classification.

    Educational Insight:
    - Measures the difference between two probability distributions
    - Assumes y_true is binary (0 or 1) and y_pred_probs are probabilities (0-1)
    - Logarithmic penalty: very confident wrong predictions are heavily penalized
    - Numerical clipping prevents log(0), which is undefined
    Formula: BCE = -(1/n) * Σ[y_true*log(y_pred) + (1-y_true)*log(1-y_pred)]
    """
    y_pred_probs = np.clip(y_pred_probs, epsilon, 1.0 - epsilon)  # Prevent log(0)
    return float(
        -np.mean(y_true * np.log(y_pred_probs) + (1 - y_true) * np.log(1 - y_pred_probs))
    )


def categorical_cross_entropy(y_true: np.ndarray, logits: np.ndarray) -> float:
    """
    Calculate Categorical Cross-Entropy - Standard for Multi-Class Classification.

    Educational Insight:
    - Requires logits (raw scores) as input, not probabilities
    - Applies softmax internally for numerical stability
    - Softmax converts logits to probabilities that sum to 1
    - Numerical stability: subtracts max logit to prevent overflow in exp()
    Formula: CCE = -(1/n) * Σ Σ(y_true_class * log(softmax(logits_class)))
    """
    # Numerical stability: subtract max logit per sample
    logits_stable = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits_stable)
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Handle both integer labels (e.g., [0, 1, 2]) and one-hot encodings (e.g., [[1,0,0], [0,1,0]])
    if y_true.ndim == 1:
        one_hot = np.eye(logits.shape[1], dtype=float)[y_true]
    else:
        one_hot = y_true.astype(float)

    return float(-np.mean(np.sum(one_hot * np.log(softmax_probs + 1e-12), axis=1)))


def hinge_loss(y_true: np.ndarray, pred_decisions: np.ndarray) -> float:
    """
    Calculate Hinge Loss - Used for Support Vector Machines (SVM).

    Educational Insight:
    - Expects binary labels as {-1, +1} (not {0, 1})
    - Encourages a margin of separation between classes
    - Zero loss if prediction is correct and confident enough (margin >= 1)
    - Linear penalty for insufficiently confident or incorrect predictions
    Formula: Hinge = max(0, 1 - y_true * y_pred_decision)
    """
    y_signed = 2 * y_true - 1  # Convert {0, 1} to {-1, +1}
    return float(np.mean(np.maximum(0.0, 1.0 - y_signed * pred_decisions)))


# ---------------------------------------------------------------------
# Optimized Classification Metrics: Evaluating Model Quality
# ---------------------------------------------------------------------

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Accuracy - Proportion of Correct Predictions.

    Educational Insight:
    - Most intuitive metric: (Correct Predictions) / (Total Predictions)
    - Can be misleading for imbalanced datasets (e.g., 95% accuracy on 95% majority class)
    - Range: [0, 1] where 1 is perfect accuracy
    Formula: Accuracy = Σ(I(y_true == y_pred)) / n
    """
    return float(np.mean(y_true == y_pred))


def precision_recall_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Efficiently compute True Positives, False Positives, False Negatives, True Negatives.

    Educational Insight:
    - Forms the basis for many other metrics (Precision, Recall, F1)
    - TP: Predicted Positive, Actually Positive
    - FP: Predicted Positive, Actually Negative (Type I Error)
    - FN: Predicted Negative, Actually Positive (Type II Error)
    - TN: Predicted Negative, Actually Negative
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
    Calculate Precision - Quality of Positive Predictions.

    Educational Insight:
    - "Of all positive predictions, how many were correct?"
    - High precision: Low false positive rate
    - Important when false positives are costly (e.g., spam detection)
    - Undefined if no positive predictions (TP + FP = 0), default to 0.0
    Formula: Precision = TP / (TP + FP)
    """
    tp, fp, _, _ = precision_recall_counts(y_true, y_pred)
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Recall (Sensitivity) - Coverage of Actual Positives.

    Educational Insight:
    - "Of all actual positives, how many did we correctly predict?"
    - High recall: Low false negative rate
    - Important when false negatives are costly (e.g., medical diagnosis)
    - Undefined if no actual positives (TP + FN = 0), default to 0.0
    Formula: Recall = TP / (TP + FN)
    """
    tp, _, fn, _ = precision_recall_counts(y_true, y_pred)
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate F1 Score - Harmonic Mean of Precision and Recall.

    Educational Insight:
    - Balances precision and recall, useful for imbalanced datasets
    - Harmonic mean is more sensitive to low values than arithmetic mean
    - Range: [0, 1] where 1 is perfect F1 score
    - Undefined if both precision and recall are 0, default to 0.0
    Formula: F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return float(2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0


def auc_roc(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, List[float], List[float]]:
    """
    Calculate Area Under the ROC Curve (AUC-ROC).

    Educational Insight:
    - ROC curve plots True Positive Rate (TPR) vs False Positive Rate (FPR) at various thresholds
    - TPR (Recall): TP / (TP + FN)
    - FPR: FP / (FP + TN) - Proportion of negatives incorrectly classified as positive
    - AUC: Probability that a randomly chosen positive sample is ranked higher than a negative one
    - Invariant to class imbalance and threshold choice
    - Range: [0, 1] where 0.5 is random classifier, 1.0 is perfect classifier
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)

    # Sort indices by scores in descending order to calculate TPR/FPR at thresholds
    sorted_idx = np.argsort(y_scores)[::-1]
    y_sorted = y_true[sorted_idx]

    total_pos = int(np.sum(y_true == 1))
    total_neg = int(np.sum(y_true == 0))

    if total_pos == 0 or total_neg == 0:
        # If no positive or negative samples, AUC is undefined; return random classifier value
        return 0.5, [0.0, 1.0], [0.0, 1.0]

    # Compute cumulative counts of true positives and false positives
    # As we lower the threshold (move along sorted scores), we accumulate counts
    cumsum = np.cumsum(y_sorted)
    tps = np.concatenate([[0], cumsum])  # TPs at each threshold (starting from 0)
    fps = np.concatenate([[0], np.cumsum(1 - y_sorted)])  # FPs at each threshold

    # Calculate TPR and FPR for each threshold point
    tpr = tps / total_pos
    fpr = fps / total_neg

    # Calculate AUC using the trapezoidal rule for numerical integration
    auc = np.trapz(tpr, fpr)
    return float(auc), fpr.tolist(), tpr.tolist()


# ---------------------------------------------------------------------
# Performance Evaluation and Visualization
# ---------------------------------------------------------------------

# Calculate and display regression metrics
mse_value = mse_loss(y_reg_true, y_reg_pred)
mae_value = mae_loss(y_reg_true, y_reg_pred)

# Calculate and display classification metrics
bce_value = binary_cross_entropy(y_clf_test, y_clf_probs)
acc_value = accuracy(y_clf_test, y_clf_pred)
prec_value = precision(y_clf_test, y_clf_pred)
rec_value = recall(y_clf_test, y_clf_pred)
f1_value = f1_score(y_clf_test, y_clf_pred)
auc_value, fpr_curve, tpr_curve = auc_roc(y_clf_test, y_clf_probs)

# Print results
print("REGRESSION METRICS:")
print(f"MSE: {mse_value:.4f} (Average squared error)")
print(f"MAE: {mae_value:.4f} (Average absolute error)")
print("\nCLASSIFICATION METRICS:")
print(f"Binary Cross-Entropy: {bce_value:.4f}")
print(f"Accuracy: {acc_value:.4f}")
print(f"Precision: {prec_value:.4f}")
print(f"Recall: {rec_value:.4f}")
print(f"F1 Score: {f1_value:.4f}")
print(f"AUC-ROC: {auc_value:.4f}")

# Visualize ROC curve to understand classifier performance across thresholds
plt.figure(figsize=(8, 6))
plt.plot(fpr_curve, tpr_curve, label=f"ROC curve (AUC = {auc_value:.4f})")
plt.plot([0.0, 1.0], [0.0, 1.0], "k--", label="Random classifier (AUC = 0.5)")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR / Recall)")
plt.title("Receiver Operating Characteristic (ROC) Curve\nMeasures classifier performance across thresholds")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Noise Sensitivity Analysis: How metrics react to prediction errors
# Demonstrates the importance of choosing appropriate metrics
noise_levels = [0.1, 0.5, 1.0, 2.0, 5.0]
print("\nNOISE SENSITIVITY ANALYSIS: Impact of Prediction Noise")
print("Noise Level\tMSE\t\tMAE")
for noise in noise_levels:
    y_noisy = y_reg_true + np.random.randn(len(y_reg_true)) * noise
    mse = mse_loss(y_reg_true, y_noisy)
    mae = mae_loss(y_reg_true, y_noisy)
    print(f"{noise:.1f}\t\t{mse:.4f}\t{mae:.4f}")
print("\nInsight: MSE increases quadratically with noise, while MAE increases linearly.")