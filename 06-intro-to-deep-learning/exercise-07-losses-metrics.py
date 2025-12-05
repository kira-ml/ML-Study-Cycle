"""
Machine Learning Fundamentals: A Beginner-Friendly Reference Implementation

This script demonstrates core ML concepts through synthetic data generation,
loss functions, and evaluation metrics. Designed for newcomers to understand
fundamentals before diving into complex frameworks.

Key Learning Objectives:
1. How synthetic data helps understand model behavior
2. How loss functions guide model learning during training
3. How evaluation metrics assess real-world performance
4. How to implement these concepts from scratch

Author: Open Source ML Community
License: MIT (Free for educational use)
"""

from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------
# SECTION 1: SYNTHETIC DATA GENERATION
# Why synthetic data? It lets us understand model behavior in controlled
# environments before working with messy real-world data.
# ---------------------------------------------------------------------

# Setting a seed makes random numbers reproducible - crucial for debugging!
# In real projects, always set seeds to get consistent results.
np.random.seed(42)

# ---------------------------------------------------------------------
# 1.1 REGRESSION DATASET: Predicting continuous values
# Think: Predicting house prices, stock values, or temperatures
# ---------------------------------------------------------------------
# Generate 1000 samples with 5 features (like: # bedrooms, square footage, etc.)
X_reg = np.random.randn(1000, 5)

# True coefficients our ideal model would learn
# In real life, we NEVER know these - that's why we need ML!
true_coeffs = np.array([2.5, -1.3, 0.8, -0.5, 1.2])

# Create target values using a linear relationship: y = X*w + noise
# Real-world data always has noise (measurement errors, unobserved factors)
y_reg_true = X_reg @ true_coeffs + 0.5 * np.random.randn(1000)

# Simulate model predictions (no real training here - just for demonstration)
# Adding more noise to simulate imperfect predictions
y_reg_pred = y_reg_true + np.random.randn(1000) * 1.0

# ---------------------------------------------------------------------
# 1.2 CLASSIFICATION DATASET: Predicting categories (0 or 1)
# Think: Spam detection, disease diagnosis, image classification
# ---------------------------------------------------------------------
# sklearn's make_classification creates realistic classification problems
# It generates features with different levels of usefulness
X_clf, y_clf_true = make_classification(
    n_samples=2000,      # More samples than regression (classification often needs more data)
    n_features=10,       # Total features
    n_informative=5,     # Only 5 actually matter for prediction (realistic!)
    n_redundant=2,       # 2 features are just combinations of others (common in real data)
    random_state=42      # Reproducibility
)

# Train-test split: The MOST important practice in ML
# We train on one set, test on another to check if our model generalizes
# 70% train, 30% test is a common starting point
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf_true, test_size=0.30, random_state=42
)

# Generate fake model outputs to demonstrate metrics
# In reality, these come from training a classifier
logits = X_clf_test @ np.random.randn(X_clf_test.shape[1]) + np.random.randn(
    X_clf_test.shape[0]
) * 0.5

# Convert logits (raw scores) to probabilities using sigmoid
# Sigmoid squashes values between 0 and 1: 0 = definitely class 0, 1 = definitely class 1
y_clf_probs = 1.0 / (1.0 + np.exp(-logits))

# Convert probabilities to binary predictions (threshold at 0.5)
# Threshold tuning is important in practice (not always 0.5!)
y_clf_pred = (y_clf_probs > 0.5).astype(int)


# ---------------------------------------------------------------------
# SECTION 2: LOSS FUNCTIONS - The "Report Card" During Training
# Loss functions tell our model how wrong it is during training.
# The model tries to minimize loss (like minimizing mistakes on homework).
# ---------------------------------------------------------------------

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error (MSE) - Most common regression loss.
    
    Beginner's Analogy: Imagine your predictions are darts. MSE heavily penalizes
    darts that land far from the bullseye. A miss by 2cm counts as 4 (2²), 
    while MAE (below) would count it as 2.
    
    Pros:
    - Strongly penalizes large errors (good for safety-critical applications)
    - Smooth curve helps gradient descent (the learning algorithm)
    
    Cons:
    - Sensitive to outliers (one bad prediction can dominate the loss)
    - Hard to interpret (units are squared)
    
    Formula: MSE = average of (true - predicted)² across all samples
    """
    return float(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error (MAE) - Robust regression loss.
    
    Beginner's Analogy: Like measuring average miss distance with a ruler.
    If you miss by 2cm, it counts as 2. Miss by 4cm, counts as 4.
    
    Pros:
    - Easy to interpret (average error in original units)
    - Less sensitive to outliers than MSE
    
    Cons:
    - Not differentiable at 0 (can cause issues with some optimizers)
    - Doesn't emphasize large errors as much
    
    When to use MAE over MSE:
    - When your data has outliers (sensor errors, data entry mistakes)
    - When you care about average error more than worst-case error
    
    Formula: MAE = average of |true - predicted| across all samples
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def binary_cross_entropy(
    y_true: np.ndarray, y_pred_probs: np.ndarray, epsilon: float = 1e-12
) -> float:
    """
    Binary Cross-Entropy (BCE) - Standard for binary classification.
    
    Beginner's Analogy: Imagine you're guessing coin flips.
    - If you predict 90% heads and it's heads: small penalty (-log(0.9) = 0.105)
    - If you predict 90% heads and it's tails: BIG penalty (-log(0.1) = 2.303)
    
    Intuition: The more confident you are in a wrong answer, the more you're penalized.
    
    Technical Details:
    - Inputs must be probabilities (0-1), not raw scores
    - We clip probabilities to avoid log(0) which is undefined
    - BCE measures distance between two probability distributions
    
    Formula: BCE = -average[ y*log(p) + (1-y)*log(1-p) ]
    Where y = true label (0 or 1), p = predicted probability
    """
    # Clip to prevent numerical errors (log(0) is infinity!)
    y_pred_probs = np.clip(y_pred_probs, epsilon, 1.0 - epsilon)
    
    # Calculate the cross-entropy loss
    loss = -np.mean(y_true * np.log(y_pred_probs) + (1 - y_true) * np.log(1 - y_pred_probs))
    return float(loss)


def categorical_cross_entropy(y_true: np.ndarray, logits: np.ndarray) -> float:
    """
    Categorical Cross-Entropy (CCE) - For multi-class classification (3+ classes).
    
    Beginner's Analogy: Imagine identifying dog breeds from photos.
    - If true breed is "Poodle" and you predict: 
        Poodle: 90%, Lab: 5%, Husky: 5% → small penalty
        Poodle: 10%, Lab: 80%, Husky: 10% → large penalty
    
    Key Differences from BCE:
    1. Uses logits (raw scores) not probabilities - more numerically stable
    2. Applies softmax internally to convert to probabilities
    3. Handles multiple classes simultaneously
    
    Numerical Stability Trick:
    We subtract the maximum logit value before exp() to prevent overflow.
    This doesn't change the probabilities due to softmax properties.
    
    Formula: CCE = -average[ sum over classes(y_true_class * log(p_pred_class)) ]
    """
    # Step 1: Numerical stability - subtract max for each sample
    # This prevents exp(large number) = infinity
    logits_stable = logits - np.max(logits, axis=1, keepdims=True)
    
    # Step 2: Apply softmax: exp(logit) / sum(exp(all logits))
    exp_logits = np.exp(logits_stable)
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Step 3: Handle different label formats
    if y_true.ndim == 1:  # Labels like [0, 2, 1, 0] (class indices)
        # Convert to one-hot: [0, 2, 1] → [[1,0,0], [0,0,1], [0,1,0]]
        num_classes = logits.shape[1]
        one_hot = np.eye(num_classes, dtype=float)[y_true]
    else:  # Already one-hot encoded
        one_hot = y_true.astype(float)
    
    # Step 4: Calculate loss
    loss = -np.mean(np.sum(one_hot * np.log(softmax_probs + 1e-12), axis=1))
    return float(loss)


def hinge_loss(y_true: np.ndarray, pred_decisions: np.ndarray) -> float:
    """
    Hinge Loss - Used by Support Vector Machines (SVMs).
    
    Beginner's Analogy: Imagine a safety margin requirement.
    - If correct and confident (margin > 1): No penalty (loss = 0)
    - If correct but not confident (margin < 1): Small penalty
    - If wrong: Larger penalty
    
    Key Insight: Hinge loss creates a "margin" around the decision boundary.
    SVMs try to maximize this margin for better generalization.
    
    Note: Expects labels as {-1, +1}, not {0, 1}!
    
    Formula: Hinge = average of max(0, 1 - y_true * y_pred_decision)
    """
    # Convert {0, 1} labels to {-1, +1} for hinge loss
    y_signed = 2 * y_true - 1
    
    # Calculate hinge loss
    losses = np.maximum(0.0, 1.0 - y_signed * pred_decisions)
    return float(np.mean(losses))


# ---------------------------------------------------------------------
# SECTION 3: EVALUATION METRICS - The "Final Exam"
# Metrics tell us how well our model performs AFTER training.
# Different metrics answer different questions about model performance.
# ---------------------------------------------------------------------

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Accuracy - Most intuitive classification metric.
    
    Beginner's Question: "What percent did we get right?"
    
    WARNING: Accuracy can be misleading with imbalanced classes!
    Example: 95% of emails are not spam (negative class).
    A dumb model that always says "not spam" gets 95% accuracy!
    
    When to use accuracy:
    - When classes are roughly balanced (50% spam, 50% not spam)
    - When false positives and false negatives are equally costly
    
    Formula: Accuracy = (Correct Predictions) / (Total Predictions)
    """
    return float(np.mean(y_true == y_pred))


def calculate_confusion_matrix_counts(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[int, int, int, int]:
    """
    Calculate the four fundamental counts for binary classification.
    
    The Confusion Matrix (it confuses everyone at first!):
    
                    Actual
                 Positive  Negative
    Predicted +   TP        FP
              -   FN        TN
    
    Memory trick:
    - True/False: Was the prediction correct? (True = correct, False = wrong)
    - Positive/Negative: What was predicted? (Positive = predicted 1, Negative = predicted 0)
    
    TP: Predicted 1, Actually 1 (Good! We want this high)
    FP: Predicted 1, Actually 0 (Type I Error - False Alarm)
    FN: Predicted 0, Actually 1 (Type II Error - Missed Detection)
    TN: Predicted 0, Actually 0 (Good! We want this high)
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
    Precision - "How careful are we with positive predictions?"
    
    Question answered: "Of all the times we said 'positive', how often were we right?"
    
    High precision means: When we predict positive, we're usually correct.
    
    Use when false positives are costly:
    - Email spam filter (don't want to lose important emails)
    - Medical test for rare disease (don't want false alarms)
    
    Formula: Precision = TP / (TP + FP)
    """
    tp, fp, _, _ = calculate_confusion_matrix_counts(y_true, y_pred)
    
    # Handle edge case: no positive predictions
    if (tp + fp) == 0:
        return 0.0  # By convention
    
    return float(tp / (tp + fp))


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Recall (Sensitivity) - "How good are we at finding all positives?"
    
    Question answered: "Of all the actual positives, how many did we find?"
    
    High recall means: We miss few actual positives.
    
    Use when false negatives are costly:
    - Cancer detection (don't want to miss actual cancer)
    - Fraud detection (don't want to miss fraudulent transactions)
    
    Formula: Recall = TP / (TP + FN)
    """
    tp, _, fn, _ = calculate_confusion_matrix_counts(y_true, y_pred)
    
    # Handle edge case: no actual positives in dataset
    if (tp + fn) == 0:
        return 0.0  # By convention
    
    return float(tp / (tp + fn))


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    F1 Score - Balances precision and recall.
    
    Why harmonic mean, not average? 
    Harmonic mean penalizes extreme values more.
    Example: Precision=1.0, Recall=0.0 → F1=0, but average would be 0.5!
    
    When to use F1:
    - When you need a single number to summarize performance
    - When both false positives and false negatives matter
    - With imbalanced datasets (common in real world)
    
    Formula: F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    
    # Handle edge case: both are zero
    if (p + r) == 0:
        return 0.0
    
    return float(2.0 * p * r / (p + r))


def roc_auc_score(
    y_true: np.ndarray, y_scores: np.ndarray
) -> Tuple[float, List[float], List[float]]:
    """
    ROC Curve & AUC - Evaluates classifier at ALL possible thresholds.
    
    ROC Curve (Receiver Operating Characteristic):
    - X-axis: False Positive Rate (FPR) = FP / (FP + TN)
    - Y-axis: True Positive Rate (TPR) = Recall = TP / (TP + FN)
    
    AUC (Area Under Curve): Probability that a random positive example
    scores higher than a random negative example.
    
    Key Insights:
    - AUC = 0.5: Random guessing (diagonal line)
    - AUC = 1.0: Perfect classifier
    - AUC is threshold-independent (evaluates all thresholds!)
    - Great for imbalanced data (unlike accuracy)
    
    How to read ROC curves:
    - Steeper curve near (0,0) → Good at avoiding false positives
    - High curve near (0,1) → Good overall performance
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)
    
    # Count positive and negative samples
    total_pos = int(np.sum(y_true == 1))
    total_neg = int(np.sum(y_true == 0))
    
    # Edge cases: if no positives or negatives, AUC is undefined
    if total_pos == 0 or total_neg == 0:
        # Return random classifier values
        return 0.5, [0.0, 1.0], [0.0, 1.0]
    
    # Sort by predicted score (descending)
    # Higher scores = more likely to be positive
    sorted_indices = np.argsort(y_scores)[::-1]
    y_sorted = y_true[sorted_indices]
    
    # Calculate cumulative true positives and false positives
    # as we move down the threshold (include more samples as positive)
    cumulative_tp = np.cumsum(y_sorted)
    cumulative_fp = np.cumsum(1 - y_sorted)
    
    # Add starting point (0, 0) - threshold where we predict all as negative
    tps = np.concatenate([[0], cumulative_tp])
    fps = np.concatenate([[0], cumulative_fp])
    
    # Calculate rates
    tpr = tps / total_pos  # True Positive Rate (Recall)
    fpr = fps / total_neg  # False Positive Rate
    
    # Calculate AUC using trapezoidal rule (numerical integration)
    auc = np.trapz(tpr, fpr)
    
    return float(auc), fpr.tolist(), tpr.tolist()


# ---------------------------------------------------------------------
# SECTION 4: PUTTING IT ALL TOGETHER - Real Application
# ---------------------------------------------------------------------

print("=" * 60)
print("MACHINE LEARNING FUNDAMENTALS DEMONSTRATION")
print("=" * 60)

# Calculate regression metrics
print("\n1. REGRESSION METRICS (Predicting continuous values)")
print("-" * 50)
mse_value = mean_squared_error(y_reg_true, y_reg_pred)
mae_value = mean_absolute_error(y_reg_true, y_reg_pred)

print(f"Mean Squared Error (MSE):     {mse_value:8.4f}")
print(f"Mean Absolute Error (MAE):    {mae_value:8.4f}")
print("\nInterpretation:")
print(f"- Average squared error: {mse_value:.2f} units²")
print(f"- Average absolute error: {mae_value:.2f} units")
print("- MAE is easier to interpret (in original units)")

# Calculate classification metrics
print("\n\n2. CLASSIFICATION METRICS (Predicting categories)")
print("-" * 50)

# Calculate all metrics
bce_value = binary_cross_entropy(y_clf_test, y_clf_probs)
acc_value = accuracy(y_clf_test, y_clf_pred)
prec_value = precision(y_clf_test, y_clf_pred)
rec_value = recall(y_clf_test, y_clf_pred)
f1_value = f1_score(y_clf_test, y_clf_pred)
auc_value, fpr_curve, tpr_curve = roc_auc_score(y_clf_test, y_clf_probs)

# Get confusion matrix counts for interpretation
tp, fp, fn, tn = calculate_confusion_matrix_counts(y_clf_test, y_clf_pred)

print("Confusion Matrix Breakdown:")
print(f"True Positives (TP):  {tp:4d} - Correctly predicted positive")
print(f"False Positives (FP): {fp:4d} - Wrongly predicted positive (Type I Error)")
print(f"False Negatives (FN): {fn:4d} - Wrongly predicted negative (Type II Error)")
print(f"True Negatives (TN):  {tn:4d} - Correctly predicted negative")

print("\nPerformance Metrics:")
print(f"Binary Cross-Entropy: {bce_value:8.4f}  (Lower is better, 0 is perfect)")
print(f"Accuracy:             {acc_value:8.4f}  ({acc_value*100:.1f}% correct)")
print(f"Precision:            {prec_value:8.4f}  ({prec_value*100:.1f}% of + predictions correct)")
print(f"Recall:               {rec_value:8.4f}  ({rec_value*100:.1f}% of actual + found)")
print(f"F1 Score:             {f1_value:8.4f}  (Balance of precision & recall)")
print(f"AUC-ROC:              {auc_value:8.4f}  (0.5 = random, 1.0 = perfect)")

print("\nPractical Interpretation:")
if prec_value > rec_value:
    print("- Model is more careful than thorough (higher precision)")
elif rec_value > prec_value:
    print("- Model is more thorough than careful (higher recall)")
else:
    print("- Model balances precision and recall equally")

if auc_value > 0.7:
    print(f"- Good discriminative power (AUC = {auc_value:.3f})")
elif auc_value > 0.5:
    print(f"- Some predictive power, but weak (AUC = {auc_value:.3f})")
else:
    print(f"- Worse than random guessing! (AUC = {auc_value:.3f})")

# ---------------------------------------------------------------------
# SECTION 5: VISUALIZATION - Seeing is Understanding
# ---------------------------------------------------------------------

# Create a 2x2 grid of plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Machine Learning Metrics Visualization', fontsize=16, fontweight='bold')

# Plot 1: ROC Curve
ax = axes[0, 0]
ax.plot(fpr_curve, tpr_curve, 'b-', linewidth=2, label=f'Classifier (AUC = {auc_value:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')
ax.set_xlabel('False Positive Rate (FPR)')
ax.set_ylabel('True Positive Rate (TPR / Recall)')
ax.set_title('ROC Curve: Trade-off between Sensitivity and Specificity')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Plot 2: Precision-Recall Trade-off
ax = axes[0, 1]
ax.scatter(rec_value, prec_value, s=100, c='red', zorder=5, label='Our Model')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Trade-off')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend()
ax.grid(True, alpha=0.3)

# Add text explanation
ax.text(0.05, 0.95, f'F1 Score = {f1_value:.3f}', 
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 3: Regression Error Visualization
ax = axes[1, 0]
sample_indices = np.random.choice(len(y_reg_true), 50, replace=False)
sample_indices = np.sort(sample_indices)

ax.plot(y_reg_true[sample_indices], 'go-', label='True Values', alpha=0.7)
ax.plot(y_reg_pred[sample_indices], 'ro-', label='Predictions', alpha=0.7)

# Draw error lines
for i in sample_indices[:20]:  # Only show first 20 for clarity
    ax.plot([i, i], [y_reg_true[i], y_reg_pred[i]], 'k-', alpha=0.3, linewidth=0.5)

ax.set_xlabel('Sample Index')
ax.set_ylabel('Target Value')
ax.set_title(f'Regression: True vs Predicted (MSE = {mse_value:.3f})')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Loss Function Comparison
ax = axes[1, 1]
errors = np.linspace(-5, 5, 100)
mse_vals = errors ** 2
mae_vals = np.abs(errors)

ax.plot(errors, mse_vals, 'r-', label='MSE (Squared Error)', linewidth=2)
ax.plot(errors, mae_vals, 'b-', label='MAE (Absolute Error)', linewidth=2)
ax.set_xlabel('Prediction Error (True - Predicted)')
ax.set_ylabel('Loss Value')
ax.set_title('Loss Functions: MSE vs MAE')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 10)

# Add annotations
ax.annotate('MSE penalizes\nlarge errors more', xy=(3, 9), xytext=(1, 7),
            arrowprops=dict(arrowstyle='->', color='red'), color='red')
ax.annotate('MAE linear\npenalty', xy=(-3, 3), xytext=(-4.5, 5),
            arrowprops=dict(arrowstyle='->', color='blue'), color='blue')

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------
# SECTION 6: PRACTICAL EXERCISE - Noise Sensitivity
# ---------------------------------------------------------------------

print("\n" + "=" * 60)
print("PRACTICAL EXERCISE: How Noise Affects Different Metrics")
print("=" * 60)

print("\nAdding increasing noise to predictions and observing metrics:")
print("-" * 60)
print("Noise Level\tMSE\t\tMAE\t\tInterpretation")
print("-" * 60)

noise_levels = [0.1, 0.5, 1.0, 2.0, 5.0]
for noise in noise_levels:
    y_noisy = y_reg_true + np.random.randn(len(y_reg_true)) * noise
    mse = mean_squared_error(y_reg_true, y_noisy)
    mae = mean_absolute_error(y_reg_true, y_noisy)
    
    interpretation = ""
    if noise <= 0.5:
        interpretation = "Good predictions"
    elif noise <= 1.0:
        interpretation = "Moderate noise"
    elif noise <= 2.0:
        interpretation = "High noise"
    else:
        interpretation = "Very high noise"
    
    print(f"{noise:>10.1f}\t{mse:8.4f}\t{mae:8.4f}\t{interpretation}")

print("\n" + "-" * 60)
print("KEY INSIGHTS FOR BEGINNERS:")
print("-" * 60)
print("1. MSE grows quadratically with noise, MAE grows linearly")
print("2. Use MSE when large errors are particularly bad (safety-critical)")
print("3. Use MAE when errors are equally bad regardless of size")
print("4. Always visualize your metrics - one number never tells the whole story!")
print("5. Different problems need different metrics - think about business impact")
print("\nNext steps: Try changing the random seed, adding more features,")
print("or implementing your own simple linear regression to see these in action!")