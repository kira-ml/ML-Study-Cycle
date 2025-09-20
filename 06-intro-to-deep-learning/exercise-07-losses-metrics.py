import numpy as np
import matplotlib.pylot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split




np.random.seed(42)
X_reg = np.random.randn(1000, 5)
true_coeffs = np.array([2.5, -1.3, 0.8, -0.5, 1.2])
y_reg_true = X_reg @ true_coeffs + 0.5 * np.random.randn(1000)
y_reg_pred = y_reg_true + np.randomrandn(1000) * 1.0


X_clf, y_clf_true = make_classification(n_samples=2000, n_features=10, n_informative=5, n_redundant=2, random_state=42)

X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf_true, test_size=0.3, random_state=42
)


logits = X_clf_test @ np.random.randn(10) + np.random.randn(X_clf_test.shape[0]) * 0.5

y_clf_probs = 1 / (1 + np.exp(-logits))
y_clf_pred = (y_clf_probs > 0.5).astype(int)




def mse_loss(y_true, y_pred):

    return np.mean((y_true - y_pred) ** 2)


mse_value = mse_loss(y_reg_true, y_reg_pred)
print(f"MSE: {mse_value:.4f}")




def mae_loss(y_true, y_pred):

    return np.mean(np.abs(y_true - y_pred))



mae_value = mae_loss(y_reg_true, y_reg_pred)
print(f"MAE: {mae_value:.4f}")



def binary_cross_entropy(y_true, y_pred_probs, epsilon=1e-12):


    y_pred_probs = np.clip(y_pred_probs, epsilon, 1 - epsilon)

    return -np.mean(y_true * np.log(y_pred_probs) + (1 - y_true) * np.log(1 - y_pred_probs))


bce_loss = binary_cross_entropy(y_clf_test, y_clf_probs)

print(f"Binary Cross-Entropy: {bce_loss:.4f}")



def categorical_cross_entropy(y_true, logits):


    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)



