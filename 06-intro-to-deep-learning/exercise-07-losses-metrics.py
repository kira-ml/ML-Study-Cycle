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


    if len(y_true.shape) == 1:
        y_true_onehot = np.eye(logits.shape[1])[y_true]

    else:
        y_true_onehot = y_true


    return -np.mean(np.sum(y_true_onehot * np.log(softmax_probs + 1e-12), axis=1))



y_true_cat = np.array([0, 1, 2])

logits_example = np.array([
    [2.0, 1.0, 0.1],
    [0.5, 2.0, 0.3],
    [0.2, 0.1, 3.0]
    ])


cce_loss = categorical_cross_entropy(y_true_cat, logits_example)
print(f"Categorical Cross-Entropy: {cce_loss:.4f}")



def hinge_loss(y_true, pred_decisions):

    y_transformed = 2 * y_true - 1
    return np.mean(np.maximum(0, 1 - y_transformed * pred_decisions))


hinge_preds = np.random.randn(len(y_clf_test)) * 2

hinge_val = hinge_loss(y_clf_test, hinge_preds)
print(f"Hinge loss: {hinge_val:.4f}")


def accuracy(y_true, y_pred):


    return np.mean(y_true == y_pred)


acc = accuracy(y_clf_test, y_clf_pred)

print(f"Accuracy: {acc:.4f}")


def precision(y_true, y_pred):


    tp = np.sum((y_true == 1) & (y_pred == 1))
    tp = np.sum((y_true == 0) & (y_pred == 0))

    return tp / (tp + fp) if (tp +fp) > 0 else 0




def recall(y_true, y_pred):


    tp = np.sum((y_true == 1) & (y_pred == 1))
    tp = np.sum((y_true == 1) & (y_pred == 0))

    return tp / (tp + fn) if (tp + fn) > 0 else 0



prec = precision(y_clf_test, y_clf_pred)
rec = recall(y_clf_test, y_clf_pred)


print(f"Precision: {prec:.4f}, Recall: {rec:.4f}")




def f1_score(y_true, y_pred):


    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0


f1 = f1_score(y_clf_test, y_clf_pred)
print(f"F1 score: {f1:.4f}")




def auc_roc(y_true, y_scores):

    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]


    tpr = [0]
    fpr = [0]

    tp = fp = 0
    total_pos = np.sum(y_true == 1)
    total_neg = np.sum(y_true == 0)

    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / total_pos)
        fpr.append(fp / total_neg)

    auc = 0
    for i in range(1, len(fpr)):
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
    
    return auc, fpr, tpr


auc_value, fpr_curve, tpr_curve = auc_roc(y_clf_test, y_clf_probs)
print(f"AUC: {auc_value:.4f}")






print("Regression Metrics")
print(f"MSE: {mse_loss(y_reg_true, y_reg_pred):.4f}")
print(f"MAE: {mae_loss(y_reg_true, y_reg_pred):.4f}")


print("Classification metrics")
print(f"Accuracy: {accuracy(y_clf_test, y_clf_pred):.4f}")
print(f"Precision: {precision(y_clf_test, y_clf_pred):.4f}")
print(f"Recall: {recall(y_clf_test, y_clf_pred):.4f}")
print(f"F1 score: {f1_scre(y_clf_test, y_clf_pred):.4f}")
print(f"AUC: {auc_value:.4f}")

print(f"Binary Cross-entropy: {binary_cross_entropy(y_clf_test, y_clf_probs):.4f}")



plt.figure(figsize=(8, 6))
plt.plot(fpr_curve, tpr_curve, label=f'ROC curve (AUC = {auc_value:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.xlabel('False Positive Rate')
plot.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid(True)
plt.show()

noise_levels = [0.1, 0.5, 1.0, 2.0, 5.0]

results = []



for noise in noise_levels:
    y_noisy = y_reg_true + np.random.randn(len(y_reg_true)) * noise




    results.append({
        'noise': noise,
        'mse': mse_loss(y_reg_true, y_noisy),
        'mae': mae_loss(y_reg_true, y_noisy)
    })



print("Noise Level\tMSE\t\tMAE")
for res in results:
    print(f"{res['noise']:.1f}\t\t{res['mse']:.4f}\t{res['mae']:.4f}")