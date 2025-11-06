# exercise-11-weight-decay.py
"""
Exercise 11 — Regularization I: L2 weight decay & early stopping
Core Concept: Weight decay and early stopping as forms of regularization.
Task: Train logistic models with/without L2 decay and implement early stopping.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set reproducible seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

class EarlyStopping:
    """Early stopping implementation with patience"""
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_weights(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_weights(model)
            self.counter = 0
            
    def save_weights(self, model):
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()
            
    def restore_weights(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)

class SimpleLogisticRegression(nn.Module):
    """Simple logistic regression model"""
    def __init__(self, input_dim):
        super(SimpleLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def create_noisy_dataset(n_samples=200, noise_level=0.3):
    """Create a noisy classification dataset"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=5,  # Only 5 informative features
        n_redundant=10,   # 10 redundant features
        n_repeated=5,     # 5 repeated features
        n_clusters_per_class=1,
        flip_y=noise_level,  # Add label noise
        random_state=SEED
    )
    return X, y

def train_model_pytorch(X_train, y_train, X_val, y_val, weight_decay=0.0, use_early_stopping=False):
    """Train model with PyTorch, with optional weight decay and early stopping"""
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = SimpleLogisticRegression(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)
    
    # Early stopping
    early_stopper = EarlyStopping(patience=10, min_delta=0.001) if use_early_stopping else None
    
    # Training loop
    train_losses = []
    val_losses = []
    epochs = 200 if not use_early_stopping else 1000  # Allow more epochs for early stopping
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        train_losses.append(epoch_train_loss / len(train_loader))
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            val_losses.append(val_loss.item())
        
        # Early stopping check
        if use_early_stopping and early_stopper:
            early_stopper(val_losses[-1], model)
            if early_stopper.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                if early_stopper.restore_best_weights:
                    early_stopper.restore_weights(model)
                break
    
    return model, train_losses, val_losses

def train_model_sklearn(X_train, y_train, X_val, y_val, C=1.0):
    """Train model with scikit-learn (inverse relationship: smaller C = stronger regularization)"""
    model = LogisticRegression(C=C, penalty='l2', solver='liblinear', random_state=SEED, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Calculate losses for comparison
    train_loss = log_loss(y_train, model.predict_proba(X_train))
    val_loss = log_loss(y_val, model.predict_proba(X_val))
    
    return model, train_loss, val_loss

def evaluate_model(model, X_test, y_test, framework='pytorch'):
    """Evaluate model performance"""
    if framework == 'pytorch':
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test)
            predictions = model(X_test_tensor).numpy().flatten()
            y_pred = (predictions > 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)
        predictions = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    test_loss = log_loss(y_test, predictions)
    
    return accuracy, test_loss

def plot_results_comparison(results):
    """Plot comparison of different regularization strategies"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    models = list(results.keys())
    train_acc = [results[model]['train_accuracy'] for model in models]
    test_acc = [results[model]['test_accuracy'] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, train_acc, width, label='Train Accuracy', alpha=0.7)
    axes[0, 0].bar(x + width/2, test_acc, width, label='Test Accuracy', alpha=0.7)
    axes[0, 0].set_xlabel('Model')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Training vs Test Accuracy')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss comparison
    train_loss = [results[model]['train_loss'] for model in models]
    test_loss = [results[model]['test_loss'] for model in models]
    
    axes[0, 1].bar(x - width/2, train_loss, width, label='Train Loss', alpha=0.7)
    axes[0, 1].bar(x + width/2, test_loss, width, label='Test Loss', alpha=0.7)
    axes[0, 1].set_xlabel('Model')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training vs Test Loss')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models, rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Weight norms comparison
    weight_norms = [results[model]['weight_norm'] for model in models]
    axes[1, 0].bar(models, weight_norms, alpha=0.7)
    axes[1, 0].set_xlabel('Model')
    axes[1, 0].set_ylabel('L2 Norm of Weights')
    axes[1, 0].set_title('Model Complexity (Weight Norm)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Generalization gap
    generalization_gap = [train_acc[i] - test_acc[i] for i in range(len(models))]
    axes[1, 1].bar(models, generalization_gap, alpha=0.7, color='red')
    axes[1, 1].set_xlabel('Model')
    axes[1, 1].set_ylabel('Accuracy Gap (Train - Test)')
    axes[1, 1].set_title('Generalization Gap')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    print("Exercise 11: Regularization with L2 Weight Decay & Early Stopping")
    print("=" * 60)
    
    # Create noisy dataset
    print("Creating noisy dataset...")
    X, y = create_noisy_dataset(n_samples=500, noise_level=0.3)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED)
    
    print(f"Dataset shapes: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
    print(f"Label distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")
    print()
    
    results = {}
    
    # 1. Baseline model (no regularization)
    print("1. Training baseline model (no regularization)...")
    model_baseline, train_losses_base, val_losses_base = train_model_pytorch(
        X_train, y_train, X_val, y_val, weight_decay=0.0, use_early_stopping=False
    )
    train_acc_base, train_loss_base = evaluate_model(model_baseline, X_train, y_train)
    test_acc_base, test_loss_base = evaluate_model(model_baseline, X_test, y_test)
    
    # Get weight norm for baseline
    with torch.no_grad():
        weight_norm_base = torch.norm(model_baseline.linear.weight).item()
    
    results['Baseline'] = {
        'train_accuracy': train_acc_base,
        'test_accuracy': test_acc_base,
        'train_loss': train_loss_base,
        'test_loss': test_loss_base,
        'weight_norm': weight_norm_base
    }
    
    # 2. Model with L2 weight decay
    print("2. Training model with L2 weight decay (λ=0.1)...")
    model_l2, train_losses_l2, val_losses_l2 = train_model_pytorch(
        X_train, y_train, X_val, y_val, weight_decay=0.1, use_early_stopping=False
    )
    train_acc_l2, train_loss_l2 = evaluate_model(model_l2, X_train, y_train)
    test_acc_l2, test_loss_l2 = evaluate_model(model_l2, X_test, y_test)
    
    with torch.no_grad():
        weight_norm_l2 = torch.norm(model_l2.linear.weight).item()
    
    results['L2 Weight Decay'] = {
        'train_accuracy': train_acc_l2,
        'test_accuracy': test_acc_l2,
        'train_loss': train_loss_l2,
        'test_loss': test_loss_l2,
        'weight_norm': weight_norm_l2
    }
    
    # 3. Model with early stopping
    print("3. Training model with early stopping...")
    model_es, train_losses_es, val_losses_es = train_model_pytorch(
        X_train, y_train, X_val, y_val, weight_decay=0.0, use_early_stopping=True
    )
    train_acc_es, train_loss_es = evaluate_model(model_es, X_train, y_train)
    test_acc_es, test_loss_es = evaluate_model(model_es, X_test, y_test)
    
    with torch.no_grad():
        weight_norm_es = torch.norm(model_es.linear.weight).item()
    
    results['Early Stopping'] = {
        'train_accuracy': train_acc_es,
        'test_accuracy': test_acc_es,
        'train_loss': train_loss_es,
        'test_loss': test_loss_es,
        'weight_norm': weight_norm_es
    }
    
    # 4. Model with both L2 and early stopping
    print("4. Training model with both L2 and early stopping...")
    model_both, train_losses_both, val_losses_both = train_model_pytorch(
        X_train, y_train, X_val, y_val, weight_decay=0.1, use_early_stopping=True
    )
    train_acc_both, train_loss_both = evaluate_model(model_both, X_train, y_train)
    test_acc_both, test_loss_both = evaluate_model(model_both, X_test, y_test)
    
    with torch.no_grad():
        weight_norm_both = torch.norm(model_both.linear.weight).item()
    
    results['L2 + Early Stopping'] = {
        'train_accuracy': train_acc_both,
        'test_accuracy': test_acc_both,
        'train_loss': train_loss_both,
        'test_loss': test_loss_both,
        'weight_norm': weight_norm_both
    }
    
    # 5. scikit-learn model with strong regularization
    print("5. Training scikit-learn model with strong L2 regularization...")
    model_sklearn, train_loss_sk, val_loss_sk = train_model_sklearn(X_train, y_train, X_val, y_val, C=0.1)
    train_acc_sk = accuracy_score(y_train, model_sklearn.predict(X_train))
    test_acc_sk = accuracy_score(y_test, model_sklearn.predict(X_test))
    test_loss_sk = log_loss(y_test, model_sklearn.predict_proba(X_test))
    
    weight_norm_sk = np.linalg.norm(model_sklearn.coef_)
    
    results['sklearn L2 (C=0.1)'] = {
        'train_accuracy': train_acc_sk,
        'test_accuracy': test_acc_sk,
        'train_loss': train_loss_sk,
        'test_loss': test_loss_sk,
        'weight_norm': weight_norm_sk
    }
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Model':<25} {'Train Acc':<10} {'Test Acc':<10} {'Train Loss':<12} {'Test Loss':<12} {'Weight Norm':<12}")
    print("-" * 80)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<25} {metrics['train_accuracy']:<10.4f} {metrics['test_accuracy']:<10.4f} "
              f"{metrics['train_loss']:<12.4f} {metrics['test_loss']:<12.4f} {metrics['weight_norm']:<12.4f}")
    
    print("\nKEY OBSERVATIONS:")
    print("• Regularization reduces overfitting (smaller generalization gap)")
    print("• L2 weight decay produces smaller weight norms")
    print("• Early stopping prevents over-training")
    print("• Combined approaches often work best")
    
    # Plot training curves for PyTorch models
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses_base, label='Baseline Train')
    plt.plot(val_losses_base, label='Baseline Val', linestyle='--')
    plt.plot(train_losses_l2, label='L2 Train')
    plt.plot(val_losses_l2, label='L2 Val', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves: Baseline vs L2 Regularization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses_es, label='Early Stop Train')
    plt.plot(val_losses_es, label='Early Stop Val', linestyle='--')
    plt.plot(train_losses_both, label='L2+ES Train')
    plt.plot(val_losses_both, label='L2+ES Val', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves: Early Stopping Methods')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Plot final comparison
    plot_results_comparison(results)

if __name__ == "__main__":
    main()