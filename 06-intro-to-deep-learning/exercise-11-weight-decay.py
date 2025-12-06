# exercise-11-weight-decay.py
"""
Exercise 11 — Regularization Techniques for Machine Learning
================================================================
Author: kira-ml (GitHub)
Description: A comprehensive tutorial on regularization techniques to prevent overfitting
             in neural networks. This exercise demonstrates L2 weight decay and early 
             stopping with detailed explanations for beginners.

Core Concepts:
1. Regularization - Techniques to prevent overfitting
2. L2 Weight Decay - Adding penalty to large weights during training
3. Early Stopping - Stopping training when validation performance worsens
4. Generalization Gap - Difference between train and test performance

Learning Objectives:
• Understand why overfitting occurs and how to detect it
• Implement L2 regularization (weight decay) in PyTorch
• Implement early stopping with patience mechanism
• Compare different regularization strategies
• Interpret model complexity through weight norms

Prerequisites:
• Basic understanding of neural networks
• Familiarity with training/validation/test splits
• Knowledge of loss functions and optimization

Note: This exercise uses a deliberately noisy dataset to exaggerate overfitting,
      making regularization effects more visible.
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

# ============================================================================
# SETUP: Reproducibility
# ============================================================================
"""
Why set seeds? 
Random initialization affects results. Setting seeds ensures you get the same
results every time you run the code, which is crucial for debugging and fair
comparisons between different models.
"""
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================================
# CLASS 1: EarlyStopping
# ============================================================================
class EarlyStopping:
    """
    Early Stopping: A regularization technique that stops training when
    validation loss stops improving.
    
    How it works:
    1. Monitor validation loss after each epoch
    2. If loss doesn't improve for 'patience' epochs, stop training
    3. Optionally restore the best weights found during training
    
    Parameters:
    -----------
    patience : int
        How many epochs to wait after last improvement
    min_delta : float
        Minimum change to qualify as an improvement
    restore_best_weights : bool
        Whether to restore model weights from the best epoch
    
    Visual Analogy:
    Think of early stopping like a coach stopping practice when the team
    starts getting worse rather than better.
    """
    
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        """
        Check if training should stop based on validation loss.
        
        Parameters:
        -----------
        val_loss : float
            Current validation loss
        model : torch.nn.Module
            The model being trained
        """
        # First epoch initialization
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_weights(model)
        
        # Check if loss improved significantly
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True  # Stop training
        
        # Loss improved - reset counter and update best loss
        else:
            self.best_loss = val_loss
            self.save_weights(model)
            self.counter = 0  # Reset patience counter
            
    def save_weights(self, model):
        """Save model state when we achieve best validation loss"""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()
            
    def restore_weights(self, model):
        """Restore model to best weights found during training"""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)

# ============================================================================
# CLASS 2: SimpleLogisticRegression
# ============================================================================
class SimpleLogisticRegression(nn.Module):
    """
    A simple logistic regression model implemented in PyTorch.
    
    Logistic Regression:
    • Linear model for binary classification
    • Output is passed through sigmoid to get probabilities
    • Can be thought of as a single-layer neural network
    
    Architecture:
    Input → Linear Layer → Sigmoid → Output (0 to 1 probability)
    
    Mathematical Form:
    y = σ(w·x + b) where σ is sigmoid: σ(z) = 1 / (1 + e^{-z})
    """
    
    def __init__(self, input_dim):
        """
        Initialize the model.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        """
        super(SimpleLogisticRegression, self).__init__()
        # Single linear layer: maps input_dim features to 1 output
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim)
            
        Returns:
        --------
        torch.Tensor
            Probability scores between 0 and 1
        """
        # Apply linear transformation followed by sigmoid
        return torch.sigmoid(self.linear(x))

# ============================================================================
# DATASET CREATION
# ============================================================================
def create_noisy_dataset(n_samples=200, noise_level=0.3):
    """
    Create a challenging classification dataset designed to cause overfitting.
    
    Why a noisy dataset?
    • Clean datasets don't show overfitting clearly
    • Real-world data often has noise and redundancy
    • Noisy data exaggerates overfitting, making regularization effects visible
    
    Dataset Characteristics:
    • 20 total features (only 5 are actually informative)
    • 10 redundant features (linear combinations of informative ones)
    • 5 repeated features (copies of other features)
    • Label noise (some labels are intentionally wrong)
    
    This creates a scenario where the model can easily memorize noise,
    making regularization essential.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=5,    # Only 5 features actually matter for prediction
        n_redundant=10,     # 10 features are linear combinations (redundant)
        n_repeated=5,       # 5 features are exact copies (repeated)
        n_clusters_per_class=1,
        flip_y=noise_level, # Randomly flip labels to add noise
        random_state=SEED
    )
    return X, y

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_model_pytorch(X_train, y_train, X_val, y_val, weight_decay=0.0, use_early_stopping=False):
    """
    Train a logistic regression model with PyTorch.
    
    Key Training Concepts:
    • Batch training: Process data in small batches (32 samples here)
    • Loss function: Binary Cross-Entropy (BCELoss) for classification
    • Optimizer: Adam with optional weight_decay parameter
    • Epochs: One pass through the entire training dataset
    
    Regularization Parameters:
    • weight_decay: L2 regularization strength (λ in textbooks)
      Higher values = stronger regularization
    • use_early_stopping: Whether to use early stopping
    
    Note: In PyTorch, weight_decay in Adam optimizer = L2 regularization
    """
    
    # Convert to PyTorch tensors (required format for PyTorch)
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1)
    
    # Create DataLoader for batch training
    # Why batches? More efficient memory use and better gradient estimates
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model, loss function, and optimizer
    model = SimpleLogisticRegression(X_train.shape[1])
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)
    
    # Setup early stopping if requested
    early_stopper = EarlyStopping(patience=10, min_delta=0.001) if use_early_stopping else None
    
    # Training history tracking
    train_losses = []
    val_losses = []
    
    # Training loop
    epochs = 200 if not use_early_stopping else 1000  # More epochs for early stopping
    for epoch in range(epochs):
        # ---------- TRAINING PHASE ----------
        model.train()  # Set model to training mode (enables dropout, batch norm updates)
        epoch_train_loss = 0
        
        for batch_X, batch_y in train_loader:
            # Zero the gradients (important! gradients accumulate by default)
            optimizer.zero_grad()
            
            # Forward pass: compute predictions
            outputs = model(batch_X)
            
            # Compute loss
            loss = criterion(outputs, batch_y)
            
            # Backward pass: compute gradients
            loss.backward()
            
            # Update weights: gradient descent step
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        # Average loss for this epoch
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # ---------- VALIDATION PHASE ----------
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation for validation
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            val_losses.append(val_loss.item())
        
        # ---------- EARLY STOPPING CHECK ----------
        if use_early_stopping and early_stopper:
            early_stopper(val_losses[-1], model)
            if early_stopper.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                if early_stopper.restore_best_weights:
                    early_stopper.restore_weights(model)
                break
    
    return model, train_losses, val_losses

def train_model_sklearn(X_train, y_train, X_val, y_val, C=1.0):
    """
    Train a logistic regression model using scikit-learn.
    
    Important Note on scikit-learn vs PyTorch parameter naming:
    • PyTorch: weight_decay = λ (L2 regularization strength)
    • scikit-learn: C = 1/λ (inverse regularization strength)
    
    So:
    • Large C in sklearn = Small weight_decay in PyTorch = Weak regularization
    • Small C in sklearn = Large weight_decay in PyTorch = Strong regularization
    
    Example equivalence:
    • weight_decay=0.1 in PyTorch ≈ C=10 in sklearn (since C = 1/λ)
    """
    model = LogisticRegression(
        C=C,                # Inverse regularization strength
        penalty='l2',       # L2 regularization (weight decay)
        solver='liblinear', # Good for small datasets
        random_state=SEED,
        max_iter=1000
    )
    model.fit(X_train, y_train)
    
    # Calculate losses for comparison with PyTorch models
    train_loss = log_loss(y_train, model.predict_proba(X_train))
    val_loss = log_loss(y_val, model.predict_proba(X_val))
    
    return model, train_loss, val_loss

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================
def evaluate_model(model, X_test, y_test, framework='pytorch'):
    """
    Evaluate model performance on test data.
    
    Metrics Computed:
    1. Accuracy: Percentage of correct predictions
    2. Log Loss: Measures confidence of predictions
       (Lower is better, penalizes wrong confident predictions heavily)
    
    The log loss formula:
    L = -1/N * Σ [y_i * log(p_i) + (1-y_i) * log(1-p_i)]
    where p_i is the predicted probability for class 1
    """
    if framework == 'pytorch':
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test)
            predictions = model(X_test_tensor).numpy().flatten()
            y_pred = (predictions > 0.5).astype(int)  # Convert probabilities to binary
    else:
        y_pred = model.predict(X_test)
        predictions = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    test_loss = log_loss(y_test, predictions)
    
    return accuracy, test_loss

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def plot_results_comparison(results):
    """
    Create comprehensive visualization comparing different models.
    
    What Each Plot Shows:
    1. Train vs Test Accuracy: How well models generalize
    2. Train vs Test Loss: How confident predictions are
    3. Weight Norm: Model complexity (lower = simpler model)
    4. Generalization Gap: Difference between train and test performance
    
    Interpretation Guide:
    • Good model: High test accuracy, small generalization gap
    • Overfit model: High train accuracy but low test accuracy
    • Underfit model: Low accuracy on both train and test
    • Good regularization: Small weight norms, small generalization gap
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Accuracy Comparison
    models = list(results.keys())
    train_acc = [results[model]['train_accuracy'] for model in models]
    test_acc = [results[model]['test_accuracy'] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, train_acc, width, label='Train Accuracy', alpha=0.7)
    axes[0, 0].bar(x + width/2, test_acc, width, label='Test Accuracy', alpha=0.7)
    axes[0, 0].set_xlabel('Model')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Training vs Test Accuracy\n(Large gaps indicate overfitting)')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Loss Comparison
    train_loss = [results[model]['train_loss'] for model in models]
    test_loss = [results[model]['test_loss'] for model in models]
    
    axes[0, 1].bar(x - width/2, train_loss, width, label='Train Loss', alpha=0.7)
    axes[0, 1].bar(x + width/2, test_loss, width, label='Test Loss', alpha=0.7)
    axes[0, 1].set_xlabel('Model')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training vs Test Loss\n(High test loss = poor generalization)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models, rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Weight Norms (Model Complexity)
    weight_norms = [results[model]['weight_norm'] for model in models]
    axes[1, 0].bar(models, weight_norms, alpha=0.7)
    axes[1, 0].set_xlabel('Model')
    axes[1, 0].set_ylabel('L2 Norm of Weights')
    axes[1, 0].set_title('Model Complexity (Weight Norm)\n(Lower = simpler model)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Generalization Gap
    generalization_gap = [train_acc[i] - test_acc[i] for i in range(len(models))]
    axes[1, 1].bar(models, generalization_gap, alpha=0.7, color='red')
    axes[1, 1].set_xlabel('Model')
    axes[1, 1].set_ylabel('Accuracy Gap (Train - Test)')
    axes[1, 1].set_title('Generalization Gap\n(Smaller is better)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """
    Main execution function demonstrating regularization techniques.
    
    Experiment Design:
    We train 5 different models to compare:
    1. Baseline: No regularization (expected to overfit)
    2. L2 Weight Decay: Regularization via weight penalty
    3. Early Stopping: Regularization via stopping early
    4. Combined: Both L2 and early stopping
    5. sklearn L2: Reference implementation from scikit-learn
    
    Expected Results:
    • Baseline should show largest generalization gap
    • Regularized models should have better test performance
    • Combined approach often works best
    • Regularized models should have smaller weight norms
    """
    
    print("=" * 70)
    print("EXERCISE 11: REGULARIZATION TECHNIQUES FOR MACHINE LEARNING")
    print("Author: kira-ml (GitHub)")
    print("=" * 70)
    print("\nWelcome! This exercise demonstrates how to prevent overfitting")
    print("using L2 weight decay and early stopping.\n")
    
    # Step 1: Create and Split Dataset
    print("STEP 1: Creating a noisy dataset designed to cause overfitting...")
    print("Why noisy data? It exaggerates overfitting, making regularization effects clear.")
    X, y = create_noisy_dataset(n_samples=500, noise_level=0.3)
    
    # Split data into train/validation/test sets
    # Why three splits?
    # • Train: Learn parameters
    # • Validation: Tune hyperparameters (like weight_decay)
    # • Test: Final evaluation (only used once!)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED)
    
    print(f"\nDataset Statistics:")
    print(f"• Training samples:   {X_train.shape[0]}")
    print(f"• Validation samples: {X_val.shape[0]}")
    print(f"• Test samples:       {X_test.shape[0]}")
    print(f"• Features per sample: {X_train.shape[1]}")
    print(f"• Positive samples in train: {np.sum(y_train)}/{len(y_train)}")
    print(f"• Positive samples in test:  {np.sum(y_test)}/{len(y_test)}")
    print("\nNote: Only 5 of 20 features are actually informative.")
    print("      The rest are redundant or repeated to encourage overfitting.")
    
    results = {}
    
    # ========================================================================
    # MODEL 1: Baseline (No Regularization)
    # ========================================================================
    print("\n" + "=" * 70)
    print("MODEL 1: BASELINE (NO REGULARIZATION)")
    print("=" * 70)
    print("Expected behavior: Will overfit to training data.")
    print("Look for: High train accuracy, lower test accuracy, large weight norms.")
    
    model_baseline, train_losses_base, val_losses_base = train_model_pytorch(
        X_train, y_train, X_val, y_val, weight_decay=0.0, use_early_stopping=False
    )
    
    train_acc_base, train_loss_base = evaluate_model(model_baseline, X_train, y_train)
    test_acc_base, test_loss_base = evaluate_model(model_baseline, X_test, y_test)
    
    # Calculate weight norm (measure of model complexity)
    with torch.no_grad():
        weight_norm_base = torch.norm(model_baseline.linear.weight).item()
    
    results['Baseline (No Reg)'] = {
        'train_accuracy': train_acc_base,
        'test_accuracy': test_acc_base,
        'train_loss': train_loss_base,
        'test_loss': test_loss_base,
        'weight_norm': weight_norm_base
    }
    
    print(f"\nResults:")
    print(f"• Training Accuracy:   {train_acc_base:.4f}")
    print(f"• Test Accuracy:       {test_acc_base:.4f}")
    print(f"• Generalization Gap:  {train_acc_base - test_acc_base:.4f}")
    print(f"• Weight Norm (L2):    {weight_norm_base:.4f}")
    
    # ========================================================================
    # MODEL 2: L2 Weight Decay
    # ========================================================================
    print("\n" + "=" * 70)
    print("MODEL 2: L2 WEIGHT DECAY (λ = 0.1)")
    print("=" * 70)
    print("How L2 works: Adds penalty = λ * Σ(w_i²) to the loss function.")
    print("Effect: Prevents weights from growing too large.")
    print("Analogy: Like putting a budget on model complexity.")
    
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
    
    print(f"\nResults:")
    print(f"• Training Accuracy:   {train_acc_l2:.4f}")
    print(f"• Test Accuracy:       {test_acc_l2:.4f}")
    print(f"• Generalization Gap:  {train_acc_l2 - test_acc_l2:.4f}")
    print(f"• Weight Norm (L2):    {weight_norm_l2:.4f}")
    print(f"• Weight Reduction:    {((weight_norm_base - weight_norm_l2)/weight_norm_base)*100:.1f}%")
    
    # ========================================================================
    # MODEL 3: Early Stopping
    # ========================================================================
    print("\n" + "=" * 70)
    print("MODEL 3: EARLY STOPPING")
    print("=" * 70)
    print("How it works: Stops training when validation loss stops improving.")
    print("Advantage: No extra hyperparameters to tune (unlike λ in L2).")
    print("Disadvantage: Requires validation set and patience parameter.")
    
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
    
    print(f"\nResults:")
    print(f"• Training Accuracy:   {train_acc_es:.4f}")
    print(f"• Test Accuracy:       {test_acc_es:.4f}")
    print(f"• Generalization Gap:  {train_acc_es - test_acc_es:.4f}")
    print(f"• Weight Norm (L2):    {weight_norm_es:.4f}")
    print(f"• Training Epochs:     {len(train_losses_es)} (stopped early)")
    
    # ========================================================================
    # MODEL 4: Combined L2 + Early Stopping
    # ========================================================================
    print("\n" + "=" * 70)
    print("MODEL 4: COMBINED L2 + EARLY STOPPING")
    print("=" * 70)
    print("Combining multiple regularization techniques often works best.")
    print("L2 prevents large weights, early stopping prevents over-training.")
    
    model_both, train_losses_both, val_losses_both = train_model_pytorch(
        X_train, y_train, X_val, y_val, weight_decay=0.1, use_early_stopping=True
    )
    
    train_acc_both, train_loss_both = evaluate_model(model_both, X_train, y_train)
    test_acc_both, test_loss_both = evaluate_model(model_both, X_test, y_test)
    
    with torch.no_grad():
        weight_norm_both = torch.norm(model_both.linear.weight).item()
    
    results['L2 + Early Stop'] = {
        'train_accuracy': train_acc_both,
        'test_accuracy': test_acc_both,
        'train_loss': train_loss_both,
        'test_loss': test_loss_both,
        'weight_norm': weight_norm_both
    }
    
    print(f"\nResults:")
    print(f"• Training Accuracy:   {train_acc_both:.4f}")
    print(f"• Test Accuracy:       {test_acc_both:.4f}")
    print(f"• Generalization Gap:  {train_acc_both - test_acc_both:.4f}")
    print(f"• Weight Norm (L2):    {weight_norm_both:.4f}")
    
    # ========================================================================
    # MODEL 5: scikit-learn Reference
    # ========================================================================
    print("\n" + "=" * 70)
    print("MODEL 5: SCIKIT-LEARN L2 REGULARIZATION (C = 0.1)")
    print("=" * 70)
    print("Note: In sklearn, C = 1/λ, so C=0.1 means strong regularization.")
    print("This serves as a reference implementation.")
    
    model_sklearn, train_loss_sk, val_loss_sk = train_model_sklearn(
        X_train, y_train, X_val, y_val, C=0.1
    )
    
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
    
    print(f"\nResults:")
    print(f"• Training Accuracy:   {train_acc_sk:.4f}")
    print(f"• Test Accuracy:       {test_acc_sk:.4f}")
    print(f"• Generalization Gap:  {train_acc_sk - test_acc_sk:.4f}")
    print(f"• Weight Norm (L2):    {weight_norm_sk:.4f}")
    
    # ========================================================================
    # FINAL RESULTS SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    print("\nKey Metrics Comparison:")
    print(f"{'Model':<25} {'Train Acc':<10} {'Test Acc':<10} {'Train Loss':<12} "
          f"{'Test Loss':<12} {'Weight Norm':<12} {'Gap':<8}")
    print("-" * 80)
    
    for model_name, metrics in results.items():
        gap = metrics['train_accuracy'] - metrics['test_accuracy']
        print(f"{model_name:<25} {metrics['train_accuracy']:<10.4f} "
              f"{metrics['test_accuracy']:<10.4f} {metrics['train_loss']:<12.4f} "
              f"{metrics['test_loss']:<12.4f} {metrics['weight_norm']:<12.4f} "
              f"{gap:<8.4f}")
    
    # ========================================================================
    # LEARNING INSIGHTS
    # ========================================================================
    print("\n" + "=" * 80)
    print("LEARNING INSIGHTS & INTERPRETATION GUIDE")
    print("=" * 80)
    
    print("\n📊 HOW TO INTERPRET THE RESULTS:")
    print("1. GENERALIZATION GAP (Train Acc - Test Acc):")
    print("   • Large gap (>0.1): Model is overfitting")
    print("   • Small gap (<0.05): Model generalizes well")
    print("   • Negative gap: Possibly underfitting or lucky test split")
    
    print("\n2. WEIGHT NORM (Model Complexity):")
    print("   • Large norms: Complex model that can memorize noise")
    print("   • Small norms: Simple model that learns patterns")
    print("   • Regularization reduces weight norms")
    
    print("\n3. TEST ACCURACY (What matters most):")
    print("   • Higher is better for real-world performance")
    print("   • Compare to baseline to see regularization benefit")
    
    print("\n✅ KEY TAKEAWAYS:")
    print("1. Overfitting happens when models learn noise instead of patterns")
    print("2. Regularization techniques prevent overfitting in different ways:")
    print("   • L2: Penalizes large weights (simpler models)")
    print("   • Early stopping: Prevents over-training")
    print("3. Combined approaches often work best")
    print("4. Always use a validation set to tune regularization strength")
    
    print("\n🔧 PRACTICAL ADVICE:")
    print("1. Start with baseline (no regularization) to establish overfitting")
    print("2. Try L2 first - easier to tune with grid search on λ")
    print("3. Add early stopping if training takes too long")
    print("4. Monitor weight norms - they should decrease with regularization")
    print("5. Use test set ONLY for final evaluation")
    
    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS...")
    print("=" * 80)
    
    # Plot 1: Training Curves Comparison
    print("\n📈 Plot 1: Training Curves Comparison")
    print("   Shows how loss changes during training for different methods")
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses_base, label='Baseline Train', linewidth=2)
    plt.plot(val_losses_base, label='Baseline Val', linestyle='--', linewidth=2)
    plt.plot(train_losses_l2, label='L2 Train', linewidth=2)
    plt.plot(val_losses_l2, label='L2 Val', linestyle='--', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Baseline vs L2 Regularization\n(Notice how L2 prevents overfitting)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses_es, label='Early Stop Train', linewidth=2)
    plt.plot(val_losses_es, label='Early Stop Val', linestyle='--', linewidth=2)
    plt.plot(train_losses_both, label='L2+ES Train', linewidth=2)
    plt.plot(val_losses_both, label='L2+ES Val', linestyle='--', linewidth=2)
    
    # Mark where early stopping happened
    if len(train_losses_es) < 200:
        plt.axvline(x=len(train_losses_es), color='red', linestyle=':', 
                   alpha=0.5, label=f'ES at epoch {len(train_losses_es)}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Early Stopping Methods\n(Stops when validation loss stops improving)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Plot 2: Comprehensive Results Comparison
    print("\n📊 Plot 2: Comprehensive Results Comparison")
    print("   Compares all models across multiple metrics")
    
    plot_results_comparison(results)
    
    # ========================================================================
    # CONCLUSION
    # ========================================================================
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("\n🎯 Regularization is essential for building models that generalize well.")
    print("\n💡 Remember:")
    print("   • Overfitting is learning the noise in your training data")
    print("   • Regularization techniques help models learn patterns instead")
    print("   • Always validate on unseen data")
    print("   • Simpler models (smaller weights) often generalize better")
    print("\n🌟 Happy learning! - kira-ml (GitHub)")
    print("=" * 80)

if __name__ == "__main__":
    main()