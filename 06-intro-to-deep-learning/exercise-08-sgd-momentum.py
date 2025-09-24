"""
Exercise 08 — Optimization basics: implement SGD & momentum
Filename: exercise-08-sgd-momentum.py
"""

import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, n_features):
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -10, 10)))  # Clip to avoid overflow
    
    def forward(self, X):
        """Forward pass: compute predictions"""
        z = X @ self.weights + self.bias
        return self.sigmoid(z)
    
    def compute_loss(self, y_true, y_pred):
        """Binary cross-entropy loss"""
        epsilon = 1e-8  # Avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def compute_gradients(self, X, y_true, y_pred):
        """Compute gradients for weights and bias"""
        m = len(y_true)
        error = y_pred - y_true
        
        dw = (X.T @ error) / m
        db = np.mean(error)
        
        return dw, db

class Optimizers:
    """Collection of optimization algorithms"""
    
    @staticmethod
    def sgd_vanilla(params, grads, learning_rate):
        """Vanilla Stochastic Gradient Descent"""
        for param, grad in zip(params, grads):
            param -= learning_rate * grad
    
    @staticmethod
    def sgd_momentum(params, grads, velocities, learning_rate, momentum=0.9):
        """SGD with Momentum"""
        for i, (param, grad) in enumerate(zip(params, grads)):
            velocities[i] = momentum * velocities[i] - learning_rate * grad
            param += velocities[i]
    
    @staticmethod
    def sgd_nesterov(params, grads, velocities, learning_rate, momentum=0.9):
        """SGD with Nesterov Momentum"""
        for i, (param, grad) in enumerate(zip(params, grads)):
            # Lookahead position
            param_ahead = param + momentum * velocities[i]
            # In practice, we simulate lookahead by using the velocity
            velocities[i] = momentum * velocities[i] - learning_rate * grad
            param += velocities[i]

def generate_synthetic_data(n_samples=1000, separable=True, noise_level=0.1):
    """Generate synthetic binary classification data"""
    np.random.seed(42)
    
    # Generate features from two Gaussian distributions
    n_features = 2
    X1 = np.random.randn(n_samples // 2, n_features) + [1, 1]  # Class 0
    X2 = np.random.randn(n_samples // 2, n_features) + [2, 2]  # Class 1
    
    X = np.vstack([X1, X2])
    y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
    
    if not separable:
        # Add noise to make it non-separable
        flip_indices = np.random.choice(n_samples, int(noise_level * n_samples), replace=False)
        y[flip_indices] = 1 - y[flip_indices]
    
    # Shuffle the data
    indices = np.random.permutation(n_samples)
    X, y = X[indices], y[indices]
    
    return X, y

def train_with_optimizer(optimizer_name, X, y, learning_rate=0.1, momentum=0.9, 
                        n_epochs=100, batch_size=32):
    """Train logistic regression with specified optimizer"""
    n_samples, n_features = X.shape
    model = LogisticRegression(n_features)
    
    # Initialize velocities for momentum-based optimizers
    velocity_w = np.zeros_like(model.weights)
    velocity_b = 0.0
    
    losses = []
    accuracies = []
    
    for epoch in range(n_epochs):
        # Shuffle data each epoch
        indices = np.random.permutation(n_samples)
        X_shuffled, y_shuffled = X[indices], y[indices]
        
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            # Get mini-batch
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Forward pass
            y_pred = model.forward(X_batch)
            batch_loss = model.compute_loss(y_batch, y_pred)
            
            # Backward pass
            dw, db = model.compute_gradients(X_batch, y_batch, y_pred)
            
            # Update parameters using chosen optimizer
            if optimizer_name == "vanilla_sgd":
                Optimizers.sgd_vanilla(
                    [model.weights, model.bias], 
                    [dw, db], 
                    learning_rate
                )
            elif optimizer_name == "momentum":
                Optimizers.sgd_momentum(
                    [model.weights, model.bias], 
                    [dw, db], 
                    [velocity_w, velocity_b], 
                    learning_rate, 
                    momentum
                )
            elif optimizer_name == "nesterov":
                Optimizers.sgd_nesterov(
                    [model.weights, model.bias], 
                    [dw, db], 
                    [velocity_w, velocity_b], 
                    learning_rate, 
                    momentum
                )
            
            epoch_loss += batch_loss
            n_batches += 1
        
        # Calculate metrics for this epoch
        avg_loss = epoch_loss / n_batches if n_batches > 0 else epoch_loss
        losses.append(avg_loss)
        
        # Calculate accuracy
        y_pred_all = model.forward(X)
        predictions = (y_pred_all > 0.5).astype(int)
        accuracy = np.mean(predictions == y)
        accuracies.append(accuracy)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
    
    return losses, accuracies, model

def main():
    """Main function to run experiments"""
    print("=" * 60)
    print("Exercise 08: Optimization Basics - SGD & Momentum")
    print("=" * 60)
    
    # Generate datasets
    print("\n1. Generating synthetic datasets...")
    X_separable, y_separable = generate_synthetic_data(1000, separable=True)
    X_nonseparable, y_nonseparable = generate_synthetic_data(1000, separable=False, noise_level=0.15)
    
    print(f"Separable data shape: {X_separable.shape}")
    print(f"Non-separable data shape: {X_nonseparable.shape}")
    
    # Hyperparameters
    learning_rate = 0.1
    momentum = 0.9
    n_epochs = 100
    batch_size = 32
    
    optimizers = ["vanilla_sgd", "momentum", "nesterov"]
    
    # Experiment 1: Separable data
    print("\n" + "=" * 40)
    print("EXPERIMENT 1: Separable Data")
    print("=" * 40)
    
    results_separable = {}
    
    for optimizer in optimizers:
        print(f"\n--- Training with {optimizer.upper().replace('_', ' ')} ---")
        losses, accuracies, model = train_with_optimizer(
            optimizer, X_separable, y_separable, 
            learning_rate, momentum, n_epochs, batch_size
        )
        results_separable[optimizer] = {
            'losses': losses,
            'accuracies': accuracies,
            'final_accuracy': accuracies[-1],
            'final_loss': losses[-1]
        }
    
    # Experiment 2: Non-separable data
    print("\n" + "=" * 40)
    print("EXPERIMENT 2: Non-separable Data")
    print("=" * 40)
    
    results_nonseparable = {}
    
    for optimizer in optimizers:
        print(f"\n--- Training with {optimizer.upper().replace('_', ' ')} ---")
        losses, accuracies, model = train_with_optimizer(
            optimizer, X_nonseparable, y_nonseparable, 
            learning_rate, momentum, n_epochs, batch_size
        )
        results_nonseparable[optimizer] = {
            'losses': losses,
            'accuracies': accuracies,
            'final_accuracy': accuracies[-1],
            'final_loss': losses[-1]
        }
    
    # Print summary results
    print("\n" + "=" * 60)
    print("SUMMARY RESULTS")
    print("=" * 60)
    
    print("\nSeparable Data Results:")
    print("-" * 40)
    for optimizer in optimizers:
        result = results_separable[optimizer]
        print(f"{optimizer:>12}: Loss = {result['final_loss']:.4f}, Accuracy = {result['final_accuracy']:.4f}")
    
    print("\nNon-separable Data Results:")
    print("-" * 40)
    for optimizer in optimizers:
        result = results_nonseparable[optimizer]
        print(f"{optimizer:>12}: Loss = {result['final_loss']:.4f}, Accuracy = {result['final_accuracy']:.4f}")
    
    # Plot results if matplotlib is available
    try:
        plt.figure(figsize=(12, 8))
        
        # Plot loss curves for separable data
        plt.subplot(2, 2, 1)
        for optimizer in optimizers:
            plt.plot(results_separable[optimizer]['losses'], 
                    label=optimizer.replace('_', ' ').title())
        plt.title('Loss Curves - Separable Data')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot accuracy curves for separable data
        plt.subplot(2, 2, 2)
        for optimizer in optimizers:
            plt.plot(results_separable[optimizer]['accuracies'], 
                    label=optimizer.replace('_', ' ').title())
        plt.title('Accuracy Curves - Separable Data')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot loss curves for non-separable data
        plt.subplot(2, 2, 3)
        for optimizer in optimizers:
            plt.plot(results_nonseparable[optimizer]['losses'], 
                    label=optimizer.replace('_', ' ').title())
        plt.title('Loss Curves - Non-separable Data')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot accuracy curves for non-separable data
        plt.subplot(2, 2, 4)
        for optimizer in optimizers:
            plt.plot(results_nonseparable[optimizer]['accuracies'], 
                    label=optimizer.replace('_', ' ').title())
        plt.title('Accuracy Curves - Non-separable Data')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('exercise-08-optimization-results.png')
        plt.show()
        
    except ImportError:
        print("\nMatplotlib not available. Using textual summaries only.")
        
        # Textual convergence analysis
        print("\nConvergence Analysis (First 5 epochs vs Last 5 epochs):")
        print("-" * 60)
        
        for dataset_name, results in [("Separable", results_separable), 
                                    ("Non-separable", results_nonseparable)]:
            print(f"\n{dataset_name} Data:")
            for optimizer in optimizers:
                losses = results[optimizer]['losses']
                early_avg = np.mean(losses[:5])
                late_avg = np.mean(losses[-5:])
                improvement = early_avg - late_avg
                print(f"  {optimizer:>12}: {improvement:>7.4f} improvement "
                      f"(from {early_avg:.4f} to {late_avg:.4f})")

if __name__ == "__main__":
    main()