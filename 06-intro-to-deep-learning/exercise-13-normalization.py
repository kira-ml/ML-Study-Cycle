"""
Exercise 13 — Batch normalization & layer normalization
Core Concept: Internal covariate shift, normalization layers, and training vs inference behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from typing import Tuple, Optional

# Set random seeds for reproducibility
np.random.seed(42)

class BatchNorm1d:
    """Batch Normalization layer implementation from scratch"""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        """
        Initialize BatchNorm1d layer
        
        Args:
            num_features: Number of features/channels
            eps: Small constant for numerical stability
            momentum: Momentum for running mean and variance
        """
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = np.ones((1, num_features))  # Scale
        self.beta = np.zeros((1, num_features))  # Shift
        
        # Running statistics (for inference)
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))
        
        # Cache for backward pass
        self.cache = {}
        self.training = True
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for batch normalization
        
        Args:
            x: Input tensor of shape (batch_size, num_features)
            
        Returns:
            Normalized output tensor
        """
        if self.training:
            # Training mode: use batch statistics
            batch_mean = np.mean(x, axis=0, keepdims=True)
            batch_var = np.var(x, axis=0, keepdims=True)
            
            # Normalize
            x_normalized = (x - batch_mean) / np.sqrt(batch_var + self.eps)
            
            # Scale and shift
            out = self.gamma * x_normalized + self.beta
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            # Store for backward pass
            self.cache = {
                'x': x,
                'x_normalized': x_normalized,
                'batch_mean': batch_mean,
                'batch_var': batch_var,
                'std': np.sqrt(batch_var + self.eps)
            }
        else:
            # Inference mode: use running statistics
            x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_normalized + self.beta
        
        return out
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass for batch normalization
        
        Args:
            dout: Gradient from subsequent layer
            
        Returns:
            Gradient w.r.t. input
        """
        if not self.training:
            raise ValueError("Backward pass should only be called during training")
        
        # Unpack cache
        x, x_normalized, batch_mean, batch_var, std = (
            self.cache['x'], self.cache['x_normalized'], 
            self.cache['batch_mean'], self.cache['batch_var'], 
            self.cache['std']
        )
        
        batch_size = dout.shape[0]
        
        # Gradients for gamma and beta
        self.dgamma = np.sum(dout * x_normalized, axis=0, keepdims=True)
        self.dbeta = np.sum(dout, axis=0, keepdims=True)
        
        # Gradient for input (simplified version)
        dx_normalized = dout * self.gamma
        dvar = np.sum(dx_normalized * (x - batch_mean) * -0.5 * (std ** -3), axis=0, keepdims=True)
        dmean = np.sum(dx_normalized * -1 / std, axis=0, keepdims=True) + dvar * np.mean(-2 * (x - batch_mean), axis=0, keepdims=True)
        
        dx = dx_normalized / std + dvar * 2 * (x - batch_mean) / batch_size + dmean / batch_size
        
        return dx
    
    def update_parameters(self, learning_rate: float):
        """Update learnable parameters"""
        self.gamma -= learning_rate * self.dgamma
        self.beta -= learning_rate * self.dbeta
    
    def train(self):
        """Set layer to training mode"""
        self.training = True
    
    def eval(self):
        """Set layer to evaluation mode"""
        self.training = False

class LayerNorm:
    """Layer Normalization implementation from scratch"""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        """
        Initialize LayerNorm layer
        
        Args:
            normalized_shape: Shape of features to normalize
            eps: Small constant for numerical stability
        """
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Learnable parameters
        self.gamma = np.ones((1, normalized_shape))
        self.beta = np.zeros((1, normalized_shape))
        
        # Cache for backward pass
        self.cache = {}
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for layer normalization
        
        Args:
            x: Input tensor of shape (batch_size, ..., features)
            
        Returns:
            Normalized output tensor
        """
        # Compute statistics along the last dimension
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        # Normalize
        x_normalized = (x - mean) / np.sqrt(var + self.eps)
        
        # Scale and shift
        out = self.gamma * x_normalized + self.beta
        
        # Store for backward pass
        self.cache = {
            'x': x,
            'x_normalized': x_normalized,
            'mean': mean,
            'var': var,
            'std': np.sqrt(var + self.eps)
        }
        
        return out
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass for layer normalization
        
        Args:
            dout: Gradient from subsequent layer
            
        Returns:
            Gradient w.r.t. input
        """
        # Unpack cache
        x, x_normalized, mean, var, std = (
            self.cache['x'], self.cache['x_normalized'],
            self.cache['mean'], self.cache['var'], 
            self.cache['std']
        )
        
        n = self.normalized_shape
        
        # Gradients for gamma and beta
        self.dgamma = np.sum(dout * x_normalized, axis=0, keepdims=True)
        self.dbeta = np.sum(dout, axis=0, keepdims=True)
        
        # Gradient for input
        dx_normalized = dout * self.gamma
        dvar = np.sum(dx_normalized * (x - mean) * -0.5 * (std ** -3), axis=-1, keepdims=True)
        dmean = np.sum(dx_normalized * -1 / std, axis=-1, keepdims=True) + dvar * np.mean(-2 * (x - mean), axis=-1, keepdims=True)
        
        dx = dx_normalized / std + dvar * 2 * (x - mean) / n + dmean / n
        
        return dx
    
    def update_parameters(self, learning_rate: float):
        """Update learnable parameters"""
        self.gamma -= learning_rate * self.dgamma
        self.beta -= learning_rate * self.dbeta

class NormalizedMLP:
    """MLP with optional normalization layers"""
    
    def __init__(self, layer_sizes: list, norm_type: str = None, norm_position: str = 'after'):
        """
        Initialize MLP with normalization
        
        Args:
            layer_sizes: List of layer sizes [input, hidden1, ..., output]
            norm_type: 'batch', 'layer', or None
            norm_position: 'before' or 'after' activation
        """
        self.layer_sizes = layer_sizes
        self.norm_type = norm_type
        self.norm_position = norm_position
        self.L = len(layer_sizes) - 1
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self.norm_layers = []
        
        for i in range(self.L):
            # He initialization for ReLU
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(W)
            self.biases.append(b)
            
            # Add normalization layers for hidden layers
            if i < self.L - 1 and norm_type:
                if norm_type == 'batch':
                    self.norm_layers.append(BatchNorm1d(layer_sizes[i+1]))
                elif norm_type == 'layer':
                    self.norm_layers.append(LayerNorm(layer_sizes[i+1]))
        
        # Cache for training
        self.cache = {}
        self.training = True
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU"""
        return (x > 0).astype(float)
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network"""
        self.cache['a0'] = x
        a = x
        
        norm_idx = 0
        for i in range(self.L - 1):
            # Linear transformation
            z = np.dot(a, self.weights[i]) + self.biases[i]
            self.cache[f'z{i+1}'] = z
            
            # Normalization and activation
            if self.norm_type and norm_idx < len(self.norm_layers):
                if self.norm_position == 'before':
                    z_norm = self.norm_layers[norm_idx].forward(z)
                    a = self.relu(z_norm)
                else:  # 'after'
                    a_relu = self.relu(z)
                    a = self.norm_layers[norm_idx].forward(a_relu)
                norm_idx += 1
            else:
                a = self.relu(z)
            
            self.cache[f'a{i+1}'] = a
        
        # Output layer (no normalization)
        z_out = np.dot(a, self.weights[-1]) + self.biases[-1]
        a_out = self.softmax(z_out)
        
        self.cache[f'z{self.L}'] = z_out
        self.cache[f'a{self.L}'] = a_out
        
        return a_out
    
    def backward(self, x: np.ndarray, y: np.ndarray, learning_rate: float = 0.01):
        """Backward pass and parameter update"""
        m = x.shape[0]
        
        # Forward pass to populate cache
        self.forward(x)
        
        # Calculate output gradient
        dz = self.cache[f'a{self.L}'] - y
        
        # Backward through layers
        norm_idx = len(self.norm_layers) - 1
        
        for l in range(self.L, 0, -1):
            if l == self.L:
                # Output layer
                dW = np.dot(self.cache[f'a{l-1}'].T, dz) / m
                db = np.sum(dz, axis=0, keepdims=True) / m
                
                # Update output layer
                self.weights[l-1] -= learning_rate * dW
                self.biases[l-1] -= learning_rate * db
                
                # Gradient for next layer
                da = np.dot(dz, self.weights[l-1].T)
                dz = da * self.relu_derivative(self.cache[f'z{l-1}'])
            else:
                # Hidden layers with normalization
                if self.norm_type and norm_idx >= 0:
                    if self.norm_position == 'before':
                        # Normalization before activation
                        dz_norm = self.norm_layers[norm_idx].backward(dz)
                        dz = dz_norm
                    else:
                        # Normalization after activation
                        da_norm = self.norm_layers[norm_idx].backward(dz)
                        dz = da_norm * self.relu_derivative(self.cache[f'z{l}'])
                    
                    # Update normalization parameters
                    self.norm_layers[norm_idx].update_parameters(learning_rate)
                    norm_idx -= 1
                else:
                    # No normalization
                    dz = dz * self.relu_derivative(self.cache[f'z{l}'])
                
                # Calculate gradients for weights and biases
                dW = np.dot(self.cache[f'a{l-1}'].T, dz) / m
                db = np.sum(dz, axis=0, keepdims=True) / m
                
                # Update parameters
                self.weights[l-1] -= learning_rate * dW
                self.biases[l-1] -= learning_rate * db
                
                # Gradient for next layer
                if l > 1:
                    da = np.dot(dz, self.weights[l-1].T)
                    dz = da * self.relu_derivative(self.cache[f'z{l-1}'])
    
    def train(self):
        """Set network to training mode"""
        self.training = True
        for norm_layer in self.norm_layers:
            if hasattr(norm_layer, 'train'):
                norm_layer.train()
    
    def eval(self):
        """Set network to evaluation mode"""
        self.training = False
        for norm_layer in self.norm_layers:
            if hasattr(norm_layer, 'eval'):
                norm_layer.eval()
    
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions"""
        self.eval()
        probs = self.forward(x)
        preds = np.argmax(probs, axis=1)
        return preds, probs

def generate_data(n_samples: int = 1000, n_features: int = 20) -> Tuple:
    """Generate synthetic classification dataset"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert labels to one-hot encoding
    y_train_onehot = np.eye(3)[y_train]
    y_test_onehot = np.eye(3)[y_test]
    
    return X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot

def train_model(X_train: np.ndarray, y_train: np.ndarray, y_train_onehot: np.ndarray,
                norm_type: str = None, batch_size: int = 32, tiny_batch: bool = False) -> dict:
    """Train MLP with specified normalization"""
    
    if tiny_batch:
        batch_size = 1  # Extreme case to demonstrate BatchNorm issues
    
    print(f"Training model with {norm_type} normalization, batch_size={batch_size}...")
    
    model = NormalizedMLP(
        layer_sizes=[X_train.shape[1], 64, 32, 3],
        norm_type=norm_type,
        norm_position='after'
    )
    
    # Training parameters
    epochs = 100 if not tiny_batch else 200  # More epochs for tiny batches
    learning_rate = 0.01
    
    train_losses = []
    train_accuracies = []
    
    for epoch in range(epochs):
        # Mini-batch training
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train_onehot[indices]
        
        epoch_loss = 0
        batch_count = 0
        
        for i in range(0, len(X_train), batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            if len(X_batch) == 0:
                continue
                
            # Forward pass
            model.train()
            predictions = model.forward(X_batch)
            
            # Calculate cross-entropy loss
            loss = -np.mean(np.sum(y_batch * np.log(predictions + 1e-8), axis=1))
            epoch_loss += loss
            
            # Backward pass and update
            model.backward(X_batch, y_batch, learning_rate)
            batch_count += 1
        
        # Calculate training accuracy
        model.eval()
        train_preds, _ = model.predict(X_train)
        train_acc = np.mean(train_preds == y_train)
        
        avg_loss = epoch_loss / batch_count if batch_count > 0 else epoch_loss
        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {train_acc:.4f}")
    
    return {
        'model': model,
        'losses': train_losses,
        'accuracies': train_accuracies
    }

def demonstrate_batchnorm_issue():
    """Demonstrate BatchNorm issues with tiny batch sizes"""
    print("\n" + "="*60)
    print("DEMONSTRATING BATCHNORM ISSUES WITH TINY BATCH SIZES")
    print("="*60)
    
    # Create a simple dataset
    X = np.random.randn(100, 10)
    
    # Test BatchNorm with different batch sizes
    batch_sizes = [32, 2, 1]
    
    for batch_size in batch_sizes:
        print(f"\nTesting BatchNorm with batch_size = {batch_size}:")
        
        batchnorm = BatchNorm1d(10)
        batchnorm.train()
        
        # Simulate training behavior
        variances = []
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i+batch_size]
            output = batchnorm.forward(X_batch)
            batch_var = np.var(output, axis=0)
            variances.append(np.mean(batch_var))
        
        print(f"  Average output variance: {np.mean(variances):.6f}")
        print(f"  Variance stability: {np.std(variances):.6f}")
        
        if batch_size == 1:
            print("  ⚠️  CRITICAL ISSUE: With batch_size=1:")
            print("     - Cannot compute meaningful batch statistics")
            print("     - Variance estimate is zero (division by zero risk)")
            print("     - No proper normalization occurs")
            print("     - Running statistics become unstable")
            print("  ✅ RECOMMENDATION: Use LayerNorm instead for batch_size=1")

def compare_normalization_methods():
    """Compare different normalization methods"""
    print("\n" + "="*60)
    print("COMPARING NORMALIZATION METHODS")
    print("="*60)
    
    # Generate data
    X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot = generate_data()
    
    # Train models with different normalization
    results = {}
    norm_types = [None, 'batch', 'layer']
    
    for norm_type in norm_types:
        results[norm_type] = train_model(
            X_train, y_train, y_train_onehot, 
            norm_type=norm_type, batch_size=32
        )
    
    # Test with tiny batch sizes
    print("\n" + "-"*40)
    print("TESTING WITH TINY BATCH SIZES")
    print("-"*40)
    
    tiny_results = {}
    for norm_type in ['batch', 'layer']:
        tiny_results[norm_type] = train_model(
            X_train, y_train, y_train_onehot,
            norm_type=norm_type, batch_size=1, tiny_batch=True
        )
    
    return results, tiny_results

def plot_results(results: dict, tiny_results: dict):
    """Plot training results and normalization comparisons"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot training loss comparison
    axes[0, 0].plot(results[None]['losses'], label='No Normalization', linewidth=2)
    axes[0, 0].plot(results['batch']['losses'], label='BatchNorm', linewidth=2)
    axes[0, 0].plot(results['layer']['losses'], label='LayerNorm', linewidth=2)
    axes[0, 0].set_title('Training Loss Comparison (Batch Size=32)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot training accuracy comparison
    axes[0, 1].plot(results[None]['accuracies'], label='No Normalization', linewidth=2)
    axes[0, 1].plot(results['batch']['accuracies'], label='BatchNorm', linewidth=2)
    axes[0, 1].plot(results['layer']['accuracies'], label='LayerNorm', linewidth=2)
    axes[0, 1].set_title('Training Accuracy Comparison (Batch Size=32)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot tiny batch size comparison
    axes[1, 0].plot(tiny_results['batch']['losses'], label='BatchNorm (BS=1)', linewidth=2, color='red')
    axes[1, 0].plot(tiny_results['layer']['losses'], label='LayerNorm (BS=1)', linewidth=2, color='green')
    axes[1, 0].set_title('Training Loss with Tiny Batch Size')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot tiny batch accuracy comparison
    axes[1, 1].plot(tiny_results['batch']['accuracies'], label='BatchNorm (BS=1)', linewidth=2, color='red')
    axes[1, 1].plot(tiny_results['layer']['accuracies'], label='LayerNorm (BS=1)', linewidth=2, color='green')
    axes[1, 1].set_title('Training Accuracy with Tiny Batch Size')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('normalization_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create summary table
    final_accuracies = {}
    for norm_type in results:
        final_accuracies[norm_type] = results[norm_type]['accuracies'][-1]
    
    for norm_type in tiny_results:
        final_accuracies[f"{norm_type}_tiny"] = tiny_results[norm_type]['accuracies'][-1]
    
    print("\n" + "="*60)
    print("FINAL PERFORMANCE SUMMARY")
    print("="*60)
    print(f"No Normalization: {final_accuracies[None]:.4f}")
    print(f"BatchNorm (BS=32): {final_accuracies['batch']:.4f}")
    print(f"LayerNorm (BS=32): {final_accuracies['layer']:.4f}")
    print(f"BatchNorm (BS=1): {final_accuracies['batch_tiny']:.4f}")
    print(f"LayerNorm (BS=1): {final_accuracies['layer_tiny']:.4f}")

def main():
    """Main function to run normalization experiments"""
    
    print("Exercise 13: Batch Normalization & Layer Normalization")
    print("="*60)
    
    # Demonstrate BatchNorm issues
    demonstrate_batchnorm_issue()
    
    # Compare normalization methods
    results, tiny_results = compare_normalization_methods()
    
    # Plot results
    plot_results(results, tiny_results)
    
    print("\n" + "="*60)
    print("KEY INSIGHTS AND RECOMMENDATIONS")
    print("="*60)
    print("1. BatchNorm Issues with Small Batches:")
    print("   - Batch size = 1: Variance estimates become zero")
    print("   - Batch size = 2: High variance in normalization")
    print("   - Solution: Use LayerNorm for small batch sizes")
    print()
    print("2. Training Stability:")
    print("   - Both BatchNorm and LayerNorm improve stability")
    print("   - LayerNorm more robust to batch size variations")
    print("   - BatchNorm slightly better with large batches")
    print()
    print("3. Use Cases:")
    print("   - BatchNorm: Large batches, CNNs, stable training")
    print("   - LayerNorm: RNNs, transformers, small batches")
    print("   - No Norm: Very small networks, simple tasks")

if __name__ == "__main__":
    main()