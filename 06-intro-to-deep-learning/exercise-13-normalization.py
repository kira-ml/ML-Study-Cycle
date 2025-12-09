"""
Exercise 13 — Batch Normalization & Layer Normalization
Core Concept: Internal covariate shift, normalization layers, and training vs inference behavior.

Author: kira-ml
Repository: https://github.com/kira-ml/open-source-ml-education

This module implements batch normalization and layer normalization from scratch,
demonstrating their effects on neural network training stability and performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List, Optional
import warnings

# Set random seeds for reproducibility across runs
np.random.seed(42)
warnings.filterwarnings('ignore')

class BatchNorm1d:
    """
    1D Batch Normalization Layer (From Scratch)
    
    Batch Normalization addresses "internal covariate shift" - the change in 
    input distribution to internal layers during training. By normalizing 
    activations to zero mean and unit variance per feature across the batch,
    it enables:
    - Faster convergence (higher learning rates)
    - Reduced sensitivity to initialization
    - Mild regularization effect
    
    Key Distinction: Uses batch statistics during training, running statistics during inference.
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        """
        Initialize BatchNorm1d parameters.
        
        Args:
            num_features: Dimensionality of input features (channels in CNN context)
            eps: Small epsilon for numerical stability (prevents division by zero)
            momentum: Controls how quickly running statistics update (0-1 range)
                      Higher = faster adaptation to new data distributions
        
        Conceptual Note:
            gamma (scale) and beta (shift) are learnable parameters that allow
            the network to "undo" normalization if beneficial for the task.
            Without them, normalized outputs would always be N(0,1), limiting
            representational power.
        """
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable affine transformation parameters
        self.gamma = np.ones((1, num_features))  # Scale parameter
        self.beta = np.zeros((1, num_features))   # Shift parameter
        
        # Running statistics (exponentially weighted averages)
        # Used during inference when batch statistics are unavailable/unreliable
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))
        
        # Cache intermediate values for gradient computation during backpropagation
        self.cache = {}
        self.training = True  # Tracks training vs inference mode
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with training/inference mode handling.
        
        Args:
            x: Input tensor of shape (batch_size, num_features)
            
        Returns:
            Normalized, scaled, and shifted output tensor
            
        Mathematical Formulation (Training):
            μ_b = mean(x) across batch dimension
            σ²_b = var(x) across batch dimension
            x̂ = (x - μ_b) / √(σ²_b + ε)  # Normalization
            y = γ * x̂ + β                 # Affine transformation
            
        Why two modes?
            During training: Normalize using current batch statistics
            During inference: Normalize using accumulated running statistics
            This ensures consistent behavior regardless of batch composition.
        """
        if self.training:
            # Training mode: compute statistics from current mini-batch
            batch_mean = np.mean(x, axis=0, keepdims=True)
            batch_var = np.var(x, axis=0, keepdims=True)
            
            # Normalize to zero mean and unit variance
            x_normalized = (x - batch_mean) / np.sqrt(batch_var + self.eps)
            
            # Learnable affine transformation (allows network to control output distribution)
            out = self.gamma * x_normalized + self.beta
            
            # Update running statistics using exponential moving average
            # This provides a smoothed estimate of population statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + \
                               self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + \
                              self.momentum * batch_var
            
            # Cache values needed for backward pass
            self.cache = {
                'x': x,
                'x_normalized': x_normalized,
                'batch_mean': batch_mean,
                'batch_var': batch_var,
                'std': np.sqrt(batch_var + self.eps)
            }
        else:
            # Inference mode: use accumulated running statistics
            # Critical for consistent predictions regardless of batch size/composition
            x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_normalized + self.beta
        
        return out
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass computing gradients w.r.t. inputs and parameters.
        
        Args:
            dout: Upstream gradient from subsequent layer, shape (batch_size, num_features)
            
        Returns:
            Gradient with respect to input x
            
        Implementation Note:
            This follows the gradient derivation from the original BatchNorm paper.
            We compute gradients for gamma, beta, and the input x.
        """
        if not self.training:
            raise ValueError(
                "Backward pass should only be called during training. "
                "Use model.train() before training and model.eval() before inference."
            )
        
        # Retrieve cached values from forward pass
        x = self.cache['x']
        x_normalized = self.cache['x_normalized']
        batch_mean = self.cache['batch_mean']
        batch_var = self.cache['batch_var']
        std = self.cache['std']
        
        batch_size = dout.shape[0]
        
        # Gradients for learnable parameters (gamma and beta)
        # These accumulate across the batch
        self.dgamma = np.sum(dout * x_normalized, axis=0, keepdims=True)
        self.dbeta = np.sum(dout, axis=0, keepdims=True)
        
        # Gradient for input x (simplified implementation)
        # Derived using chain rule through normalization operations
        dx_normalized = dout * self.gamma
        dvar = np.sum(dx_normalized * (x - batch_mean) * -0.5 * (std ** -3), 
                     axis=0, keepdims=True)
        dmean = np.sum(dx_normalized * -1 / std, axis=0, keepdims=True) + \
               dvar * np.mean(-2 * (x - batch_mean), axis=0, keepdims=True)
        
        # Combined gradient for input
        dx = (dx_normalized / std + 
              dvar * 2 * (x - batch_mean) / batch_size + 
              dmean / batch_size)
        
        return dx
    
    def update_parameters(self, learning_rate: float):
        """Update learnable parameters using gradient descent."""
        self.gamma -= learning_rate * self.dgamma
        self.beta -= learning_rate * self.dbeta
    
    def train(self):
        """Switch to training mode (use batch statistics)."""
        self.training = True
    
    def eval(self):
        """Switch to evaluation mode (use running statistics)."""
        self.training = False


class LayerNorm:
    """
    Layer Normalization Layer (From Scratch)
    
    Layer Normalization normalizes across features for each sample independently.
    Unlike BatchNorm, it doesn't depend on batch statistics, making it ideal for:
    - Variable batch sizes (especially batch_size = 1)
    - Recurrent Neural Networks (RNNs)
    - Transformer architectures
    - Online/streaming learning scenarios
    
    Key Insight: Normalization across feature dimension rather than batch dimension
    makes it batch-size invariant.
    """
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        """
        Initialize LayerNorm parameters.
        
        Args:
            normalized_shape: Size of feature dimension to normalize
            eps: Small epsilon for numerical stability
            
        Note: LayerNorm has no running statistics since it doesn't depend on batch.
        """
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Learnable affine parameters (same as BatchNorm)
        self.gamma = np.ones((1, normalized_shape))
        self.beta = np.zeros((1, normalized_shape))
        
        # Cache for backward pass
        self.cache = {}
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for Layer Normalization.
        
        Args:
            x: Input tensor of shape (batch_size, ..., features)
               Last dimension must match normalized_shape
            
        Returns:
            Normalized output tensor
            
        Mathematical Formulation:
            μ = mean(x) across feature dimension (axis=-1)
            σ² = var(x) across feature dimension
            x̂ = (x - μ) / √(σ² + ε)
            y = γ * x̂ + β
        """
        # Compute statistics along the feature dimension (last axis)
        # This makes normalization sample-specific, not batch-dependent
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        # Normalize each sample independently
        x_normalized = (x - mean) / np.sqrt(var + self.eps)
        
        # Affine transformation (preserves representational capacity)
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
        Backward pass for LayerNorm.
        
        Similar to BatchNorm but gradients computed along feature dimension
        instead of batch dimension.
        """
        # Unpack cached values
        x = self.cache['x']
        x_normalized = self.cache['x_normalized']
        mean = self.cache['mean']
        var = self.cache['var']
        std = self.cache['std']
        
        n = self.normalized_shape
        
        # Gradients for learnable parameters
        self.dgamma = np.sum(dout * x_normalized, axis=0, keepdims=True)
        self.dbeta = np.sum(dout, axis=0, keepdims=True)
        
        # Gradient for input (similar to BatchNorm but along feature dimension)
        dx_normalized = dout * self.gamma
        dvar = np.sum(dx_normalized * (x - mean) * -0.5 * (std ** -3), 
                     axis=-1, keepdims=True)
        dmean = np.sum(dx_normalized * -1 / std, axis=-1, keepdims=True) + \
               dvar * np.mean(-2 * (x - mean), axis=-1, keepdims=True)
        
        # Combined gradient
        dx = (dx_normalized / std + 
              dvar * 2 * (x - mean) / n + 
              dmean / n)
        
        return dx
    
    def update_parameters(self, learning_rate: float):
        """Update learnable parameters."""
        self.gamma -= learning_rate * self.dgamma
        self.beta -= learning_rate * self.dbeta


class NormalizedMLP:
    """
    Multi-Layer Perceptron with optional normalization layers.
    
    Demonstrates how normalization layers integrate into a neural network
    architecture and their impact on training dynamics.
    """
    
    def __init__(self, layer_sizes: List[int], norm_type: Optional[str] = None, 
                 norm_position: str = 'after'):
        """
        Initialize MLP with optional normalization.
        
        Args:
            layer_sizes: List of neuron counts per layer [input, hidden1, ..., output]
            norm_type: 'batch', 'layer', or None (no normalization)
            norm_position: 'before' or 'after' activation function
                         Common practice: 'after' for BatchNorm, varies for LayerNorm
        """
        self.layer_sizes = layer_sizes
        self.norm_type = norm_type
        self.norm_position = norm_position
        self.L = len(layer_sizes) - 1  # Number of learnable layers
        
        # Initialize network parameters
        self.weights = []
        self.biases = []
        self.norm_layers = []
        
        # He initialization for ReLU activations (prevents vanishing/exploding gradients)
        for i in range(self.L):
            std_dev = np.sqrt(2.0 / layer_sizes[i])  # He initialization scale
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * std_dev
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(W)
            self.biases.append(b)
            
            # Add normalization layers for hidden layers (not output layer)
            if i < self.L - 1 and norm_type:
                if norm_type == 'batch':
                    self.norm_layers.append(BatchNorm1d(layer_sizes[i+1]))
                elif norm_type == 'layer':
                    self.norm_layers.append(LayerNorm(layer_sizes[i+1]))
        
        # Cache for backpropagation
        self.cache = {}
        self.training = True
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation: max(0, x) - introduces non-linearity."""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU: 1 where x > 0, else 0."""
        return (x > 0).astype(float)
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """
        Softmax activation for multi-class classification.
        
        Numerically stable implementation: subtract max for exponent stability.
        """
        # Shift for numerical stability (prevents overflow)
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted_x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the complete network.
        
        Demonstrates the data flow:
        Input → [Linear → Normalization → Activation] → Output Softmax
        """
        # Cache input for backpropagation
        self.cache['a0'] = x
        a = x
        
        norm_idx = 0
        for i in range(self.L - 1):
            # Linear transformation: Wx + b
            z = np.dot(a, self.weights[i]) + self.biases[i]
            self.cache[f'z{i+1}'] = z
            
            # Apply normalization and activation based on configuration
            if self.norm_type and norm_idx < len(self.norm_layers):
                if self.norm_position == 'before':
                    # Common in some architectures: Normalize → Activate
                    z_norm = self.norm_layers[norm_idx].forward(z)
                    a = self.relu(z_norm)
                else:  # 'after' - Most common for BatchNorm
                    # Activate → Normalize
                    a_relu = self.relu(z)
                    a = self.norm_layers[norm_idx].forward(a_relu)
                norm_idx += 1
            else:
                # No normalization: just activation
                a = self.relu(z)
            
            self.cache[f'a{i+1}'] = a
        
        # Output layer: linear transformation + softmax (no normalization)
        z_out = np.dot(a, self.weights[-1]) + self.biases[-1]
        a_out = self.softmax(z_out)
        
        self.cache[f'z{self.L}'] = z_out
        self.cache[f'a{self.L}'] = a_out
        
        return a_out
    
    def backward(self, x: np.ndarray, y: np.ndarray, learning_rate: float = 0.01):
        """
        Backward pass (backpropagation) with parameter updates.
        
        Computes gradients for all parameters and updates them using
        gradient descent. Handles normalization layers appropriately.
        """
        batch_size = x.shape[0]
        
        # Ensure cache is populated (forward pass already called in training loop)
        # The softmax output is cached as aL
        
        # Gradient at output (softmax + cross-entropy combined gradient)
        # Derivative: dL/dz = predictions - true_labels
        dz = self.cache[f'a{self.L}'] - y
        
        # Backward through layers (reverse order)
        norm_idx = len(self.norm_layers) - 1
        
        for l in range(self.L, 0, -1):
            if l == self.L:
                # Output layer gradients
                dW = np.dot(self.cache[f'a{l-1}'].T, dz) / batch_size
                db = np.sum(dz, axis=0, keepdims=True) / batch_size
                
                # Update output layer parameters
                self.weights[l-1] -= learning_rate * dW
                self.biases[l-1] -= learning_rate * db
                
                # Propagate gradient backward
                da = np.dot(dz, self.weights[l-1].T)
                dz = da * self.relu_derivative(self.cache[f'z{l-1}'])
            else:
                # Hidden layers with optional normalization
                if self.norm_type and norm_idx >= 0:
                    if self.norm_position == 'before':
                        # Gradient through normalization then activation
                        dz_norm = self.norm_layers[norm_idx].backward(dz)
                        dz = dz_norm
                    else:
                        # Gradient through activation then normalization
                        da_norm = self.norm_layers[norm_idx].backward(dz)
                        dz = da_norm * self.relu_derivative(self.cache[f'z{l}'])
                    
                    # Update normalization layer parameters
                    self.norm_layers[norm_idx].update_parameters(learning_rate)
                    norm_idx -= 1
                else:
                    # No normalization: just activation gradient
                    dz = dz * self.relu_derivative(self.cache[f'z{l}'])
                
                # Compute gradients for weights and biases
                dW = np.dot(self.cache[f'a{l-1}'].T, dz) / batch_size
                db = np.sum(dz, axis=0, keepdims=True) / batch_size
                
                # Update parameters
                self.weights[l-1] -= learning_rate * dW
                self.biases[l-1] -= learning_rate * db
                
                # Propagate gradient to previous layer
                if l > 1:
                    da = np.dot(dz, self.weights[l-1].T)
                    dz = da * self.relu_derivative(self.cache[f'z{l-1}'])
    
    def train(self):
        """Set all layers to training mode."""
        self.training = True
        for norm_layer in self.norm_layers:
            if hasattr(norm_layer, 'train'):
                norm_layer.train()
    
    def eval(self):
        """Set all layers to evaluation mode."""
        self.training = False
        for norm_layer in self.norm_layers:
            if hasattr(norm_layer, 'eval'):
                norm_layer.eval()
    
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on input data.
        
        Returns:
            Tuple of (class_predictions, class_probabilities)
        """
        self.eval()  # Ensure evaluation mode
        probs = self.forward(x)
        preds = np.argmax(probs, axis=1)  # Convert probabilities to class labels
        return preds, probs


def generate_data(n_samples: int = 1000, n_features: int = 20) -> Tuple:
    """
    Generate synthetic classification dataset for normalization experiments.
    
    Creates a multi-class classification problem with informative and redundant
    features to test normalization effectiveness.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,      # Actually predictive features
        n_redundant=5,         # Linear combinations of informative features
        n_classes=3,           # Multi-class problem
        random_state=42
    )
    
    # Stratified split preserves class distribution in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize features (mean=0, std=1) - important for neural networks
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # One-hot encode labels for softmax cross-entropy
    y_train_onehot = np.eye(3)[y_train]
    y_test_onehot = np.eye(3)[y_test]
    
    return X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot


def train_model(X_train: np.ndarray, y_train: np.ndarray, y_train_onehot: np.ndarray,
                norm_type: Optional[str] = None, batch_size: int = 32, 
                tiny_batch: bool = False) -> Dict:
    """
    Train MLP with specified normalization type.
    
    Args:
        norm_type: Type of normalization ('batch', 'layer', or None)
        batch_size: Mini-batch size for gradient descent
        tiny_batch: If True, uses batch_size=1 to demonstrate BatchNorm issues
    
    Returns:
        Dictionary containing trained model and training history
    """
    if tiny_batch:
        batch_size = 1  # Extreme case to show BatchNorm limitations
    
    print(f"Training model with {norm_type or 'no'} normalization, batch_size={batch_size}...")
    
    # Initialize model with architecture: input -> 64 -> 32 -> 3 (output)
    model = NormalizedMLP(
        layer_sizes=[X_train.shape[1], 64, 32, 3],
        norm_type=norm_type,
        norm_position='after'  # Common configuration
    )
    
    # Training hyperparameters
    epochs = 100 if not tiny_batch else 200  # More epochs needed for unstable training
    learning_rate = 0.01
    
    # Track training progress
    train_losses = []
    train_accuracies = []
    
    for epoch in range(epochs):
        # Shuffle data each epoch to prevent ordering bias
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train_onehot[indices]
        
        epoch_loss = 0
        batch_count = 0
        
        # Mini-batch gradient descent
        for i in range(0, len(X_train), batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            if len(X_batch) == 0:
                continue
                
            # Set training mode for normalization layers
            model.train()
            
            # Forward pass
            predictions = model.forward(X_batch)
            
            # Cross-entropy loss (with epsilon for numerical stability)
            loss = -np.mean(np.sum(y_batch * np.log(predictions + 1e-8), axis=1))
            epoch_loss += loss
            
            # Backward pass and parameter update
            model.backward(X_batch, y_batch, learning_rate)
            batch_count += 1
        
        # Calculate training accuracy for this epoch
        model.eval()
        train_preds, _ = model.predict(X_train)
        train_acc = np.mean(train_preds == y_train)
        
        # Store metrics
        avg_loss = epoch_loss / batch_count if batch_count > 0 else epoch_loss
        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)
        
        # Progress reporting
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: Loss = {avg_loss:.4f}, Accuracy = {train_acc:.4f}")
    
    return {
        'model': model,
        'losses': train_losses,
        'accuracies': train_accuracies
    }


def demonstrate_batchnorm_issue():
    """
    Demonstrate BatchNorm's fundamental limitation with small batch sizes.
    
    Shows why BatchNorm fails with batch_size=1 and struggles with small batches:
    1. Cannot compute meaningful variance with batch_size=1 (always zero)
    2. High variance in normalization with tiny batches
    3. Running statistics become unstable
    """
    print("\n" + "="*60)
    print("DEMONSTRATING BATCHNORM ISSUES WITH TINY BATCH SIZES")
    print("="*60)
    
    # Generate sample data
    X = np.random.randn(100, 10)  # 100 samples, 10 features
    
    # Test with progressively smaller batch sizes
    batch_sizes = [32, 2, 1]
    
    for batch_size in batch_sizes:
        print(f"\nTesting BatchNorm with batch_size = {batch_size}:")
        
        batchnorm = BatchNorm1d(10)
        batchnorm.train()
        
        # Simulate training by processing data in batches
        variances = []
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i+batch_size]
            output = batchnorm.forward(X_batch)
            batch_var = np.var(output, axis=0)
            variances.append(np.mean(batch_var))
        
        print(f"  Average output variance: {np.mean(variances):.6f}")
        print(f"  Variance stability (std): {np.std(variances):.6f}")
        
        if batch_size == 1:
            print("  ⚠️  CRITICAL ISSUE: With batch_size=1:")
            print("     - Variance estimate is mathematically zero")
            print("     - Division by (near) zero in normalization")
            print("     - No actual normalization occurs")
            print("     - Running statistics fail to converge")
            print("  ✅ RECOMMENDATION: Use LayerNorm instead for batch_size=1")
        elif batch_size == 2:
            print("  ⚠️  WARNING: With batch_size=2:")
            print("     - High variance in batch statistics")
            print("     - Unstable normalization across batches")
            print("     - Consider LayerNorm or increase batch size")


def compare_normalization_methods() -> Tuple[Dict, Dict]:
    """
    Comprehensive comparison of normalization techniques.
    
    Returns:
        Tuple of (normal_results, tiny_batch_results)
    """
    print("\n" + "="*60)
    print("COMPARING NORMALIZATION METHODS")
    print("="*60)
    
    # Generate dataset
    X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot = generate_data()
    
    # Train models with different normalization (normal batch size)
    results = {}
    norm_types = [None, 'batch', 'layer']
    
    print("\nTraining with normal batch size (32):")
    for norm_type in norm_types:
        results[norm_type] = train_model(
            X_train, y_train, y_train_onehot, 
            norm_type=norm_type, batch_size=32
        )
    
    # Train models with tiny batch size (demonstrating BatchNorm limitations)
    print("\n" + "-"*40)
    print("Training with tiny batch size (1):")
    print("-"*40)
    
    tiny_results = {}
    for norm_type in ['batch', 'layer']:  # No normalization case omitted for clarity
        tiny_results[norm_type] = train_model(
            X_train, y_train, y_train_onehot,
            norm_type=norm_type, batch_size=1, tiny_batch=True
        )
    
    return results, tiny_results


def plot_results(results: Dict, tiny_results: Dict):
    """
    Visualize training dynamics across normalization methods.
    
    Creates a 2x2 grid showing:
    1. Loss comparison with normal batch size
    2. Accuracy comparison with normal batch size
    3. Loss with tiny batch size
    4. Accuracy with tiny batch size
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Color scheme for consistency
    colors = {
        None: 'blue',
        'batch': 'green',
        'layer': 'orange'
    }
    
    # 1. Training Loss Comparison (Batch Size = 32)
    ax = axes[0, 0]
    for norm_type in results:
        ax.plot(results[norm_type]['losses'], 
                label=f'{norm_type or "No Norm"}', 
                linewidth=2, 
                color=colors[norm_type])
    ax.set_title('Training Loss Comparison (Batch Size = 32)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Cross-Entropy Loss', fontsize=10)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(results[None]['losses'][:5]) * 1.1)  # Consistent y-axis
    
    # 2. Training Accuracy Comparison (Batch Size = 32)
    ax = axes[0, 1]
    for norm_type in results:
        ax.plot(results[norm_type]['accuracies'], 
                label=f'{norm_type or "No Norm"}', 
                linewidth=2, 
                color=colors[norm_type])
    ax.set_title('Training Accuracy Comparison (Batch Size = 32)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Accuracy', fontsize=10)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # 3. Training Loss with Tiny Batch Size
    ax = axes[1, 0]
    ax.plot(tiny_results['batch']['losses'], 
            label='BatchNorm (BS=1)', 
            linewidth=2, 
            color='red', 
            linestyle='--')
    ax.plot(tiny_results['layer']['losses'], 
            label='LayerNorm (BS=1)', 
            linewidth=2, 
            color='green')
    ax.set_title('Training Loss with Tiny Batch Size', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Cross-Entropy Loss', fontsize=10)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 4. Training Accuracy with Tiny Batch Size
    ax = axes[1, 1]
    ax.plot(tiny_results['batch']['accuracies'], 
            label='BatchNorm (BS=1)', 
            linewidth=2, 
            color='red', 
            linestyle='--')
    ax.plot(tiny_results['layer']['accuracies'], 
            label='LayerNorm (BS=1)', 
            linewidth=2, 
            color='green')
    ax.set_title('Training Accuracy with Tiny Batch Size', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Accuracy', fontsize=10)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig('normalization_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Performance summary table
    print("\n" + "="*60)
    print("FINAL PERFORMANCE SUMMARY")
    print("="*60)
    
    final_metrics = {}
    for norm_type in results:
        final_metrics[norm_type] = {
            'loss': results[norm_type]['losses'][-1],
            'accuracy': results[norm_type]['accuracies'][-1]
        }
    
    for norm_type in tiny_results:
        final_metrics[f"{norm_type}_tiny"] = {
            'loss': tiny_results[norm_type]['losses'][-1],
            'accuracy': tiny_results[norm_type]['accuracies'][-1]
        }
    
    # Display formatted results
    print(f"{'Method':<25} {'Final Loss':<15} {'Final Accuracy':<15}")
    print("-" * 55)
    
    for name, metrics in final_metrics.items():
        display_name = name.replace('_tiny', ' (BS=1)').replace('None', 'No Norm')
        print(f"{display_name:<25} {metrics['loss']:<15.4f} {metrics['accuracy']:<15.4f}")


def main():
    """
    Main execution function for normalization experiments.
    
    Orchestrates the complete demonstration:
    1. Theoretical explanation of BatchNorm limitations
    2. Empirical comparison of normalization methods
    3. Visualization of results
    4. Practical recommendations
    """
    print("Exercise 13: Batch Normalization & Layer Normalization")
    print("="*60)
    print("\nUnderstanding Internal Covariate Shift and Normalization Techniques\n")
    
    # Part 1: Demonstrate theoretical limitations
    demonstrate_batchnorm_issue()
    
    # Part 2: Empirical comparison
    results, tiny_results = compare_normalization_methods()
    
    # Part 3: Visualization
    plot_results(results, tiny_results)
    
    # Part 4: Practical insights
    print("\n" + "="*60)
    print("KEY INSIGHTS AND PRACTICAL RECOMMENDATIONS")
    print("="*60)
    print("\n1. BATCHNORM LIMITATIONS WITH SMALL BATCHES:")
    print("   • Batch size = 1: Variance is mathematically zero")
    print("   • Batch size = 2-8: High variance in normalization")
    print("   • Solution: Use LayerNorm for small batch scenarios")
    
    print("\n2. TRAINING STABILITY AND CONVERGENCE:")
    print("   • Both BatchNorm and LayerNorm improve stability")
    print("   • LayerNorm is batch-size invariant (more robust)")
    print("   • BatchNorm often slightly better with large batches")
    
    print("\n3. PRACTICAL USE CASES:")
    print("   • BatchNorm: CNNs, large batch training, stable domains")
    print("   • LayerNorm: RNNs, Transformers, reinforcement learning")
    print("   • No Norm: Simple networks, well-scaled data")
    
    print("\n4. IMPLEMENTATION NOTES:")
    print("   • Always track training/eval modes for normalization")
    print("   • Use running statistics during inference")
    print("   • Consider batch size when choosing normalization")
    
    print("\n" + "="*60)
    print("Open Source ML Education - kira-ml")
    print("Repository: https://github.com/kira-ml/open-source-ml-education")
    print("="*60)


if __name__ == "__main__":
    main()