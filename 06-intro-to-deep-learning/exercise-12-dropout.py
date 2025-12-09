"""
Exercise 12 — Regularization II: Dropout & Monte Carlo Dropout
Core Concept: Implementing dropout as a stochastic regularizer and Monte Carlo dropout for uncertainty estimation.

Author: kira-ml (GitHub)
Educational Purpose: Learn how dropout prevents overfitting and how to estimate model uncertainty
                     through multiple stochastic forward passes (Monte Carlo Dropout).

Key Learning Objectives:
1. Understand dropout as regularization technique during training
2. Implement Monte Carlo Dropout for uncertainty estimation during inference
3. Visualize how uncertainty correlates with prediction confidence
4. Compare models with/without dropout on synthetic classification data
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Set random seeds for reproducibility - CRITICAL for ML experiments
np.random.seed(42)

class Dropout:
    """
    A neural network dropout layer that randomly "drops" (sets to zero) neurons during training.
    
    Why Dropout Matters:
    - Prevents co-adaptation: Neurons learn to be useful on their own, not just in combination
    - Acts as ensemble training: Each forward pass uses a different "thinned" network
    - Reduces overfitting: Forces network to be robust to missing information
    
    Key Concepts:
    - Training mode: Apply dropout with probability 'rate'
    - Evaluation mode: No dropout, but scale weights by (1-rate) for consistency
    - Scale factor: Divide by (1-rate) to maintain expected activation magnitude
    """
    
    def __init__(self, rate=0.5):
        """
        Initialize dropout layer.
        
        Args:
            rate: Probability of setting a neuron to zero (dropping it).
                  Higher rate = more regularization but slower learning.
                  Typical values: 0.2-0.5 for hidden layers, 0.0 for input/output layers.
        """
        self.rate = rate
        self.mask = None  # Store dropout mask for backward pass
        self.training = True  # Mode flag: True for training, False for evaluation
    
    def forward(self, x):
        """
        Forward pass with dropout.
        
        Mathematical Note:
        During training: output = x * mask / (1 - rate)
        During evaluation: output = x (no dropout applied)
        
        The division by (1-rate) maintains the expected value of activations
        across training and evaluation modes.
        """
        if self.training:
            # Create binary mask (1 = keep neuron, 0 = drop neuron)
            self.mask = np.random.binomial(1, 1 - self.rate, size=x.shape)
            # Scale output to maintain expected activation magnitude
            return x * self.mask / (1 - self.rate)
        else:
            # During evaluation: no dropout, just pass through
            return x
    
    def backward(self, grad_output):
        """
        Backward pass for dropout - just propagate gradient through kept neurons.
        
        In dropout, backward pass is simply the gradient multiplied by the mask.
        This means gradients only flow through neurons that were active in forward pass.
        """
        if self.training:
            return grad_output * self.mask / (1 - self.rate)
        else:
            return grad_output
    
    def train(self):
        """Set layer to training mode (dropout active)."""
        self.training = True
    
    def eval(self):
        """Set layer to evaluation mode (dropout inactive)."""
        self.training = False

class MLP:
    """
    Multi-Layer Perceptron (Neural Network) with optional dropout layers.
    
    This implementation demonstrates:
    1. Basic neural network architecture
    2. Forward/backward propagation
    3. Dropout integration
    4. Monte Carlo Dropout for uncertainty estimation
    
    Architecture: Input -> Hidden Layers (with ReLU) -> Output (with Softmax)
    """
    
    def __init__(self, layer_sizes, dropout_rates=None):
        """
        Initialize neural network with specified architecture.
        
        Args:
            layer_sizes: List defining network architecture.
                        Example: [20, 64, 32, 3] means:
                        20 input features -> 64 neurons in 1st hidden layer ->
                        32 neurons in 2nd hidden layer -> 3 output classes
            dropout_rates: List of dropout probabilities for each hidden layer.
                          Must be length = len(layer_sizes) - 2 (not for input/output)
        """
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1  # Number of weight layers
        
        # Initialize weights with He initialization (good for ReLU)
        self.weights = []
        self.biases = []
        
        for i in range(self.L):
            # He initialization: weights ~ N(0, sqrt(2/fan_in))
            # This helps ReLU networks converge faster
            std_dev = np.sqrt(2.0 / layer_sizes[i])
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * std_dev
            b = np.zeros((1, layer_sizes[i+1]))  # Zero bias initialization
            self.weights.append(W)
            self.biases.append(b)
        
        # Initialize dropout layers for each hidden layer
        self.dropout_layers = []
        if dropout_rates:
            if len(dropout_rates) != len(layer_sizes) - 2:
                raise ValueError(f"Need {len(layer_sizes)-2} dropout rates, got {len(dropout_rates)}")
            for rate in dropout_rates:
                self.dropout_layers.append(Dropout(rate))
        
        # Cache for storing intermediate values during forward pass (needed for backprop)
        self.cache = {}
        self.training = True  # Network mode flag
    
    def relu(self, x):
        """
        Rectified Linear Unit (ReLU) activation function.
        
        ReLU(x) = max(0, x)
        Advantages: Computationally cheap, helps with vanishing gradient problem
        """
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """
        Derivative of ReLU function.
        
        dReLU/dx = 1 if x > 0, else 0
        This is needed for backpropagation through ReLU layers.
        """
        return (x > 0).astype(float)
    
    def softmax(self, x):
        """
        Softmax activation for multi-class classification.
        
        Converts raw scores (logits) to probabilities that sum to 1.
        The subtraction of max(x) improves numerical stability.
        """
        # Numerical stability: subtract max to prevent large exponentials
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, x, apply_dropout=True):
        """
        Forward pass through the network.
        
        Flow: Input -> (Linear -> ReLU -> Dropout) * hidden_layers -> Linear -> Softmax
        
        Args:
            x: Input data of shape (batch_size, input_features)
            apply_dropout: Whether to apply dropout (True for training, False for MC dropout eval)
            
        Returns:
            Probability distribution over classes for each input sample
        """
        self.cache['a0'] = x  # Store input for backprop
        a = x  # Current activation
        
        # Forward through hidden layers
        dropout_idx = 0
        for i in range(self.L - 1):  # All layers except output
            # Linear transformation: z = W * a + b
            z = np.dot(a, self.weights[i]) + self.biases[i]
            
            # Non-linear activation: a = ReLU(z)
            a = self.relu(z)
            
            # Apply dropout after activation (standard practice)
            if apply_dropout and dropout_idx < len(self.dropout_layers):
                self.dropout_layers[dropout_idx].training = self.training
                a = self.dropout_layers[dropout_idx].forward(a)
                dropout_idx += 1
            
            # Store for backward pass
            self.cache[f'z{i+1}'] = z
            self.cache[f'a{i+1}'] = a
        
        # Output layer (no activation here, softmax applied separately)
        z_out = np.dot(a, self.weights[-1]) + self.biases[-1]
        a_out = self.softmax(z_out)  # Apply softmax to get probabilities
        
        # Store output layer values
        self.cache[f'z{self.L}'] = z_out
        self.cache[f'a{self.L}'] = a_out
        
        return a_out
    
    def backward(self, x, y, learning_rate=0.01):
        """
        Backward pass (backpropagation) with gradient descent update.
        
        This implements:
        1. Compute gradients using chain rule
        2. Update weights with learning_rate * gradient
        
        Args:
            x: Input data
            y: True labels in one-hot encoded format
            learning_rate: Step size for gradient descent
        """
        m = x.shape[0]  # Batch size
        
        # Forward pass to populate cache (if not already done)
        if f'a{self.L}' not in self.cache:
            self.forward(x, apply_dropout=True)
        
        # Initialize gradients dictionary
        gradients = {}
        
        # Start backprop from output layer
        # For softmax + cross-entropy, gradient at output is simple: (prediction - true)
        dz_output = self.cache[f'a{self.L}'] - y
        
        # Backpropagate through layers (from output to input)
        for l in range(self.L, 0, -1):
            if l == self.L:  # Output layer
                # dW = (1/m) * a_prev.T * dz
                gradients[f'dW{l}'] = np.dot(self.cache[f'a{l-1}'].T, dz_output) / m
                # db = (1/m) * sum(dz) across batch
                gradients[f'db{l}'] = np.sum(dz_output, axis=0, keepdims=True) / m
                
                # Prepare gradient for next layer (going backward)
                dz = dz_output
            else:  # Hidden layers
                # Gradient coming from next layer
                da = np.dot(dz, self.weights[l].T)
                
                # Apply dropout mask in backward pass (gradient only flows through active neurons)
                if l - 1 < len(self.dropout_layers):
                    da = self.dropout_layers[l-1].backward(da)
                
                # Gradient through ReLU: dz = da * ReLU'(z)
                dz = da * self.relu_derivative(self.cache[f'z{l}'])
                
                # Compute weight and bias gradients
                gradients[f'dW{l}'] = np.dot(self.cache[f'a{l-1}'].T, dz) / m
                gradients[f'db{l}'] = np.sum(dz, axis=0, keepdims=True) / m
        
        # Update weights and biases using gradient descent
        for l in range(1, self.L + 1):
            self.weights[l-1] -= learning_rate * gradients[f'dW{l}']
            self.biases[l-1] -= learning_rate * gradients[f'db{l}']
        
        # Clear cache for next forward pass
        self.cache.clear()
    
    def train(self):
        """Set network to training mode (dropout active)."""
        self.training = True
        for dropout_layer in self.dropout_layers:
            dropout_layer.train()
    
    def eval(self):
        """Set network to evaluation mode (dropout inactive)."""
        self.training = False
        for dropout_layer in self.dropout_layers:
            dropout_layer.eval()
    
    def predict(self, x, mc_samples=1):
        """
        Make predictions, optionally using Monte Carlo Dropout.
        
        Monte Carlo Dropout (MC Dropout):
        - Keep dropout active during inference
        - Run multiple forward passes with different dropout masks
        - Average predictions across passes
        - Use variance of predictions as uncertainty measure
        
        Args:
            x: Input data
            mc_samples: Number of stochastic forward passes
                      1 = standard deterministic prediction
                      >1 = MC Dropout prediction
                      
        Returns:
            Tuple containing:
            - Class predictions (argmax of probabilities)
            - Mean class probabilities across MC samples
            - All MC samples probabilities (only when mc_samples > 1)
        """
        if mc_samples == 1:
            # Standard deterministic prediction
            self.eval()  # Turn off dropout
            probs = self.forward(x, apply_dropout=False)
            preds = np.argmax(probs, axis=1)
            return preds, probs, None
        else:
            # Monte Carlo Dropout prediction
            mc_probs = []
            for _ in range(mc_samples):
                # Important: keep dropout active during inference
                self.train()
                probs = self.forward(x, apply_dropout=True)
                mc_probs.append(probs)
            
            # Convert to numpy array for easier computation
            mc_probs = np.array(mc_probs)  # Shape: (mc_samples, batch_size, n_classes)
            
            # Average probabilities across MC samples
            mean_probs = np.mean(mc_probs, axis=0)
            preds = np.argmax(mean_probs, axis=1)
            
            return preds, mean_probs, mc_probs

def generate_data():
    """
    Generate synthetic classification dataset for experimentation.
    
    Creates a non-trivial 3-class classification problem with:
    - 20 features (15 informative, 5 redundant)
    - 1000 total samples
    - Standardized features (zero mean, unit variance)
    
    Returns:
        Train/test splits with one-hot encoded labels for training
    """
    # Create synthetic dataset with 3 classes
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=15,  # Actually informative features
        n_redundant=5,     # Linear combinations of informative features
        n_classes=3, 
        random_state=42
    )
    
    # Split into training (80%) and testing (20%) sets
    # stratify=y ensures balanced class distribution in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize features (critical for neural network training)
    # Learn scaling parameters from training data only, apply to both sets
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # fit on train, transform train
    X_test = scaler.transform(X_test)        # transform test with same parameters
    
    # Convert labels to one-hot encoding for softmax/cross-entropy training
    n_classes = len(np.unique(y_train))
    y_train_onehot = np.eye(n_classes)[y_train]
    y_test_onehot = np.eye(n_classes)[y_test]
    
    return X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot

def train_model(X_train, y_train, y_train_onehot, use_dropout=True):
    """
    Train MLP model with or without dropout.
    
    This function demonstrates:
    1. Mini-batch gradient descent training loop
    2. Loss calculation (cross-entropy)
    3. Accuracy tracking
    4. Effect of dropout on training dynamics
    
    Args:
        use_dropout: Boolean flag to enable/disable dropout
        
    Returns:
        Trained model and training history (loss/accuracy per epoch)
    """
    if use_dropout:
        print("Training model WITH dropout (regularization active)...")
        model = MLP(
            layer_sizes=[20, 64, 32, 3],  # 20 input -> 64 hidden -> 32 hidden -> 3 output
            dropout_rates=[0.3, 0.3]  # Dropout after each hidden layer
        )
    else:
        print("Training model WITHOUT dropout (baseline for comparison)...")
        model = MLP(
            layer_sizes=[20, 64, 32, 3],
            dropout_rates=None  # No dropout layers
        )
    
    # Training hyperparameters
    epochs = 100
    batch_size = 32
    learning_rate = 0.01
    
    # Track training progress
    train_losses = []
    train_accuracies = []
    
    print(f"\nTraining for {epochs} epochs with batch size {batch_size}")
    print("-" * 40)
    
    for epoch in range(epochs):
        # Shuffle training data each epoch (important for convergence)
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train_onehot[indices]
        
        epoch_loss = 0
        n_batches = 0
        
        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Forward pass to get predictions
            model.train()  # Set to training mode
            predictions = model.forward(X_batch, apply_dropout=use_dropout)
            
            # Calculate cross-entropy loss (standard for classification)
            # Add small epsilon to prevent log(0)
            loss = -np.mean(np.sum(y_batch * np.log(predictions + 1e-8), axis=1))
            epoch_loss += loss
            n_batches += 1
            
            # Backward pass and parameter update
            model.backward(X_batch, y_batch, learning_rate)
        
        # Calculate average loss for this epoch
        avg_loss = epoch_loss / n_batches
        
        # Calculate training accuracy (with dropout turned off)
        model.eval()
        train_preds, _, _ = model.predict(X_train)
        train_acc = np.mean(train_preds == y_train)
        
        # Store metrics
        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)
        
        # Print progress every 20 epochs
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: Loss = {avg_loss:.4f}, Accuracy = {train_acc:.4f}")
    
    print(f"Final: Loss = {train_losses[-1]:.4f}, Accuracy = {train_accuracies[-1]:.4f}")
    return model, train_losses, train_accuracies

def evaluate_mc_dropout(model, X_test, y_test, mc_samples=30):
    """
    Evaluate model using Monte Carlo Dropout for uncertainty estimation.
    
    Monte Carlo Dropout provides two key benefits:
    1. Improved accuracy (ensemble effect)
    2. Uncertainty estimates (variance across forward passes)
    
    High variance = model is uncertain about its prediction
    Low variance = model is confident about its prediction
    
    Args:
        mc_samples: Number of stochastic forward passes (more = better estimate but slower)
        
    Returns:
        Various metrics including predictive variance and accuracy
    """
    print(f"\nPerforming Monte Carlo Dropout with {mc_samples} samples...")
    print("This runs multiple forward passes with different dropout masks.")
    print("Variance across passes indicates model uncertainty.")
    
    # Get MC dropout predictions
    mc_preds, mean_probs, mc_probs_all = model.predict(X_test, mc_samples=mc_samples)
    
    # Calculate predictive variance (uncertainty measure)
    # Shape: (batch_size, n_classes) - variance for each class probability
    predictive_variance = np.var(mc_probs_all, axis=0)
    
    # Average variance across all samples and classes
    mean_variance = np.mean(predictive_variance)
    
    # Calculate accuracy
    accuracy = np.mean(mc_preds == y_test)
    
    print(f"\nMC Dropout Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Mean Predictive Variance: {mean_variance:.6f}")
    print(f"  Higher variance = more uncertainty in predictions")
    
    return mc_preds, mean_probs, mc_probs_all, predictive_variance, accuracy

def plot_results(models_history, mc_results):
    """
    Create comprehensive visualization of dropout and MC dropout results.
    
    Four plots showing:
    1. Training loss with/without dropout
    2. Training accuracy with/without dropout
    3. Distribution of predictive variances
    4. Uncertainty vs confidence relationship
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Training Loss Comparison
    axes[0, 0].plot(models_history['with_dropout']['loss'], 
                   label='With Dropout', color='blue', linewidth=2)
    axes[0, 0].plot(models_history['without_dropout']['loss'], 
                   label='Without Dropout', color='red', linewidth=2, alpha=0.7)
    axes[0, 0].set_title('Training Loss: Dropout vs No Dropout', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Cross-Entropy Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_facecolor('#f8f9fa')
    
    # Add annotation explaining dropout's effect on loss
    axes[0, 0].annotate('Dropout slows convergence\nbut reduces overfitting',
                       xy=(0.6, 0.7), xycoords='axes fraction',
                       fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Plot 2: Training Accuracy Comparison
    axes[0, 1].plot(models_history['with_dropout']['accuracy'], 
                   label='With Dropout', color='blue', linewidth=2)
    axes[0, 1].plot(models_history['without_dropout']['accuracy'], 
                   label='Without Dropout', color='red', linewidth=2, alpha=0.7)
    axes[0, 1].set_title('Training Accuracy: Dropout vs No Dropout', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_facecolor('#f8f9fa')
    axes[0, 1].set_ylim(0, 1.05)
    
    # Plot 3: Predictive Variance Distribution
    correct_mask = mc_results['mc_preds'] == mc_results['y_test']
    incorrect_mask = ~correct_mask
    
    # Extract mean variance per sample (average across classes)
    mean_var_per_sample = mc_results['predictive_variance'].mean(axis=1)
    
    axes[1, 0].hist(mean_var_per_sample[correct_mask], 
                   alpha=0.7, label='Correct Predictions', bins=20,
                   color='green', edgecolor='black')
    axes[1, 0].hist(mean_var_per_sample[incorrect_mask], 
                   alpha=0.7, label='Incorrect Predictions', bins=20,
                   color='red', edgecolor='black')
    axes[1, 0].set_title('Predictive Variance Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Mean Predictive Variance (Uncertainty)')
    axes[1, 0].set_ylabel('Number of Samples')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_facecolor('#f8f9fa')
    
    # Add insight annotation
    axes[1, 0].annotate('Incorrect predictions tend to\nhave higher uncertainty',
                       xy=(0.6, 0.7), xycoords='axes fraction',
                       fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Plot 4: Uncertainty vs Confidence
    max_probs = np.max(mc_results['mean_probs'], axis=1)  # Model's confidence
    mean_variances = mean_var_per_sample  # Model's uncertainty
    
    # Color by correctness
    scatter = axes[1, 1].scatter(max_probs, mean_variances, c=correct_mask, 
                                cmap='RdYlGn', alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[1, 1].set_title('Uncertainty vs Confidence', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Maximum Class Probability (Confidence)')
    axes[1, 1].set_ylabel('Mean Predictive Variance (Uncertainty)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_facecolor('#f8f9fa')
    
    # Add colorbar for correctness
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label('Prediction Correctness')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Incorrect', 'Correct'])
    
    # Add quadrant annotations
    axes[1, 1].axvline(x=0.7, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].axhline(y=np.median(mean_variances), color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].annotate('High Confidence\nLow Uncertainty', xy=(0.85, 0.1), xycoords='axes fraction',
                       fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    axes[1, 1].annotate('Low Confidence\nHigh Uncertainty', xy=(0.15, 0.8), xycoords='axes fraction',
                       fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.suptitle('Dropout & Monte Carlo Dropout Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('dropout_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def main():
    """
    Main function to orchestrate the complete dropout and MC dropout experiment.
    
    Flow:
    1. Generate synthetic data
    2. Train two models (with/without dropout)
    3. Evaluate standard predictions
    4. Perform MC dropout analysis
    5. Visualize and analyze results
    """
    print("=" * 70)
    print("EXERCISE 12: Dropout & Monte Carlo Dropout")
    print("Educational Implementation for Machine Learning Beginners")
    print("Author: kira-ml (GitHub)")
    print("=" * 70)
    
    print("\n📊 PART 1: Data Preparation")
    print("-" * 40)
    
    # Generate and prepare data
    X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot = generate_data()
    print(f"✓ Generated synthetic classification dataset")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Classes: {len(np.unique(y_train))}")
    
    print("\n🎯 PART 2: Model Training")
    print("-" * 40)
    
    # Train model without dropout (baseline)
    print("\n1. Training baseline model (no dropout)...")
    model_no_dropout, loss_no_dropout, acc_no_dropout = train_model(
        X_train, y_train, y_train_onehot, use_dropout=False
    )
    
    # Train model with dropout (regularized)
    print("\n2. Training regularized model (with dropout)...")
    model_dropout, loss_dropout, acc_dropout = train_model(
        X_train, y_train, y_train_onehot, use_dropout=True
    )
    
    # Store training history for visualization
    models_history = {
        'with_dropout': {'loss': loss_dropout, 'accuracy': acc_dropout},
        'without_dropout': {'loss': loss_no_dropout, 'accuracy': acc_no_dropout}
    }
    
    print("\n📈 PART 3: Standard Evaluation")
    print("-" * 40)
    
    # Evaluate standard predictions (no MC dropout)
    print("\nStandard Test Set Evaluation:")
    print("=" * 40)
    
    # Without dropout
    model_no_dropout.eval()
    preds_no_dropout, probs_no_dropout, _ = model_no_dropout.predict(X_test)
    acc_no_dropout_test = np.mean(preds_no_dropout == y_test)
    print(f"\nModel WITHOUT Dropout:")
    print(f"  Test Accuracy: {acc_no_dropout_test:.4f}")
    
    # With dropout (standard evaluation - dropout turned off)
    model_dropout.eval()
    preds_dropout, probs_dropout, _ = model_dropout.predict(X_test)
    acc_dropout_test = np.mean(preds_dropout == y_test)
    print(f"\nModel WITH Dropout (standard evaluation):")
    print(f"  Test Accuracy: {acc_dropout_test:.4f}")
    
    print("\n🔮 PART 4: Monte Carlo Dropout Analysis")
    print("-" * 40)
    
    # MC Dropout evaluation (uncertainty estimation)
    mc_samples = 30
    mc_preds, mean_probs, mc_probs_all, predictive_variance, mc_accuracy = evaluate_mc_dropout(
        model_dropout, X_test, y_test, mc_samples=mc_samples
    )
    
    # Store MC results for visualization
    mc_results = {
        'mc_preds': mc_preds,
        'mean_probs': mean_probs,
        'mc_probs_all': mc_probs_all,
        'predictive_variance': predictive_variance,
        'y_test': y_test
    }
    
    print("\n📊 PART 5: Detailed Statistical Analysis")
    print("-" * 40)
    
    print("\nPer-Class Performance Analysis:")
    print("=" * 40)
    
    # Calculate per-class statistics
    for class_idx in range(3):
        class_mask = y_test == class_idx
        if np.sum(class_mask) > 0:
            # Average variance for this class
            class_variance = predictive_variance[class_mask].mean(axis=1).mean()
            # Accuracy for this class
            class_accuracy = np.mean(mc_preds[class_mask] == y_test[class_mask])
            print(f"Class {class_idx}:")
            print(f"  Accuracy: {class_accuracy:.4f}")
            print(f"  Mean Uncertainty: {class_variance:.6f}")
            print()
    
    print("\nUncertainty Analysis by Prediction Correctness:")
    print("=" * 40)
    
    # Compare uncertainties for correct vs incorrect predictions
    correct_predictions = mc_preds == y_test
    mean_variance_correct = predictive_variance[correct_predictions].mean(axis=1).mean()
    mean_variance_incorrect = predictive_variance[~correct_predictions].mean(axis=1).mean()
    
    print(f"Correct predictions: {np.sum(correct_predictions)} samples")
    print(f"  Mean uncertainty: {mean_variance_correct:.6f}")
    print(f"\nIncorrect predictions: {np.sum(~correct_predictions)} samples")
    print(f"  Mean uncertainty: {mean_variance_incorrect:.6f}")
    print(f"\nUncertainty ratio (incorrect/correct): {mean_variance_incorrect/mean_variance_correct:.2f}x")
    print("  → Incorrect predictions are typically more uncertain!")
    
    print("\n🎨 PART 6: Visualization")
    print("-" * 40)
    print("Generating comprehensive visual analysis...")
    
    # Create and save visualizations
    plot_results(models_history, mc_results)
    print("✓ Visualizations saved as 'dropout_analysis.png'")
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print("\nModel Performance Comparison:")
    print("-" * 40)
    print(f"Standard MLP (no dropout):     {acc_no_dropout_test:.4f}")
    print(f"Standard MLP (with dropout):   {acc_dropout_test:.4f}")
    print(f"MC Dropout ({mc_samples} samples):    {mc_accuracy:.4f}")
    
    print("\nKey Insights:")
    print("-" * 40)
    print("✓ Dropout reduces overfitting (compare training vs test accuracy)")
    print("✓ Dropout may slightly reduce training accuracy but improves generalization")
    print("✓ MC Dropout provides uncertainty estimates alongside predictions")
    print("✓ High uncertainty often correlates with incorrect predictions")
    print("✓ MC Dropout can improve accuracy through ensemble averaging")
    
    print("\n" + "=" * 70)
    print("Exercise Completed Successfully! 🎉")
    print("=" * 70)
    print("\nNext Steps for Learning:")
    print("1. Try different dropout rates (0.1, 0.5, 0.7)")
    print("2. Experiment with dropout placement (before/after activation)")
    print("3. Implement dropout in convolutional neural networks")
    print("4. Explore other uncertainty estimation methods (Deep Ensembles, Bayesian NN)")

if __name__ == "__main__":
    # Execute the main experiment
    main()