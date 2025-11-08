"""
Exercise 12 — Regularization II: Dropout & Monte Carlo Dropout
Core Concept: Dropout as stochastic regularizer and Monte Carlo dropout for uncertainty estimation.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)

class Dropout:
    """Dropout layer implementation with train/eval modes"""
    
    def __init__(self, rate=0.5):
        """
        Initialize dropout layer
        
        Args:
            rate: Dropout rate (probability of dropping a unit)
        """
        self.rate = rate
        self.mask = None
        self.training = True
    
    def forward(self, x):
        """
        Forward pass with dropout
        
        Args:
            x: Input tensor
            
        Returns:
            Output with dropout applied during training
        """
        if self.training:
            # Create binary mask and scale outputs
            self.mask = np.random.binomial(1, 1 - self.rate, size=x.shape) / (1 - self.rate)
            return x * self.mask
        else:
            # During evaluation, no dropout is applied
            return x
    
    def backward(self, grad_output):
        """
        Backward pass for dropout
        
        Args:
            grad_output: Gradient from subsequent layer
            
        Returns:
            Gradient with dropout mask applied
        """
        if self.training:
            return grad_output * self.mask
        else:
            return grad_output
    
    def train(self):
        """Set layer to training mode"""
        self.training = True
    
    def eval(self):
        """Set layer to evaluation mode"""
        self.training = False

class MLP:
    """Multi-Layer Perceptron with Dropout"""
    
    def __init__(self, layer_sizes, dropout_rates=None):
        """
        Initialize MLP with dropout
        
        Args:
            layer_sizes: List of layer sizes [input_size, hidden1_size, ..., output_size]
            dropout_rates: List of dropout rates for each hidden layer
        """
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1  # Number of weight layers
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(self.L):
            # He initialization for ReLU
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(W)
            self.biases.append(b)
        
        # Initialize dropout layers
        self.dropout_layers = []
        if dropout_rates:
            for rate in dropout_rates:
                self.dropout_layers.append(Dropout(rate))
        
        # Store intermediate values for backpropagation
        self.cache = {}
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)
    
    def softmax(self, x):
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, x, apply_dropout=True):
        """
        Forward pass through the network
        
        Args:
            x: Input data
            apply_dropout: Whether to apply dropout
            
        Returns:
            Network output
        """
        self.cache['a0'] = x
        a = x
        
        # Forward through hidden layers
        dropout_idx = 0
        for i in range(self.L - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self.relu(z)
            
            # Apply dropout after activation (except in output layer)
            if apply_dropout and dropout_idx < len(self.dropout_layers):
                self.dropout_layers[dropout_idx].training = self.training
                a = self.dropout_layers[dropout_idx].forward(a)
                dropout_idx += 1
            
            self.cache[f'z{i+1}'] = z
            self.cache[f'a{i+1}'] = a
        
        # Output layer (no dropout)
        z_out = np.dot(a, self.weights[-1]) + self.biases[-1]
        a_out = self.softmax(z_out)
        
        self.cache[f'z{self.L}'] = z_out
        self.cache[f'a{self.L}'] = a_out
        
        return a_out
    
    def backward(self, x, y, learning_rate=0.01):
        """
        Backward pass and parameter update
        
        Args:
            x: Input data
            y: True labels
            learning_rate: Learning rate for gradient descent
        """
        m = x.shape[0]
        
        # Forward pass to populate cache
        self.forward(x, apply_dropout=True)
        
        # Calculate output gradient
        dz = self.cache[f'a{self.L}'] - y
        gradients = {}
        
        # Backpropagate through layers
        for l in range(self.L, 0, -1):
            if l == self.L:
                # Output layer
                gradients[f'dW{l}'] = np.dot(self.cache[f'a{l-1}'].T, dz) / m
                gradients[f'db{l}'] = np.sum(dz, axis=0, keepdims=True) / m
            else:
                # Hidden layers
                da = np.dot(dz, self.weights[l].T)
                
                # Apply dropout mask in backward pass
                if l - 1 < len(self.dropout_layers):
                    da = self.dropout_layers[l-1].backward(da)
                
                dz = da * self.relu_derivative(self.cache[f'z{l}'])
                gradients[f'dW{l}'] = np.dot(self.cache[f'a{l-1}'].T, dz) / m
                gradients[f'db{l}'] = np.sum(dz, axis=0, keepdims=True) / m
        
        # Update weights and biases
        for l in range(1, self.L + 1):
            self.weights[l-1] -= learning_rate * gradients[f'dW{l}']
            self.biases[l-1] -= learning_rate * gradients[f'db{l}']
    
    def train(self):
        """Set network to training mode"""
        self.training = True
        for dropout_layer in self.dropout_layers:
            dropout_layer.train()
    
    def eval(self):
        """Set network to evaluation mode"""
        self.training = False
        for dropout_layer in self.dropout_layers:
            dropout_layer.eval()
    
    def predict(self, x, mc_samples=1):
        """
        Make predictions, optionally using MC dropout
        
        Args:
            x: Input data
            mc_samples: Number of MC samples (1 for deterministic)
            
        Returns:
            predictions: Class predictions
            probabilities: Class probabilities
        """
        if mc_samples == 1:
            # Standard prediction (no MC dropout)
            self.eval()
            probs = self.forward(x, apply_dropout=False)
            preds = np.argmax(probs, axis=1)
            return preds, probs
        else:
            # MC Dropout prediction
            mc_probs = []
            for _ in range(mc_samples):
                self.train()  # Keep dropout active
                probs = self.forward(x, apply_dropout=True)
                mc_probs.append(probs)
            
            # Average probabilities across MC samples
            mc_probs = np.array(mc_probs)
            mean_probs = np.mean(mc_probs, axis=0)
            preds = np.argmax(mean_probs, axis=1)
            
            return preds, mean_probs, mc_probs

def generate_data():
    """Generate synthetic classification dataset"""
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
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

def train_model(X_train, y_train, y_train_onehot, use_dropout=True):
    """Train MLP with or without dropout"""
    
    if use_dropout:
        print("Training model WITH dropout...")
        # MLP with dropout
        model = MLP(
            layer_sizes=[20, 64, 32, 3],  # input -> hidden1 -> hidden2 -> output
            dropout_rates=[0.3, 0.3]  # dropout after hidden1 and hidden2
        )
    else:
        print("Training model WITHOUT dropout...")
        # MLP without dropout
        model = MLP(
            layer_sizes=[20, 64, 32, 3],
            dropout_rates=None
        )
    
    # Training parameters
    epochs = 100
    batch_size = 32
    learning_rate = 0.01
    
    train_losses = []
    train_accuracies = []
    
    for epoch in range(epochs):
        # Mini-batch training
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train_onehot[indices]
        
        epoch_loss = 0
        for i in range(0, len(X_train), batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Forward pass to get predictions
            model.train()
            predictions = model.forward(X_batch, apply_dropout=use_dropout)
            
            # Calculate cross-entropy loss
            loss = -np.mean(np.sum(y_batch * np.log(predictions + 1e-8), axis=1))
            epoch_loss += loss
            
            # Backward pass and update
            model.backward(X_batch, y_batch, learning_rate)
        
        # Calculate training accuracy
        model.eval()
        train_preds, _ = model.predict(X_train)
        train_acc = np.mean(train_preds == y_train)
        
        train_losses.append(epoch_loss / (len(X_train) // batch_size))
        train_accuracies.append(train_acc)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {train_losses[-1]:.4f}, Accuracy = {train_acc:.4f}")
    
    return model, train_losses, train_accuracies

def evaluate_mc_dropout(model, X_test, y_test, mc_samples=30):
    """Evaluate model using Monte Carlo dropout"""
    
    print(f"\nEvaluating with MC Dropout ({mc_samples} samples)...")
    
    # Get MC dropout predictions
    mc_preds, mean_probs, mc_probs_all = model.predict(X_test, mc_samples=mc_samples)
    
    # Calculate predictive variance
    predictive_variance = np.var(mc_probs_all, axis=0)
    mean_variance = np.mean(predictive_variance)
    
    # Calculate accuracy
    accuracy = np.mean(mc_preds == y_test)
    
    print(f"MC Dropout Accuracy: {accuracy:.4f}")
    print(f"Mean Predictive Variance: {mean_variance:.6f}")
    
    return mc_preds, mean_probs, mc_probs_all, predictive_variance, accuracy

def plot_results(models_history, mc_results):
    """Plot training history and MC dropout results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training loss
    axes[0, 0].plot(models_history['with_dropout']['loss'], label='With Dropout', color='blue')
    axes[0, 0].plot(models_history['without_dropout']['loss'], label='Without Dropout', color='red')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot training accuracy
    axes[0, 1].plot(models_history['with_dropout']['accuracy'], label='With Dropout', color='blue')
    axes[0, 1].plot(models_history['without_dropout']['accuracy'], label='Without Dropout', color='red')
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot predictive variances
    correct_mask = mc_results['mc_preds'] == mc_results['y_test']
    incorrect_mask = ~correct_mask
    
    axes[1, 0].hist(mc_results['predictive_variance'][correct_mask].mean(axis=1), 
                   alpha=0.7, label='Correct predictions', bins=20)
    axes[1, 0].hist(mc_results['predictive_variance'][incorrect_mask].mean(axis=1), 
                   alpha=0.7, label='Incorrect predictions', bins=20)
    axes[1, 0].set_title('Predictive Variance Distribution')
    axes[1, 0].set_xlabel('Mean Predictive Variance')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot uncertainty vs confidence
    max_probs = np.max(mc_results['mean_probs'], axis=1)
    mean_variances = mc_results['predictive_variance'].mean(axis=1)
    
    scatter = axes[1, 1].scatter(max_probs, mean_variances, c=correct_mask, 
                                cmap='coolwarm', alpha=0.6)
    axes[1, 1].set_title('Uncertainty vs Confidence')
    axes[1, 1].set_xlabel('Maximum Class Probability')
    axes[1, 1].set_ylabel('Mean Predictive Variance')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label('Correct Prediction')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Incorrect', 'Correct'])
    
    plt.tight_layout()
    plt.savefig('dropout_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run the dropout and MC dropout experiment"""
    
    print("Exercise 12: Dropout & Monte Carlo Dropout")
    print("=" * 50)
    
    # Generate data
    X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot = generate_data()
    print(f"Dataset shape: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"Number of features: {X_train.shape[1]}, Number of classes: {len(np.unique(y_train))}")
    
    # Train model without dropout
    model_no_dropout, loss_no_dropout, acc_no_dropout = train_model(
        X_train, y_train, y_train_onehot, use_dropout=False
    )
    
    # Train model with dropout
    model_dropout, loss_dropout, acc_dropout = train_model(
        X_train, y_train, y_train_onehot, use_dropout=True
    )
    
    # Store training history
    models_history = {
        'with_dropout': {'loss': loss_dropout, 'accuracy': acc_dropout},
        'without_dropout': {'loss': loss_no_dropout, 'accuracy': acc_no_dropout}
    }
    
    # Evaluate standard predictions
    print("\n" + "="*50)
    print("Standard Prediction Results:")
    print("="*50)
    
    # Without dropout
    model_no_dropout.eval()
    preds_no_dropout, probs_no_dropout = model_no_dropout.predict(X_test)
    acc_no_dropout_test = np.mean(preds_no_dropout == y_test)
    print(f"Model WITHOUT Dropout - Test Accuracy: {acc_no_dropout_test:.4f}")
    
    # With dropout (standard evaluation)
    model_dropout.eval()
    preds_dropout, probs_dropout = model_dropout.predict(X_test)
    acc_dropout_test = np.mean(preds_dropout == y_test)
    print(f"Model WITH Dropout - Test Accuracy: {acc_dropout_test:.4f}")
    
    # MC Dropout evaluation
    print("\n" + "="*50)
    print("Monte Carlo Dropout Evaluation:")
    print("="*50)
    
    mc_samples = 30
    mc_preds, mean_probs, mc_probs_all, predictive_variance, mc_accuracy = evaluate_mc_dropout(
        model_dropout, X_test, y_test, mc_samples=mc_samples
    )
    
    # Store MC results
    mc_results = {
        'mc_preds': mc_preds,
        'mean_probs': mean_probs,
        'mc_probs_all': mc_probs_all,
        'predictive_variance': predictive_variance,
        'y_test': y_test
    }
    
    # Print detailed statistics
    print("\n" + "="*50)
    print("Detailed Statistics:")
    print("="*50)
    
    # Calculate per-class statistics
    for class_idx in range(3):
        class_mask = y_test == class_idx
        if np.sum(class_mask) > 0:
            class_variance = predictive_variance[class_mask].mean(axis=1).mean()
            class_accuracy = np.mean(mc_preds[class_mask] == y_test[class_mask])
            print(f"Class {class_idx}: Accuracy = {class_accuracy:.4f}, Mean Variance = {class_variance:.6f}")
    
    # Compare uncertainties for correct vs incorrect predictions
    correct_predictions = mc_preds == y_test
    mean_variance_correct = predictive_variance[correct_predictions].mean(axis=1).mean()
    mean_variance_incorrect = predictive_variance[~correct_predictions].mean(axis=1).mean()
    
    print(f"\nUncertainty Analysis:")
    print(f"Mean variance for CORRECT predictions: {mean_variance_correct:.6f}")
    print(f"Mean variance for INCORRECT predictions: {mean_variance_incorrect:.6f}")
    print(f"Uncertainty ratio (incorrect/correct): {mean_variance_incorrect/mean_variance_correct:.2f}")
    
    # Plot results
    plot_results(models_history, mc_results)
    
    print("\n" + "="*50)
    print("Summary:")
    print("="*50)
    print(f"Standard MLP (no dropout): {acc_no_dropout_test:.4f}")
    print(f"Standard MLP (with dropout): {acc_dropout_test:.4f}")
    print(f"MC Dropout ({mc_samples} samples): {mc_accuracy:.4f}")
    print(f"Overall mean predictive variance: {np.mean(predictive_variance):.6f}")

if __name__ == "__main__":
    main()