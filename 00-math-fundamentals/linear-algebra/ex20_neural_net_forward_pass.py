"""
Educational Neural Network Forward Pass Implementation

A clear, well-documented implementation of multi-layer perceptron forward pass
designed for learning and understanding neural network fundamentals.

Author: kira-ml
Version: 2.0.0
Date: 20/12/2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Constants for numerical stability
NUMERICAL_EPSILON = 1e-15
SIGMOID_CLIP_VALUE = 500.0
DEFAULT_RANDOM_SEED = 42

@dataclass
class NetworkConfig:
    """
    Configuration for neural network initialization.
    
    This class defines the blueprint for our neural network architecture.
    Key concepts:
    - layer_dims: Tuple defining neurons in each layer (input, hidden, output)
    - random_seed: Ensures reproducible weight initialization
    - activation_function: Nonlinear function applied to each layer's output
    """
    layer_dims: Tuple[int, ...]
    random_seed: int = DEFAULT_RANDOM_SEED
    activation_function: str = "sigmoid"

    def __post_init__(self):
        """Validate configuration parameters."""
        if len(self.layer_dims) < 2:
            raise ValueError("Network must have at least input and output layers")
        if any(dim <= 0 for dim in self.layer_dims):
            raise ValueError("All layer dimensions must be positive")
        if self.activation_function not in ["sigmoid"]:
            raise ValueError(f"Unsupported activation: {self.activation_function}")

class ActivationFunction(ABC):
    """
    Abstract base class for activation functions.
    
    Activation functions introduce non-linearity to neural networks, enabling
    them to learn complex patterns. Without activation functions, neural networks
    would only be able to learn linear relationships.
    """
    
    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        """
        Compute activation forward pass.
        
        Args:
            z: Pre-activation values (linear transformation of inputs)
            
        Returns:
            Activated values with same shape as input
        """
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return activation function name."""
        pass

class SigmoidActivation(ActivationFunction):
    """
    Sigmoid activation function implementation.
    
    Mathematical definition: σ(z) = 1 / (1 + exp(-z))
    
    Properties:
    - Output range: (0, 1)
    - Useful for: Binary classification, probabilities
    - Limitations: Vanishing gradient for extreme inputs
    """
    
    def forward(self, z: np.ndarray) -> np.ndarray:
        """
        Compute sigmoid with numerical stability.
        
        The clipping prevents overflow in the exponential function
        for very large positive or negative inputs.
        """
        # Input validation
        if not isinstance(z, np.ndarray):
            raise TypeError("Input must be numpy array")
        if z.ndim != 2:
            raise ValueError("Input must be 2D array with shape (neurons, samples)")
        
        # Clip extreme values to prevent numerical overflow
        z_clipped = np.clip(z, -SIGMOID_CLIP_VALUE, SIGMOID_CLIP_VALUE)
        
        # Compute sigmoid: 1 / (1 + exp(-z))
        return 1 / (1 + np.exp(-z_clipped))
    
    def name(self) -> str:
        return "sigmoid"

class Layer:
    """
    Represents a single layer in a neural network.
    
    Each layer performs two operations:
    1. Linear transformation: z = W·x + b
    2. Non-linear activation: a = σ(z)
    
    Where:
    - W: Weight matrix (output_dim × input_dim)
    - x: Input vector (input_dim × batch_size)
    - b: Bias vector (output_dim × 1)
    - σ: Activation function
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 activation: ActivationFunction, random_seed: Optional[int] = None):
        """
        Initialize neural network layer.
        
        Weight initialization is critical for training convergence.
        Xavier initialization helps maintain stable gradient flow.
        """
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("Layer dimensions must be positive integers")
        
        # Initialize random number generator for reproducibility
        rng = np.random.RandomState(random_seed)
        
        # Xavier initialization: scale weights based on input and output dimensions
        # This helps prevent exploding/vanishing gradients during training
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        self.weights = rng.randn(output_dim, input_dim) * scale
        self.biases = np.zeros((output_dim, 1))
        self.activation = activation
        
        print(f"Layer initialized: {input_dim} → {output_dim} neurons")
        print(f"  Weight matrix shape: {self.weights.shape}")
        print(f"  Bias vector shape: {self.biases.shape}")
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Perform forward pass through the layer.
        
        Mathematical operations:
        1. Linear transformation: z = W·x + b
        2. Activation: a = σ(z)
        
        Args:
            inputs: Matrix of shape (input_dim, batch_size)
            
        Returns:
            Activated outputs of shape (output_dim, batch_size)
        """
        # Validate input dimensions match weight matrix
        expected_input_dim = self.weights.shape[1]
        if inputs.shape[0] != expected_input_dim:
            raise ValueError(
                f"Input dimension mismatch. "
                f"Expected: {expected_input_dim}, Got: {inputs.shape[0]}"
            )
        
        # Step 1: Linear transformation
        # z = weights · inputs + biases
        # Broadcasting automatically adds bias to each sample in batch
        z = np.dot(self.weights, inputs) + self.biases
        
        # Step 2: Apply activation function
        activated_output = self.activation.forward(z)
        
        return activated_output

class NeuralNetwork:
    """
    Multi-layer perceptron (MLP) implementation.
    
    A feedforward neural network consisting of multiple layers.
    Data flows from input layer through hidden layers to output layer.
    """
    
    def __init__(self, config: NetworkConfig):
        """
        Construct neural network from configuration.
        
        The network architecture is defined by layer_dims.
        Example: (2, 4, 3, 1) creates:
        - Input layer: 2 neurons
        - Hidden layer 1: 4 neurons
        - Hidden layer 2: 3 neurons
        - Output layer: 1 neuron
        """
        self.config = config
        self.layers: List[Layer] = []
        
        # Set global random seed for reproducibility
        np.random.seed(config.random_seed)
        
        # Create activation function instance
        # Note: Currently only sigmoid is implemented
        # In practice, you might use different activations for different layers
        activation = SigmoidActivation()
        
        # Build network layer by layer
        print(f"\nBuilding neural network with architecture: {config.layer_dims}")
        print("=" * 50)
        
        for layer_idx in range(len(config.layer_dims) - 1):
            input_dim = config.layer_dims[layer_idx]
            output_dim = config.layer_dims[layer_idx + 1]
            
            # Create layer with unique random seed for each layer
            layer_seed = config.random_seed + layer_idx if config.random_seed else None
            layer = Layer(
                input_dim=input_dim,
                output_dim=output_dim,
                activation=activation,
                random_seed=layer_seed
            )
            self.layers.append(layer)
        
        print("=" * 50)
        print(f"Network built with {len(self.layers)} layers\n")
    
    def forward(self, inputs: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Perform forward propagation through all layers.
        
        Forward propagation is the process of passing input data through
        the network to produce an output prediction.
        
        Args:
            inputs: Input data of shape (input_dim, batch_size)
            
        Returns:
            Tuple containing:
            - Final network output
            - List of activations from each layer
        """
        # Input validation
        if not isinstance(inputs, np.ndarray):
            raise TypeError("Inputs must be numpy array")
        if inputs.ndim != 2:
            raise ValueError("Inputs must be 2D array (features × samples)")
        
        expected_input_dim = self.config.layer_dims[0]
        if inputs.shape[0] != expected_input_dim:
            raise ValueError(
                f"Input feature dimension mismatch. "
                f"Expected: {expected_input_dim}, Got: {inputs.shape[0]}"
            )
        
        # Store activations for visualization and potential backpropagation
        activations = []
        current_activation = inputs
        
        print(f"\nForward propagation starting...")
        print(f"Input shape: {inputs.shape}")
        print("-" * 30)
        
        # Pass data through each layer sequentially
        for layer_idx, layer in enumerate(self.layers):
            print(f"Layer {layer_idx + 1}: {layer.weights.shape[1]} → {layer.weights.shape[0]} neurons")
            
            # Forward pass through current layer
            current_activation = layer.forward(current_activation)
            activations.append(current_activation)
            
            print(f"  Output shape: {current_activation.shape}")
            print(f"  Output range: [{current_activation.min():.4f}, {current_activation.max():.4f}]")
        
        print("-" * 30)
        print(f"Final output shape: {activations[-1].shape}")
        
        return activations[-1], activations

def visualize_activation_function():
    """
    Visualize the sigmoid activation function.
    
    This helps understand how the sigmoid transforms inputs:
    - Maps any real number to range (0, 1)
    - Acts as a smooth threshold function
    """
    activation = SigmoidActivation()
    
    # Create input values ranging from -10 to 10
    z_values = np.linspace(-10, 10, 200).reshape(1, -1)
    sigmoid_values = activation.forward(z_values).flatten()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Sigmoid function
    ax = axes[0, 0]
    ax.plot(z_values.flatten(), sigmoid_values, 'b-', linewidth=3)
    ax.set_title('Sigmoid Activation Function: σ(z) = 1 / (1 + exp(-z))', fontsize=12)
    ax.set_xlabel('Input z (pre-activation)', fontsize=10)
    ax.set_ylabel('Output σ(z)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Decision boundary (0.5)')
    ax.legend()
    
    # Plot 2: Derivative of sigmoid (for educational context)
    ax = axes[0, 1]
    sigmoid_derivative = sigmoid_values * (1 - sigmoid_values)  # σ'(z) = σ(z)(1 - σ(z))
    ax.plot(z_values.flatten(), sigmoid_derivative, 'g-', linewidth=3)
    ax.set_title('Derivative of Sigmoid: σ\'(z) = σ(z)(1 - σ(z))', fontsize=12)
    ax.set_xlabel('Input z', fontsize=10)
    ax.set_ylabel('Gradient σ\'(z)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.25, color='r', linestyle='--', alpha=0.7, 
               label='Maximum gradient (0.25 at z=0)')
    ax.legend()
    
    # Plot 3: Network architecture
    ax = axes[1, 0]
    example_config = NetworkConfig(layer_dims=(3, 4, 4, 2))
    layer_x = np.arange(len(example_config.layer_dims))
    
    for i, (x, layer_size) in enumerate(zip(layer_x, example_config.layer_dims)):
        y_positions = np.linspace(0, 1, layer_size)
        ax.scatter([x] * layer_size, y_positions, s=300, 
                   label=f'Layer {i}: {layer_size} neurons', alpha=0.8)
        
        # Draw connections between layers
        if i > 0:
            prev_y = np.linspace(0, 1, example_config.layer_dims[i-1])
            # Draw subset of connections for clarity
            for y1 in prev_y[:min(3, len(prev_y))]:
                for y2 in y_positions[:min(3, len(y_positions))]:
                    ax.plot([x-1, x], [y1, y2], 'k-', alpha=0.2, linewidth=0.5)
    
    ax.set_title('Neural Network Architecture', fontsize=12)
    ax.set_xlabel('Layer Index', fontsize=10)
    ax.set_ylabel('Neuron Position', fontsize=10)
    ax.set_xticks(layer_x)
    ax.set_xticklabels([f'L{i}' for i in range(len(example_config.layer_dims))])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # Plot 4: Information flow through network
    ax = axes[1, 1]
    # Simulate simple forward pass
    test_input = np.array([[1.0, -1.0, 0.5]]).T  # 3 input features, 1 sample
    network = NeuralNetwork(example_config)
    output, activations = network.forward(test_input)
    
    # Plot activation magnitudes through layers
    layer_indices = range(len(activations) + 1)
    activation_magnitudes = [np.mean(np.abs(test_input))]
    activation_magnitudes.extend([np.mean(np.abs(a)) for a in activations])
    
    ax.plot(layer_indices, activation_magnitudes, 'o-', linewidth=2, markersize=8)
    ax.set_title('Average Activation Magnitude Through Layers', fontsize=12)
    ax.set_xlabel('Layer Index', fontsize=10)
    ax.set_ylabel('Mean Absolute Activation', fontsize=10)
    ax.set_xticks(layer_indices)
    ax.set_xticklabels(['Input'] + [f'Hidden {i+1}' for i in range(len(activations)-1)] + ['Output'])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Neural Network Fundamentals by kira-ml', fontsize=14, y=1.02)
    plt.show()

def run_educational_example():
    """
    Demonstrate neural network forward pass with clear explanations.
    """
    print("=" * 70)
    print("EDUCATIONAL NEURAL NETWORK DEMONSTRATION")
    print("Author: kira-ml")
    print("=" * 70)
    
    # Part 1: Understanding the architecture
    print("\nPART 1: NETWORK ARCHITECTURE")
    print("-" * 40)
    
    # Define a simple network: 2 input features, 2 hidden layers, 1 output
    config = NetworkConfig(
        layer_dims=(2, 4, 3, 1),  # 2 → 4 → 3 → 1
        random_seed=DEFAULT_RANDOM_SEED
    )
    
    print(f"Network architecture: {config.layer_dims}")
    print(f"Total layers (excluding input): {len(config.layer_dims) - 1}")
    print(f"Total parameters (weights + biases):")
    
    # Calculate parameter count
    total_params = 0
    for i in range(len(config.layer_dims) - 1):
        weights = config.layer_dims[i] * config.layer_dims[i + 1]
        biases = config.layer_dims[i + 1]
        layer_params = weights + biases
        total_params += layer_params
        print(f"  Layer {i}: {config.layer_dims[i]}→{config.layer_dims[i+1]}: "
              f"{weights} weights + {biases} biases = {layer_params} parameters")
    
    print(f"\nTotal: {total_params} trainable parameters")
    
    # Part 2: Initialize network
    print("\n\nPART 2: NETWORK INITIALIZATION")
    print("-" * 40)
    
    network = NeuralNetwork(config)
    
    # Part 3: Create sample data
    print("\n\nPART 3: SAMPLE DATA")
    print("-" * 40)
    
    # Create a batch of 5 samples, each with 2 features
    np.random.seed(42)
    X = np.random.randn(2, 5)  # Shape: (2 features × 5 samples)
    
    print(f"Input data shape: {X.shape}")
    print(f"Features (rows): {X.shape[0]}")
    print(f"Samples (columns): {X.shape[1]}")
    print("\nSample data (first 3 samples):")
    print(X[:, :3])
    
    # Part 4: Forward propagation
    print("\n\nPART 4: FORWARD PROPAGATION")
    print("-" * 40)
    
    output, all_activations = network.forward(X)
    
    print(f"\nFinal network output shape: {output.shape}")
    print(f"Number of activation matrices stored: {len(all_activations)}")
    
    # Part 5: Analysis
    print("\n\nPART 5: OUTPUT ANALYSIS")
    print("-" * 40)
    
    print(f"\nOutput values range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"Output mean: {output.mean():.4f}")
    print(f"Output standard deviation: {output.std():.4f}")
    
    # Since we used sigmoid activation, all outputs should be between 0 and 1
    print(f"\nAll outputs are between 0 and 1 (sigmoid property): "
          f"{np.all((output >= 0) & (output <= 1))}")
    
    # Part 6: Understanding what happened
    print("\n\nPART 6: WHAT JUST HAPPENED?")
    print("-" * 40)
    
    print("\nThe forward pass performed these operations for each sample:")
    print("1. Input layer: Passed 2 features to first hidden layer")
    print("2. Hidden layer 1: Applied linear transform (W1·x + b1) then sigmoid")
    print("3. Hidden layer 2: Applied linear transform (W2·a1 + b2) then sigmoid")
    print("4. Output layer: Applied linear transform (W3·a2 + b3) then sigmoid")
    print("\nEach transformation added non-linearity, allowing the network")
    print("to potentially learn complex relationships between inputs and outputs.")
    
    return output, all_activations, X

def demonstrate_weight_initialization():
    """
    Show the importance of proper weight initialization.
    """
    print("\n" + "=" * 70)
    print("WEIGHT INITIALIZATION DEMONSTRATION")
    print("=" * 70)
    
    # Create a simple network
    config = NetworkConfig(layer_dims=(10, 5, 1))
    
    # Initialize with different strategies
    np.random.seed(42)
    
    print("\n1. Xavier Initialization (used in our implementation):")
    print("   Scale = sqrt(2 / (fan_in + fan_out))")
    
    # Show example calculation
    fan_in = 10
    fan_out = 5
    scale = np.sqrt(2.0 / (fan_in + fan_out))
    print(f"   For layer 10→5: scale = sqrt(2 / (10 + 5)) = {scale:.4f}")
    
    print("\n2. Why initialization matters:")
    print("   - Too large weights: Exploding gradients")
    print("   - Too small weights: Vanishing gradients")
    print("   - Proper scale: Stable gradient flow for training")

if __name__ == "__main__":
    """
    Main execution: Run the educational demonstration.
    """
    # Run the complete educational example
    final_output, activations, inputs = run_educational_example()
    
    # Show weight initialization concepts
    demonstrate_weight_initialization()
    
    # Visualize activation function and network concepts
    print("\n" + "=" * 70)
    print("VISUALIZING NEURAL NETWORK CONCEPTS")
    print("=" * 70)
    visualize_activation_function()
    
    print("\n" + "=" * 70)
    print("EDUCATIONAL DEMONSTRATION COMPLETE")
    print("Key concepts covered:")
    print("1. Neural network architecture and layer composition")
    print("2. Forward propagation mathematics")
    print("3. Sigmoid activation function properties")
    print("4. Proper weight initialization techniques")
    print("5. Batch processing with multiple samples")
    print("=" * 70)