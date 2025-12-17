"""
Neural Network Forward Pass Implementation

A production-ready implementation of a multi-layer perceptron forward pass,
adhering to elite machine learning engineering standards.

Version: 1.0.0
Authors: [Your Name]
Date: 2025-12-17
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Configure logging for telemetry
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for numerical stability and versioning
NUMERICAL_EPSILON = 1e-15
SIGMOID_CLIP_VALUE = 500.0
DEFAULT_RANDOM_SEED = 42
VERSION = "1.0.0"

@dataclass
class NetworkConfig:
    """Configuration contract for neural network initialization."""
    layer_dims: Tuple[int, ...]
    random_seed: int = DEFAULT_RANDOM_SEED
    activation_function: str = "sigmoid"

    def __post_init__(self):
        """Validate configuration invariants."""
        if len(self.layer_dims) < 2:
            raise ValueError("Network must have at least input and output layers")
        if any(dim <= 0 for dim in self.layer_dims):
            raise ValueError("All layer dimensions must be positive")
        if self.activation_function not in ["sigmoid"]:
            raise ValueError(f"Unsupported activation: {self.activation_function}")

@dataclass
class ForwardPassResult:
    """Contract for forward propagation results."""
    output: np.ndarray
    activations: Tuple[np.ndarray, ...]
    input_shape: Tuple[int, int]
    network_config: NetworkConfig

class ActivationFunction(ABC):
    """Abstract base for activation functions with mathematical rigor."""

    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        """Compute activation forward pass."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Return activation function name."""
        pass

class SigmoidActivation(ActivationFunction):
    """Numerically stable sigmoid implementation."""

    def forward(self, z: np.ndarray) -> np.ndarray:
        """
        Compute sigmoid with overflow protection.

        Mathematical foundation: σ(z) = 1 / (1 + exp(-z))
        Numerical stability: Clip z to prevent overflow in exp.
        """
        # Defensive: Validate input
        if not isinstance(z, np.ndarray):
            raise TypeError("Input must be numpy array")
        if z.ndim != 2:
            raise ValueError("Input must be 2D array")

        # Clip for numerical stability
        z_clipped = np.clip(z, -SIGMOID_CLIP_VALUE, SIGMOID_CLIP_VALUE)
        return 1 / (1 + np.exp(-z_clipped))

    def name(self) -> str:
        return "sigmoid"

class Layer:
    """Encapsulates a single neural network layer with explicit contracts."""

    def __init__(self, input_dim: int, output_dim: int, activation: ActivationFunction,
                 random_seed: Optional[int] = None):
        """
        Initialize layer with validated parameters.

        Args:
            input_dim: Number of input features
            output_dim: Number of output neurons
            activation: Activation function instance
            random_seed: Seed for reproducible weight initialization
        """
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("Dimensions must be positive")

        rng = np.random.RandomState(random_seed)
        # Xavier initialization for better gradient flow
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        self.weights = rng.randn(output_dim, input_dim) * scale
        self.biases = np.zeros((output_dim, 1))
        self.activation = activation

        logger.info(f"Initialized layer: {input_dim} -> {output_dim} with {activation.name()}")

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Compute layer forward pass with shape validation.

        Mathematical foundation: a = σ(W·x + b)

        Args:
            inputs: Input tensor of shape (input_dim, batch_size)

        Returns:
            Activated output of shape (output_dim, batch_size)
        """
        # Contract validation
        if inputs.shape[0] != self.weights.shape[1]:
            raise ValueError(f"Input dimension {inputs.shape[0]} != expected {self.weights.shape[1]}")

        # Linear transformation
        z = np.dot(self.weights, inputs) + self.biases

        # Activation
        output = self.activation.forward(z)

        logger.debug(f"Layer forward: input {inputs.shape} -> output {output.shape}")
        return output

class NeuralNetwork:
    """Production-grade multi-layer perceptron with comprehensive telemetry."""

    def __init__(self, config: NetworkConfig):
        """
        Initialize neural network with configuration contract.

        Args:
            config: Network configuration
        """
        self.config = config
        self.layers: List[Layer] = []

        # Set global random seed for deterministic execution
        np.random.seed(config.random_seed)

        # Initialize layers
        activation = SigmoidActivation()  # Factory pattern for extensibility
        for i in range(len(config.layer_dims) - 1):
            layer = Layer(
                config.layer_dims[i],
                config.layer_dims[i + 1],
                activation,
                random_seed=config.random_seed + i  # Granular seeding
            )
            self.layers.append(layer)

        logger.info(f"Initialized {len(self.layers)}-layer network: {config.layer_dims}")

    def forward(self, inputs: np.ndarray) -> ForwardPassResult:
        """
        Execute forward propagation with full telemetry.

        Args:
            inputs: Input tensor of shape (input_dim, batch_size)

        Returns:
            ForwardPassResult containing outputs and intermediate activations
        """
        # Input validation
        if not isinstance(inputs, np.ndarray):
            raise TypeError("Inputs must be numpy array")
        if inputs.ndim != 2:
            raise ValueError("Inputs must be 2D")
        if inputs.shape[0] != self.config.layer_dims[0]:
            raise ValueError(f"Input features {inputs.shape[0]} != network input {self.config.layer_dims[0]}")

        activations = []
        current_input = inputs

        # Progressive forward pass
        for i, layer in enumerate(self.layers):
            try:
                output = layer.forward(current_input)
                activations.append(output)
                current_input = output
                logger.debug(f"Completed layer {i+1}/{len(self.layers)}")
            except Exception as e:
                logger.error(f"Forward pass failed at layer {i+1}: {e}")
                raise

        # Final result packaging
        result = ForwardPassResult(
            output=activations[-1],
            activations=tuple(activations),
            input_shape=inputs.shape,
            network_config=self.config
        )

        logger.info(f"Forward pass completed: {inputs.shape} -> {result.output.shape}")
        return result

class VisualizationEngine:
    """Handles network visualization with production-quality rendering."""

    @staticmethod
    def plot_activation_function(activation: ActivationFunction):
        """Plot activation function with mathematical annotations."""
        z = np.linspace(-10, 10, 200)
        a = activation.forward(z.reshape(1, -1)).flatten()

        plt.figure(figsize=(8, 6))
        plt.plot(z, a, 'b-', linewidth=2, label=f"{activation.name()}(z)")
        plt.axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Decision boundary")
        plt.title(f"{activation.name().capitalize()} Activation Function")
        plt.xlabel("z (pre-activation)")
        plt.ylabel(f"{activation.name()}(z)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_network_architecture(layer_dims: Tuple[int, ...]):
        """Visualize network topology."""
        plt.figure(figsize=(10, 8))
        n_layers = len(layer_dims)
        layer_x = np.linspace(0, n_layers - 1, n_layers)

        for i, layer_size in enumerate(layer_dims):
            y_positions = np.linspace(0, 1, layer_size)
            plt.scatter([layer_x[i]] * layer_size, y_positions, s=600,
                       label=f"Layer {i} ({layer_size} neurons)", alpha=0.8)

            if i > 0:
                prev_y_positions = np.linspace(0, 1, layer_dims[i-1])
                for y1 in prev_y_positions[:min(10, len(prev_y_positions))]:  # Limit connections for clarity
                    for y2 in y_positions[:min(10, len(y_positions))]:
                        plt.plot([layer_x[i-1], layer_x[i]], [y1, y2],
                               "k-", alpha=0.1, linewidth=0.5)

        plt.title("Neural Network Architecture", fontsize=14, fontweight='bold')
        plt.axis("off")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_forward_flow(result: ForwardPassResult):
        """Visualize activation flows through network."""
        fig, axes = plt.subplots(1, len(result.activations), figsize=(15, 5))

        layer_names = [f"Hidden Layer {i+1}" for i in range(len(result.activations)-1)]
        layer_names.append("Output Layer")

        for ax, activation, name in zip(axes, result.activations, layer_names):
            im = ax.imshow(activation, cmap="viridis", aspect="auto", interpolation='nearest')
            ax.set_title(f"{name}\nShape: {activation.shape}")
            ax.set_xlabel("Samples")
            ax.set_ylabel("Neurons")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle("Forward Propagation Activations", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

def run_validation_tests():
    """Execute comprehensive validation tests."""
    logger.info("Running validation tests...")

    # Test 1: Basic functionality
    config = NetworkConfig(layer_dims=(2, 3, 1))
    network = NeuralNetwork(config)
    X = np.random.randn(2, 5)
    result = network.forward(X)

    assert result.output.shape == (1, 5), f"Output shape mismatch: {result.output.shape}"
    assert len(result.activations) == 2, f"Activations count mismatch: {len(result.activations)}"

    # Test 2: Input validation
    try:
        network.forward(np.array([1, 2, 3]))  # Wrong shape
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Test 3: Numerical stability
    large_input = np.array([[1000.0], [1000.0]])
    result_large = network.forward(large_input)
    assert np.all(np.isfinite(result_large.output)), "Output should be finite"

    logger.info("All validation tests passed!")

def main():
    """Main execution with comprehensive telemetry."""
    logger.info(f"Starting Neural Network Forward Pass v{VERSION}")

    # Configuration
    config = NetworkConfig(
        layer_dims=(2, 4, 3, 1),
        random_seed=DEFAULT_RANDOM_SEED
    )

    # Initialize network
    network = NeuralNetwork(config)

    # Generate test input
    X = np.random.RandomState(config.random_seed).randn(config.layer_dims[0], 10)

    # Execute forward pass
    result = network.forward(X)

    # Output results
    print(f"Network Configuration: {config.layer_dims}")
    print(f"Input Shape: {result.input_shape}")
    print(f"Output Shape: {result.output.shape}")
    print(f"Sample Output: {result.output[:, :3]}")

    # Visualizations
    vis_engine = VisualizationEngine()
    vis_engine.plot_activation_function(SigmoidActivation())
    vis_engine.plot_network_architecture(config.layer_dims)
    vis_engine.plot_forward_flow(result)

    logger.info("Execution completed successfully")

if __name__ == "__main__":
    # Run validation first
    run_validation_tests()

    # Execute main workflow
    main()
