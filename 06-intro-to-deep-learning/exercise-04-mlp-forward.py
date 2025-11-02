# Author: kira-ml (GitHub)
# Educational Implementation: Multi-Layer Perceptron (MLP) with Analysis Tools
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)


def relu(x: np.ndarray) -> np.ndarray:
    """
    Apply ReLU activation function element-wise.

    ReLU (Rectified Linear Unit) is the most widely used activation function
    in deep learning due to its simplicity and effectiveness. It introduces
    non-linearity while being computationally efficient and helping mitigate
    the vanishing gradient problem common in deep networks with sigmoid/tanh.

    Mathematical form: f(x) = max(0, x)
    - For positive inputs: f(x) = x (identity function)
    - For negative inputs: f(x) = 0 (zero function)
    - Derivative: f'(x) = 1 if x > 0, else 0 (non-differentiable at x=0)

    Key advantages:
    - Computationally efficient (no exponentials or divisions)
    - Helps with gradient flow in deep networks (mitigates vanishing gradients)
    - Biologically plausible (neurons fire or don't fire)

    Key considerations:
    - Can cause "dying ReLU" problem (neurons stuck at 0)
    - Not zero-centered (can affect optimization dynamics)

    Args:
        x: Input tensor of any shape

    Returns:
        np.ndarray: ReLU activated tensor with same shape as input
    """
    return np.maximum(0, x)


def dense_forward(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Perform dense/linear transformation: y = x @ weights + bias.

    This is the fundamental operation in fully connected (dense) layers. The matrix
    multiplication (x @ weights) computes linear combinations of input features
    using learned weights. The bias term allows the model to fit data that doesn't
    pass through the origin, adding an offset to each output dimension.

    In mathematical terms:
    - x: (batch_size, input_dim) - batch of input samples
    - weights: (input_dim, output_dim) - learnable parameters
    - bias: (output_dim,) - learnable bias vector
    - Output: (batch_size, output_dim) - transformed batch

    In ML frameworks, this is typically optimized using BLAS (Basic Linear Algebra Subprograms)
    libraries for maximum computational efficiency. The @ operator in NumPy performs
    efficient matrix multiplication using underlying optimized libraries.

    Args:
        x: Input tensor of shape (batch_size, input_dim)
        weights: Weight matrix of shape (input_dim, output_dim)
        bias: Bias vector of shape (output_dim,)

    Returns:
        np.ndarray: Output tensor of shape (batch_size, output_dim)
    """
    return x @ weights + bias


def init_dense_layer(input_dim: int, output_dim: int):
    """
    Initialize weights and biases for a dense layer using Xavier/Glorot initialization.

    Xavier/Glorot initialization is crucial for maintaining signal variance
    across layers during both forward and backward propagation. It prevents neurons
    from saturating and helps maintain gradient flow, leading to more stable training.

    The initialization strategy:
    - Weights: sampled from normal distribution with std = sqrt(1/input_dim)
    - Biases: initialized to zero (standard practice, as weights can learn the offset)

    The variance scaling factor sqrt(1/input_dim) ensures that:
    - Forward pass: signal variance is preserved across layers
    - Backward pass: gradient variance is preserved across layers
    This prevents the signal from exploding or vanishing as it propagates.

    Alternative: For ReLU activations, He initialization (sqrt(2/input_dim)) often works better.

    Args:
        input_dim: Number of input features to the layer
        output_dim: Number of output features from the layer

    Returns:
        tuple: (weights, bias) initialized parameters
    """
    weights = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
    bias = np.zeros(output_dim)
    return weights, bias


def mlp_forward(x, params):
    """
    Forward pass through 3-layer MLP: affine → ReLU → affine → ReLU → affine.

    This represents a standard feedforward architecture commonly used in
    deep learning. The architecture is:
    Input → [Dense + ReLU] → [Dense + ReLU] → [Dense] → Output

    The ReLU activations after each hidden layer introduce non-linearity,
    enabling the network to learn complex, non-linear patterns. The final
    layer has no activation (linear), which is typical for regression tasks
    or when followed by a task-specific activation (e.g., softmax).

    The architecture follows best practices:
    - Alternating linear transformations (Dense) and non-linear activations (ReLU)
    - Progressive dimensionality change (input → hidden → hidden → output)
    - Proper parameter organization for easy access and gradient computation

    Args:
        x: Input tensor of shape (batch_size, input_dim)
        params: Dictionary containing layer parameters (W1, b1, W2, b2, W3, b3)

    Returns:
        tuple: (output, intermediates) where intermediates contains all layer activations
               intermediates = (x, z1, h1, z2, h2, output) for potential backprop use
    """
    # Layer 1: Affine transformation (linear combination) + ReLU activation
    z1 = dense_forward(x, params['W1'], params['b1'])  # Linear: W1*x + b1
    h1 = relu(z1)                                      # Non-linear activation

    # Layer 2: Affine transformation + ReLU activation
    z2 = dense_forward(h1, params['W2'], params['b2']) # Linear: W2*h1 + b2
    h2 = relu(z2)                                      # Non-linear activation

    # Layer 3: Final affine transformation (no activation - linear output)
    output = dense_forward(h2, params['W3'], params['b3'])  # Linear: W3*h2 + b3

    return output, (x, z1, h1, z2, h2, output)


def init_mlp(input_dim, hidden_dim, output_dim):
    """
    Initialize parameters for a 3-layer MLP with Xavier initialization.

    This function sets up the weights and biases for a network with:
    - Layer 1: input_dim → hidden_dim
    - Layer 2: hidden_dim → hidden_dim (same hidden size)
    - Layer 3: hidden_dim → output_dim

    Uses Xavier initialization for all layers to maintain proper signal
    propagation. The initialization is scaled appropriately for the input
    dimension of each layer to prevent vanishing/exploding gradients.

    Parameter organization follows standard ML framework conventions where
    weights and biases are grouped by layer for easy access during training
    and inference.

    Args:
        input_dim: Dimension of input features
        hidden_dim: Dimension of hidden layers
        output_dim: Dimension of output

    Returns:
        dict: Dictionary of initialized parameters {W1, b1, W2, b2, W3, b3}
    """
    return {
        'W1': np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim),     # Scale by input_dim
        'b1': np.zeros(hidden_dim),
        'W2': np.random.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim),   # Scale by hidden_dim
        'b2': np.zeros(hidden_dim),
        'W3': np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim),   # Scale by hidden_dim
        'b3': np.zeros(output_dim),
    }


def layer_param_count(W: np.ndarray, b: np.ndarray) -> int:
    """
    Count total parameters in a dense layer (weights + biases).

    Essential for model complexity analysis, memory estimation, and performance planning.
    In production ML systems, parameter counting helps with:
    - Model size constraints for deployment (e.g., mobile devices, edge computing)
    - Memory allocation planning (GPU/CPU memory requirements)
    - Computational resource estimation (training/inference time)
    - Model selection and architecture comparison

    Args:
        W: Weight matrix of shape (input_dim, output_dim)
        b: Bias vector of shape (output_dim,)

    Returns:
        int: Total number of parameters in the layer = W.size + b.size
    """
    return W.size + b.size


def layer_flops(input_dim: int, output_dim: int) -> int:
    """
    Calculate FLOPs (Floating Point Operations) for a dense layer: 2 * input_dim * output_dim.

    FLOP counting is crucial for performance optimization in ML systems. For a dense layer
    performing matrix multiplication (input_dim × output_dim):
    - Each output element requires input_dim multiplications and (input_dim-1) additions
    - Total ≈ input_dim * output_dim * 2 FLOPs (simplified approximation)
    - Multiplication: input_dim * output_dim ops
    - Addition: input_dim * output_dim ops (for bias and accumulation)

    This metric helps compare computational efficiency of different architectures,
    estimate inference latency, and optimize models for deployment constraints.

    Args:
        input_dim: Number of input features to the layer
        output_dim: Number of output features from the layer

    Returns:
        int: Approximate number of floating point operations for one forward pass
    """
    return 2 * input_dim * output_dim


def activation_stats(name: str, x: np.ndarray):
    """
    Compute and return statistics for an activation tensor.

    Activation statistics are critical for diagnosing training issues and monitoring
    model health during development and deployment:
    - Mean and std help identify vanishing/exploding activations (should be stable)
    - Sparsity indicates ReLU dead neuron problem (high sparsity is problematic)
    - Shape verification ensures proper tensor flow and architecture integrity

    These metrics are commonly monitored in production ML systems to
    detect model degradation, training instability, or architectural bugs.

    Args:
        name: Descriptive name for the layer (for identification in logs)
        x: Activation tensor from a specific layer

    Returns:
        dict: Dictionary containing statistical measures:
              - 'name': Layer name
              - 'mean': Average activation value
              - 'std': Standard deviation of activations
              - 'sparsity': Fraction of zero elements (important for ReLU)
              - 'shape': Shape of the activation tensor
    """
    mean = x.mean()
    std = x.std()
    sparsity = np.mean(x == 0)  # Fraction of zero elements (key for ReLU layers)
    return {
        'name': name,
        'mean': mean,
        'std': std,
        'sparsity': sparsity,
        'shape': x.shape
    }


def visualize_activations(activations_dict):
    """
    Visualize activation distributions and sparsity patterns across network layers.

    Activation distribution visualization helps identify:
    - Saturation (activations clustering at extremes, common with sigmoid/tanh)
    - Dead neurons (excessive sparsity in ReLU layers, sparsity > 50-70% is concerning)
    - Proper initialization (mean ≈ 0, std ≈ 1 in early layers is ideal)
    - Gradient flow issues (distributions changing drastically between layers)

    Histograms provide intuitive understanding of activation behavior
    across different network layers, which is crucial for debugging and optimization.

    Args:
        activations_dict: Dictionary mapping descriptive layer names to activation tensors
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # Layer names should correspond to keys in activations_dict
    layer_names = list(activations_dict.keys())

    for i, name in enumerate(layer_names):
        ax = axes[i]
        activation = activations_dict[name]

        # Flatten activation for histogram (sample if too large for performance)
        flat_act = activation.flatten()
        if len(flat_act) > 1000:
            flat_act = np.random.choice(flat_act, size=1000, replace=False)

        # Plot histogram to show distribution
        ax.hist(flat_act, bins=50, alpha=0.7, color=plt.cm.Set1(i))
        ax.set_title(f'{name}\nShape: {activation.shape}')
        ax.set_xlabel('Activation Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

        # Add computed statistics as text overlay
        stats = activation_stats(name, activation)
        stats_text = f'μ={stats["mean"]:.3f}\nσ={stats["std"]:.3f}\nSparsity={stats["sparsity"]:.1%}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.suptitle('MLP Activation Distributions - Layer-by-Layer Analysis', y=1.02, fontsize=16)
    plt.show()


def visualize_sparsity_patterns(activations_dict):
    """
    Visualize sparsity patterns introduced by ReLU activations using heatmaps.

    ReLU sparsity patterns reveal important training dynamics:
    - High sparsity (>50%) may indicate dead neurons (neurons stuck at 0)
    - Structured sparsity patterns can suggest initialization issues or poor training
    - Row-wise sparsity shows per-sample activation behavior across neurons
    - Column-wise sparsity shows neuron-wise activation across samples

    This visualization is particularly useful for debugging ReLU-based networks
    and understanding how information flows through the network.

    Args:
        activations_dict: Dictionary containing ReLU activation tensors (layers with ReLU)
    """
    # Identify ReLU layers (those with significant sparsity)
    relu_layers = {k: v for k, v in activations_dict.items() if 'ReLU' in k}
    if not relu_layers:
        print("No ReLU activation layers found for sparsity visualization.")
        return

    num_relu_layers = len(relu_layers)
    fig, axes = plt.subplots(1, num_relu_layers, figsize=(8*num_relu_layers, 6))
    if num_relu_layers == 1:
        axes = [axes]

    for i, (name, activation) in enumerate(relu_layers.items()):
        ax = axes[i]

        # Sample data if too large for efficient visualization
        vis_data = activation
        if activation.shape[0] > 50:  # Limit to first 50 samples
            vis_data = activation[:50, :]

        # Create heatmap to visualize activation patterns
        im = ax.imshow(vis_data, cmap='Blues', aspect='auto', interpolation='nearest')
        ax.set_title(f'{name} - Sparsity Pattern\n'
                    f'Sparsity: {activation_stats(name, activation)["sparsity"]:.1%}')
        ax.set_xlabel('Neuron Index')
        ax.set_ylabel('Sample Index')

        # Add colorbar to indicate activation value scale
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Activation Value')

    plt.tight_layout()
    plt.suptitle('ReLU Sparsity Patterns - Information Flow Analysis', y=1.02, fontsize=16)
    plt.show()


def visualize_model_complexity(params, input_dim, hidden_dim, output_dim, batch_size):
    """
    Visualize parameter counts and FLOPs per layer for model complexity analysis.

    Model complexity analysis is essential for:
    - Deployment resource planning (memory, compute, latency)
    - Architecture comparison and selection
    - Performance optimization and bottlenecks identification
    - Memory requirement estimation for training/inference
    - Understanding model capacity vs. generalization trade-off

    Log-scale visualization helps compare layers with vastly different parameter/FLOP scales.

    Args:
        params: Dictionary of model parameters {W1, b1, W2, b2, W3, b3}
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        batch_size: Batch size for FLOP calculation (affects per-batch computation)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Calculate parameter counts per layer
    param_counts = [
        layer_param_count(params['W1'], params['b1']),
        layer_param_count(params['W2'], params['b2']),
        layer_param_count(params['W3'], params['b3'])
    ]
    layer_names = ['Layer 1 (Input→Hidden)', 'Layer 2 (Hidden→Hidden)', 'Layer 3 (Hidden→Output)']

    # Plot parameter counts
    bars1 = ax1.bar(layer_names, param_counts, color=['skyblue', 'lightgreen', 'salmon'])
    ax1.set_title('Parameter Count per Layer\n(Model Size Analysis)')
    ax1.set_ylabel('Number of Parameters')
    ax1.set_yscale('log')  # Log scale for better comparison
    ax1.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, count in zip(bars1, param_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{count:,}', ha='center', va='bottom')

    # Calculate FLOPs per layer (for one forward pass on a batch)
    flops = [
        layer_flops(input_dim, hidden_dim) * batch_size,  # Scaled by batch size
        layer_flops(hidden_dim, hidden_dim) * batch_size,
        layer_flops(hidden_dim, output_dim) * batch_size
    ]

    # Plot FLOPs
    bars2 = ax2.bar(layer_names, flops, color=['skyblue', 'lightgreen', 'salmon'])
    ax2.set_title(f'FLOPs per Layer (Batch Size: {batch_size})\n(Computational Complexity)')
    ax2.set_ylabel('Number of FLOPs')
    ax2.set_yscale('log')  # Log scale for better comparison
    ax2.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, count in zip(bars2, flops):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{count:,}', ha='center', va='bottom')

    plt.tight_layout()
    plt.suptitle('Model Complexity Analysis - Size vs. Computation', y=1.02, fontsize=16)
    plt.show()


def print_activation_summary(activations_dict):
    """
    Print a formatted summary table of activation statistics for all layers.

    Tabular format provides quick overview of network health:
    - Balanced mean/std indicate proper initialization and gradient flow
    - Low sparsity in linear layers (close to 0%), controlled sparsity in ReLU layers (up to ~50%)
    - Consistent shapes verify proper tensor flow and architecture integrity

    This summary is invaluable for debugging, monitoring training progress,
    and ensuring model stability during development and deployment.

    Args:
        activations_dict: Dictionary mapping layer names to activation tensors
    """
    print("\n" + "="*100)
    print("ACTIVATION STATISTICS SUMMARY - Layer Health Check")
    print("="*100)
    print(f"{'Layer Name':<25} {'Shape':<18} {'Mean':<10} {'Std Dev':<10} {'Sparsity':<10}")
    print("-"*100)

    for name, activation in activations_dict.items():
        stats = activation_stats(name, activation)
        print(f"{stats['name']:<25} {str(stats['shape']):<18} "
              f"{stats['mean']:<10.4f} {stats['std']:<10.4f} {stats['sparsity']:<10.2%}")

    print("="*100)
    print("Interpretation Guide:")
    print("  - Mean ≈ 0, Std ≈ 1: Good initialization/flow")
    print("  - Sparsity > 0.5 in ReLU layers: Potential dead neurons")
    print("  - Sparsity > 0.1 in Linear layers: Unusual, check for errors")
    print("="*100)


def main():
    """
    Main demonstration function showcasing MLP forward pass with comprehensive analysis tools.

    This comprehensive example demonstrates a complete workflow for:
    1. Proper model initialization with Xavier scaling (prevents gradient issues)
    2. Forward pass through multi-layer architecture (standard deep learning pattern)
    3. Activation analysis for training diagnostics (identifies model health issues)
    4. Model complexity profiling for deployment planning (resource estimation)
    5. Visualization of key ML concepts (sparsity, distributions, FLOPs)

    The workflow mirrors production ML pipelines where models are analyzed
    for both functional correctness and computational efficiency before deployment.
    """
    print("=== Multi-Layer Perceptron (MLP) Forward Pass Demonstration ===")
    print("This example demonstrates core deep learning concepts with analysis tools.\n")

    # Model configuration - typical settings for a small demonstration network
    input_dim = 16
    hidden_dim = 32
    output_dim = 8
    batch_size = 16

    print(f"Model Configuration:")
    print(f"  Input dimension: {input_dim} (e.g., features in a dataset)")
    print(f"  Hidden dimensions: {hidden_dim} (internal representation size)")
    print(f"  Output dimension: {output_dim} (e.g., number of classes, regression targets)")
    print(f"  Batch size: {batch_size} (number of samples processed together)")

    # Initialize MLP parameters with proper initialization strategy
    params = init_mlp(input_dim, hidden_dim, output_dim)
    print(f"\nInitialized 3-layer MLP with Xavier initialization.")

    # Generate synthetic input data (simulates real data batch)
    x = np.random.randn(batch_size, input_dim)
    print(f"Generated synthetic input data: shape {x.shape}")

    # Perform forward pass through the network
    output, (x_in, z1, h1, z2, h2, out) = mlp_forward(x, params)
    print(f"Forward pass completed. Output shape: {output.shape}")

    # Organize activations for analysis and visualization
    activations = {
        'Input': x_in,
        'Layer 1 (Linear)': z1,
        'Layer 1 (ReLU)': h1,
        'Layer 2 (Linear)': z2,
        'Layer 2 (ReLU)': h2,
        'Output': out
    }

    # Calculate and display model complexity metrics
    total_params = sum([
        layer_param_count(params['W1'], params['b1']),
        layer_param_count(params['W2'], params['b2']),
        layer_param_count(params['W3'], params['b3'])
    ])

    total_flops = sum([
        layer_flops(input_dim, hidden_dim) * batch_size,  # Scaled by batch size
        layer_flops(hidden_dim, hidden_dim) * batch_size,
        layer_flops(hidden_dim, output_dim) * batch_size
    ])

    print(f"\nModel Complexity Analysis:")
    print(f"  Total parameters: {total_params:,} (affects model capacity and memory)")
    print(f"  Total FLOPs per batch: {total_flops:,} (affects computation time)")

    # Print detailed activation statistics summary
    print_activation_summary(activations)

    # Generate visualizations to understand model behavior
    print("\nGenerating visualizations for deeper analysis...")
    print("  - Activation Distributions: Shows how values are distributed across layers")
    print("  - Sparsity Patterns: Reveals ReLU behavior and potential dead neurons")
    print("  - Model Complexity: Parameters and computational cost per layer")

    # 1. Visualize activation distributions across all layers
    visualize_activations(activations)

    # 2. Analyze sparsity patterns in ReLU layers
    visualize_sparsity_patterns(activations)

    # 3. Analyze model complexity (parameters and FLOPs)
    visualize_model_complexity(params, input_dim, hidden_dim, output_dim, batch_size)

    print(f"\nFinal output shape: {output.shape}")
    print("Demonstration complete. This workflow is typical for analyzing neural networks.")


if __name__ == "__main__":
    main()