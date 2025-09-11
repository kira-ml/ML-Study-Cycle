# Author: kira-ml (GitHub)
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)


def relu(x: np.ndarray) -> np.ndarray:
    """
    Apply ReLU activation function element-wise.
    
    ReLU (Rectified Linear Unit) is the most widely used activation function
    in deep learning due to its simplicity and effectiveness. It introduces
    non-linearity while being computationally efficient and helping mitigate
    the vanishing gradient problem.
    
    Mathematical form: f(x) = max(0, x)
    
    Args:
        x: Input tensor of any shape
        
    Returns:
        np.ndarray: ReLU activated tensor with same shape as input
    """
    return np.maximum(0, x)


def dense_forward(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Perform dense/linear transformation: y = x @ weights + bias.
    
    This is the fundamental operation in fully connected layers. The matrix
    multiplication (x @ weights) performs linear combination of input features,
    while the bias term allows the model to fit data that doesn't pass through
    the origin.
    
    In ML frameworks, this is typically optimized using BLAS libraries for
    maximum computational efficiency.
    
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
    Initialize weights and biases for a dense layer using Xavier initialization.
    
    Xavier/Glorot initialization is crucial for maintaining signal variance
    across layers during forward and backward propagation. It prevents neurons
    from saturating and helps maintain gradient flow.
    
    Weights are sampled from normal distribution with std = sqrt(1/input_dim)
    Biases are initialized to zero, which is standard practice.
    
    Args:
        input_dim: Number of input features
        output_dim: Number of output features
        
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
    deep learning. The ReLU activations after each hidden layer introduce
    non-linearity, enabling the network to learn complex patterns.
    
    The architecture follows best practices:
    - Alternating linear transformations and non-linear activations
    - No activation on final layer (suitable for regression or pre-softmax)
    - Proper parameter organization for easy access
    
    Args:
        x: Input tensor of shape (batch_size, input_dim)
        params: Dictionary containing layer parameters (W1, b1, W2, b2, W3, b3)
        
    Returns:
        tuple: (output, intermediates) where intermediates contains all layer activations
    """
    # Layer 1: affine + ReLU
    z1 = dense_forward(x, params['W1'], params['b1'])
    h1 = relu(z1)
    
    # Layer 2: affine + ReLU
    z2 = dense_forward(h1, params['W2'], params['b2'])
    h2 = relu(z2)
    
    # Layer 3: affine (output)
    output = dense_forward(h2, params['W3'], params['b3'])
    
    return output, (x, z1, h1, z2, h2, output)


def init_mlp(input_dim, hidden_dim, output_dim):
    """
    Initialize parameters for a 3-layer MLP.
    
    Uses Xavier initialization for all layers to maintain proper signal
    propagation. Hidden layers use ReLU activation, so the initialization
    is scaled appropriately for the input dimension of each layer.
    
    Parameter organization follows standard ML framework conventions where
    weights and biases are grouped by layer for easy access during training.
    
    Args:
        input_dim: Dimension of input features
        hidden_dim: Dimension of hidden layers
        output_dim: Dimension of output
        
    Returns:
        dict: Dictionary of initialized parameters
    """
    return {
        'W1': np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim),
        'b1': np.zeros(hidden_dim),
        'W2': np.random.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim),
        'b2': np.zeros(hidden_dim),
        'W3': np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim),
        'b3': np.zeros(output_dim),
    }


def layer_param_count(W: np.ndarray, b: np.ndarray) -> int:
    """
    Count total parameters in a layer (weights + biases).
    
    Essential for model complexity analysis and memory estimation.
    In production ML systems, parameter counting helps with:
    - Model size constraints for deployment
    - Memory allocation planning
    - Computational resource estimation
    
    Args:
        W: Weight matrix
        b: Bias vector
        
    Returns:
        int: Total number of parameters in the layer
    """
    return W.size + b.size


def layer_flops(input_dim: int, output_dim: int) -> int:
    """
    Calculate FLOPs for a dense layer: 2 * input_dim * output_dim.
    
    FLOP (Floating Point Operation) counting is crucial for performance
    optimization in ML. For matrix multiplication (input_dim × output_dim):
    - Each output element requires input_dim multiplications and (input_dim-1) additions
    - Total ≈ 2 * input_dim * output_dim FLOPs
    
    This metric helps compare computational efficiency of different architectures.
    
    Args:
        input_dim: Number of input features
        output_dim: Number of output features
        
    Returns:
        int: Approximate number of floating point operations
    """
    return 2 * input_dim * output_dim


def activation_stats(name: str, x: np.ndarray):
    """
    Compute and return statistics for an activation tensor.
    
    Activation statistics are critical for diagnosing training issues:
    - Mean and std help identify vanishing/exploding activations
    - Sparsity indicates ReLU dead neuron problem
    - Shape verification ensures proper tensor flow
    
    These metrics are commonly monitored in production ML systems to
    detect model degradation or training instability.
    
    Args:
        name: Layer name for identification
        x: Activation tensor
        
    Returns:
        dict: Dictionary containing statistical measures
    """
    mean = x.mean()
    std = x.std()
    sparsity = np.mean(x == 0)  # Fraction of zero elements
    return {
        'name': name,
        'mean': mean,
        'std': std,
        'sparsity': sparsity,
        'shape': x.shape
    }


def visualize_activations(activations_dict):
    """
    Visualize activation distributions and sparsity patterns.
    
    Activation distribution visualization helps identify:
    - Saturation (activations clustering at extremes)
    - Dead neurons (excessive sparsity in ReLU layers)
    - Proper initialization (mean ≈ 0, std ≈ 1 in early layers)
    
    Histograms provide intuitive understanding of activation behavior
    across different network layers.
    
    Args:
        activations_dict: Dictionary mapping layer names to activation tensors
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    layer_names = ['Input', 'Layer 1 (Linear)', 'Layer 1 (ReLU)', 
                   'Layer 2 (Linear)', 'Layer 2 (ReLU)', 'Output']
    
    for i, (name, activation) in enumerate(activations_dict.items()):
        ax = axes[i]
        
        # Flatten activation for histogram (sample if too large)
        flat_act = activation.flatten()
        if len(flat_act) > 1000:
            flat_act = np.random.choice(flat_act, size=1000, replace=False)
        
        # Plot histogram
        ax.hist(flat_act, bins=50, alpha=0.7, color=plt.cm.Set1(i))
        ax.set_title(f'{name}\nShape: {activation.shape}')
        ax.set_xlabel('Activation Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        stats = activation_stats(name, activation)
        stats_text = f'μ={stats["mean"]:.3f}\nσ={stats["std"]:.3f}\nSparsity={stats["sparsity"]:.1%}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.suptitle('MLP Activation Distributions', y=1.02, fontsize=16)
    plt.show()


def visualize_sparsity_patterns(activations_dict):
    """
    Visualize sparsity patterns introduced by ReLU activations.
    
    ReLU sparsity patterns reveal important training dynamics:
    - High sparsity may indicate dead neurons
    - Structured sparsity patterns can suggest initialization issues
    - Row-wise sparsity shows per-sample activation behavior
    
    This visualization is particularly useful for debugging ReLU-based networks.
    
    Args:
        activations_dict: Dictionary containing ReLU activation tensors
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Get ReLU activations
    relu_layers = [('Layer 1 (ReLU)', activations_dict['Layer 1 (ReLU)']), 
                   ('Layer 2 (ReLU)', activations_dict['Layer 2 (ReLU)'])]
    
    for i, (name, activation) in enumerate(relu_layers):
        ax = axes[i]
        
        # Sample data if too large for visualization
        vis_data = activation
        if activation.shape[0] > 50:
            vis_data = activation[:50, :]  # Take first 50 samples
        
        # Create heatmap
        im = ax.imshow(vis_data, cmap='Blues', aspect='auto')
        ax.set_title(f'{name} - Sparsity Pattern\n'
                    f'Sparsity: {activation_stats(name, activation)["sparsity"]:.1%}')
        ax.set_xlabel('Neuron Index')
        ax.set_ylabel('Sample Index')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Activation Value')
    
    plt.tight_layout()
    plt.suptitle('ReLU Sparsity Patterns', y=1.02, fontsize=16)
    plt.show()


def visualize_model_complexity(params, input_dim, hidden_dim, output_dim, batch_size):
    """
    Visualize parameter counts and FLOPs per layer.
    
    Model complexity analysis is essential for:
    - Deployment resource planning
    - Architecture comparison
    - Performance optimization
    - Memory requirement estimation
    
    Log-scale visualization helps compare layers with vastly different scales.
    
    Args:
        params: Dictionary of model parameters
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        batch_size: Batch size for FLOP calculation
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Parameter counts per layer
    param_counts = [
        layer_param_count(params['W1'], params['b1']),
        layer_param_count(params['W2'], params['b2']),
        layer_param_count(params['W3'], params['b3'])
    ]
    layer_names = ['Layer 1', 'Layer 2', 'Layer 3']
    
    bars1 = ax1.bar(layer_names, param_counts, color=['skyblue', 'lightgreen', 'salmon'])
    ax1.set_title('Parameters per Layer')
    ax1.set_ylabel('Number of Parameters')
    ax1.set_yscale('log')
    
    # Add value labels on bars
    for bar, count in zip(bars1, param_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{count:,}', ha='center', va='bottom')
    
    # FLOPs per layer
    flops = [
        layer_flops(input_dim, hidden_dim),
        layer_flops(hidden_dim, hidden_dim),
        layer_flops(hidden_dim, output_dim)
    ]
    
    bars2 = ax2.bar(layer_names, flops, color=['skyblue', 'lightgreen', 'salmon'])
    ax2.set_title(f'FLOPs per Layer (Batch Size: {batch_size})')
    ax2.set_ylabel('Number of FLOPs')
    ax2.set_yscale('log')
    
    # Add value labels on bars
    for bar, count in zip(bars2, flops):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{count:,}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.suptitle('Model Complexity Analysis', y=1.02, fontsize=16)
    plt.show()


def print_activation_summary(activations_dict):
    """
    Print a summary table of activation statistics.
    
    Tabular format provides quick overview of network health:
    - Balanced mean/std indicate proper initialization
    - Low sparsity in linear layers, controlled sparsity in ReLU layers
    - Consistent shapes verify proper tensor flow
    
    This summary is invaluable for debugging and monitoring training.
    
    Args:
        activations_dict: Dictionary mapping layer names to activation tensors
    """
    print("\n" + "="*80)
    print("ACTIVATION STATISTICS SUMMARY")
    print("="*80)
    print(f"{'Layer':<20} {'Shape':<15} {'Mean':<10} {'Std':<10} {'Sparsity':<10}")
    print("-"*80)
    
    for name, activation in activations_dict.items():
        stats = activation_stats(name, activation)
        print(f"{stats['name']:<20} {str(stats['shape']):<15} "
              f"{stats['mean']:<10.4f} {stats['std']:<10.4f} {stats['sparsity']:<10.2%}")
    print("="*80)


def main():
    """
    Main function to demonstrate MLP forward pass with visualizations.
    
    This comprehensive example showcases:
    1. Proper model initialization with Xavier scaling
    2. Forward pass through multi-layer architecture
    3. Activation analysis for training diagnostics
    4. Model complexity profiling for deployment planning
    5. Visualization of key ML concepts (sparsity, distributions, FLOPs)
    
    The workflow mirrors production ML pipelines where models are analyzed
    for both functional correctness and computational efficiency.
    """
    print("=== MLP Forward Pass Demonstration ===")
    
    # Model configuration
    input_dim = 16
    hidden_dim = 32
    output_dim = 8
    batch_size = 16
    
    print(f"Model Configuration:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Hidden dimensions: {hidden_dim}")
    print(f"  Output dimension: {output_dim}")
    print(f"  Batch size: {batch_size}")
    
    # Initialize MLP
    params = init_mlp(input_dim, hidden_dim, output_dim)
    
    # Generate synthetic data
    x = np.random.randn(batch_size, input_dim)
    print(f"\nInput data shape: {x.shape}")
    
    # Forward pass
    output, (x_in, z1, h1, z2, h2, out) = mlp_forward(x, params)
    
    # Store activations for visualization
    activations = {
        'Input': x_in,
        'Layer 1 (Linear)': z1,
        'Layer 1 (ReLU)': h1,
        'Layer 2 (Linear)': z2,
        'Layer 2 (ReLU)': h2,
        'Output': out
    }
    
    # Print model complexity
    total_params = sum([
        layer_param_count(params['W1'], params['b1']),
        layer_param_count(params['W2'], params['b2']),
        layer_param_count(params['W3'], params['b3'])
    ])
    
    total_flops = sum([
        layer_flops(input_dim, hidden_dim),
        layer_flops(hidden_dim, hidden_dim),
        layer_flops(hidden_dim, output_dim)
    ])
    
    print(f"\nModel Complexity:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Total FLOPs: {total_flops:,}")
    
    # Print activation summary
    print_activation_summary(activations)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # 1. Activation distributions
    visualize_activations(activations)
    
    # 2. Sparsity patterns
    visualize_sparsity_patterns(activations)
    
    # 3. Model complexity
    visualize_model_complexity(params, input_dim, hidden_dim, output_dim, batch_size)
    
    print(f"\nFinal output shape: {output.shape}")


if __name__ == "__main__":
    main()