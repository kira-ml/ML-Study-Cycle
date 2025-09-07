import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)


def relu(x: np.ndarray) -> np.ndarray:
    """Apply ReLU activation function element-wise."""
    return np.maximum(0, x)


def dense_forward(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Perform dense/linear transformation: y = x @ weights + bias."""
    return x @ weights + bias


def init_dense_layer(input_dim: int, output_dim: int):
    """Initialize weights and biases for a dense layer using Xavier initialization."""
    weights = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
    bias = np.zeros(output_dim)
    return weights, bias


def mlp_forward(x, params):
    """Forward pass through 3-layer MLP: affine → ReLU → affine → ReLU → affine."""
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
    """Initialize parameters for a 3-layer MLP."""
    return {
        'W1': np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim),
        'b1': np.zeros(hidden_dim),
        'W2': np.random.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim),
        'b2': np.zeros(hidden_dim),
        'W3': np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim),
        'b3': np.zeros(output_dim),
    }


def layer_param_count(W: np.ndarray, b: np.ndarray) -> int:
    """Count total parameters in a layer (weights + biases)."""
    return W.size + b.size


def layer_flops(input_dim: int, output_dim: int) -> int:
    """Calculate FLOPs for a dense layer: 2 * input_dim * output_dim."""
    return 2 * input_dim * output_dim


def activation_stats(name: str, x: np.ndarray):
    """Compute and return statistics for an activation tensor."""
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
    """Visualize activation distributions and sparsity patterns."""
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
    """Visualize sparsity patterns introduced by ReLU activations."""
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
    """Visualize parameter counts and FLOPs per layer."""
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
    """Print a summary table of activation statistics."""
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
    """Main function to demonstrate MLP forward pass with visualizations."""
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