# Author: kira-ml (GitHub)
# Educational Implementation: Multi-Layer Perceptron (MLP) with Analysis Tools

# Import core libraries for numerical computation and visualization
# NumPy provides efficient array operations - essential for ML
# Matplotlib enables data visualization - crucial for understanding model behavior
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
# In ML, reproducibility is crucial - same initialization leads to same results
np.random.seed(42)

# ============================================================================
# ACTIVATION FUNCTION: ReLU
# ============================================================================

def relu(x: np.ndarray) -> np.ndarray:
    """
    Apply ReLU activation function element-wise.
    
    Think of ReLU as a gate: if input > 0, let it pass; otherwise block it.
    This simple non-linearity is what allows neural networks to learn complex patterns.
    """
    # max(0, x) is computationally efficient - no expensive exponentials like sigmoid
    return np.maximum(0, x)

# ============================================================================
# LAYER OPERATIONS
# ============================================================================

def dense_forward(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Perform dense/linear transformation: y = x @ weights + bias.
    
    This is the fundamental building block of neural networks:
    - weights: learn how to combine input features
    - bias: adds flexibility (like the intercept in linear regression)
    - @ operator: matrix multiplication (combines features in linear way)
    """
    # Matrix multiplication followed by bias addition
    # x shape: (batch_size, input_features) - a batch of samples
    # weights shape: (input_features, output_features) - learned patterns
    # Result: each sample gets transformed to output_features dimension
    return x @ weights + bias

def init_dense_layer(input_dim: int, output_dim: int):
    """
    Initialize weights and biases for a dense layer.
    
    Good initialization is crucial - think of it as starting your learning journey
    from a good position rather than a random spot in the wilderness.
    
    Xavier initialization scales weights based on input dimension to prevent
    signals from getting too big (exploding) or too small (vanishing) as they
    pass through the network.
    """
    # Weights: random values scaled by 1/sqrt(input_dim)
    # Why divide by sqrt(input_dim)? To maintain consistent variance across layers
    weights = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
    
    # Biases: initialized to zeros (common practice)
    # Biases can learn offsets during training, starting at 0 is neutral
    bias = np.zeros(output_dim)
    
    return weights, bias

# ============================================================================
# MULTI-LAYER PERCEPTRON (MLP) FUNCTIONS
# ============================================================================

def mlp_forward(x, params):
    """
    Forward pass through 3-layer MLP: Input → [Linear + ReLU] → [Linear + ReLU] → Linear → Output.
    
    This is the "inference" step - passing data through the network to get predictions.
    The pattern: Linear transform → Non-linearity → Linear transform → Non-linearity → Final linear
    
    Why ReLU after each linear layer? Without non-linearities, multiple linear layers
    would collapse into a single linear layer (no added complexity).
    """
    # Layer 1: Linear combination + ReLU activation
    # z1 represents pre-activation values (raw linear combination)
    z1 = dense_forward(x, params['W1'], params['b1'])
    # h1 represents post-activation values (after ReLU gating)
    h1 = relu(z1)

    # Layer 2: Same pattern - building hierarchical features
    # Each layer learns increasingly complex representations of the input
    z2 = dense_forward(h1, params['W2'], params['b2'])
    h2 = relu(z2)

    # Layer 3: Final linear layer (no activation)
    # For regression tasks or when followed by task-specific activation (e.g., softmax)
    output = dense_forward(h2, params['W3'], params['b3'])

    # Return both final output and intermediate values
    # Intermediate values are needed for backpropagation (calculating gradients)
    return output, (x, z1, h1, z2, h2, output)

def init_mlp(input_dim, hidden_dim, output_dim):
    """
    Initialize parameters for a 3-layer MLP.
    
    Network architecture:
    Input (input_dim) → Hidden1 (hidden_dim) → Hidden2 (hidden_dim) → Output (output_dim)
    
    Each layer gets its own weight matrix and bias vector, all properly initialized.
    """
    return {
        # Layer 1: Input to first hidden layer
        'W1': np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim),
        'b1': np.zeros(hidden_dim),
        
        # Layer 2: Hidden to hidden (same dimension)
        'W2': np.random.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim),
        'b2': np.zeros(hidden_dim),
        
        # Layer 3: Hidden to output
        'W3': np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim),
        'b3': np.zeros(output_dim),
    }

# ============================================================================
# ANALYSIS AND DIAGNOSTIC FUNCTIONS
# ============================================================================

def layer_param_count(W: np.ndarray, b: np.ndarray) -> int:
    """
    Count total parameters in a dense layer.
    
    Parameters = weights + biases
    More parameters = more capacity to learn, but also more risk of overfitting
    and more computation/memory required.
    """
    # Each weight and bias is a learnable parameter
    # Total parameters = all elements in W + all elements in b
    return W.size + b.size

def layer_flops(input_dim: int, output_dim: int) -> int:
    """
    Calculate FLOPs (Floating Point Operations) for a dense layer.
    
    FLOPs estimate computational cost. For each output element:
    - Need input_dim multiplications (weight * input)
    - Need input_dim-1 additions (summing products)
    - Plus 1 addition for bias
    Total ≈ 2 * input_dim * output_dim operations
    
    Why care? Helps understand if model will run fast enough on your hardware.
    """
    # Simplified calculation: 2 operations per weight connection
    return 2 * input_dim * output_dim

def activation_stats(name: str, x: np.ndarray):
    """
    Compute statistics for an activation tensor.
    
    Monitoring these stats during training helps catch problems:
    - Mean close to 0: good, network isn't biased
    - Std around 1: good, activations aren't exploding/vanshing
    - Sparsity: fraction of zeros (high sparsity in ReLU = many "dead" neurons)
    """
    # Calculate basic statistics
    mean = x.mean()          # Average activation value
    std = x.std()            # How spread out are the activations?
    sparsity = np.mean(x == 0)  # Percentage of neurons outputting 0 (ReLU-specific)
    
    return {
        'name': name,
        'mean': mean,
        'std': std,
        'sparsity': sparsity,
        'shape': x.shape
    }

def visualize_activations(activations_dict):
    """
    Visualize activation distributions across network layers.
    
    Histograms show how values are distributed - looking for:
    - Normal-like distributions: good initialization
    - All zeros or all large values: problematic
    - Changing distributions across layers: expected (network transforming data)
    """
    # Create 2x3 grid of subplots (for 6 layers in our MLP)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()  # Convert 2D grid to 1D list for easy iteration
    
    # Get layer names from dictionary keys
    layer_names = list(activations_dict.keys())
    
    # Create histogram for each layer's activations
    for i, name in enumerate(layer_names):
        ax = axes[i]
        activation = activations_dict[name]
        
        # Flatten to 1D for histogram, sample if too large for performance
        flat_act = activation.flatten()
        if len(flat_act) > 1000:
            flat_act = np.random.choice(flat_act, size=1000, replace=False)
        
        # Plot histogram
        ax.hist(flat_act, bins=50, alpha=0.7, color=plt.cm.Set1(i))
        ax.set_title(f'{name}\nShape: {activation.shape}')
        ax.set_xlabel('Activation Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add statistics as text on plot
        stats = activation_stats(name, activation)
        stats_text = f'μ={stats["mean"]:.3f}\nσ={stats["std"]:.3f}\nSparsity={stats["sparsity"]:.1%}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.suptitle('MLP Activation Distributions - Layer-by-Layer Analysis', y=1.02, fontsize=16)
    plt.show()

def visualize_sparsity_patterns(activations_dict):
    """
    Visualize sparsity patterns from ReLU activations.
    
    Heatmaps show which neurons are active (non-zero) for which samples.
    Patterns can reveal:
    - Dead neurons: always zero columns
    - Rarely active neurons: mostly zero columns
    - Sample-specific patterns: different rows activate different neurons
    """
    # Filter for ReLU layers only
    relu_layers = {k: v for k, v in activations_dict.items() if 'ReLU' in k}
    
    if not relu_layers:
        print("No ReLU activation layers found for sparsity visualization.")
        return
    
    # Create subplots - one for each ReLU layer
    num_relu_layers = len(relu_layers)
    fig, axes = plt.subplots(1, num_relu_layers, figsize=(8*num_relu_layers, 6))
    if num_relu_layers == 1:
        axes = [axes]  # Make it iterable even if single plot
    
    for i, (name, activation) in enumerate(relu_layers.items()):
        ax = axes[i]
        
        # Sample data for visualization if too large
        vis_data = activation
        if activation.shape[0] > 50:  # Limit to first 50 samples
            vis_data = activation[:50, :]
        
        # Create heatmap - blue intensity = activation strength
        im = ax.imshow(vis_data, cmap='Blues', aspect='auto', interpolation='nearest')
        ax.set_title(f'{name} - Sparsity Pattern\n'
                    f'Sparsity: {activation_stats(name, activation)["sparsity"]:.1%}')
        ax.set_xlabel('Neuron Index')
        ax.set_ylabel('Sample Index')
        
        # Add colorbar legend
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Activation Value')
    
    plt.tight_layout()
    plt.suptitle('ReLU Sparsity Patterns - Information Flow Analysis', y=1.02, fontsize=16)
    plt.show()

def visualize_model_complexity(params, input_dim, hidden_dim, output_dim, batch_size):
    """
    Visualize parameter counts and FLOPs per layer.
    
    Two key metrics for model analysis:
    1. Parameters: Memory/storage requirement, model capacity
    2. FLOPs: Computational requirement, inference speed
    
    Helps answer: "Is this model too big for my application?"
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # ===== LEFT PLOT: Parameter Counts =====
    param_counts = [
        layer_param_count(params['W1'], params['b1']),
        layer_param_count(params['W2'], params['b2']),
        layer_param_count(params['W3'], params['b3'])
    ]
    layer_names = ['Layer 1 (Input→Hidden)', 'Layer 2 (Hidden→Hidden)', 'Layer 3 (Hidden→Output)']
    
    bars1 = ax1.bar(layer_names, param_counts, color=['skyblue', 'lightgreen', 'salmon'])
    ax1.set_title('Parameter Count per Layer\n(Model Size Analysis)')
    ax1.set_ylabel('Number of Parameters')
    ax1.set_yscale('log')  # Log scale helps compare vastly different numbers
    ax1.tick_params(axis='x', rotation=45)
    
    # Add exact numbers on bars
    for bar, count in zip(bars1, param_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{count:,}', ha='center', va='bottom')
    
    # ===== RIGHT PLOT: FLOPs =====
    # FLOPs scale with batch size - more samples = more computation
    flops = [
        layer_flops(input_dim, hidden_dim) * batch_size,
        layer_flops(hidden_dim, hidden_dim) * batch_size,
        layer_flops(hidden_dim, output_dim) * batch_size
    ]
    
    bars2 = ax2.bar(layer_names, flops, color=['skyblue', 'lightgreen', 'salmon'])
    ax2.set_title(f'FLOPs per Layer (Batch Size: {batch_size})\n(Computational Complexity)')
    ax2.set_ylabel('Number of FLOPs')
    ax2.set_yscale('log')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, count in zip(bars2, flops):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{count:,}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.suptitle('Model Complexity Analysis - Size vs. Computation', y=1.02, fontsize=16)
    plt.show()

def print_activation_summary(activations_dict):
    """
    Print formatted table of activation statistics.
    
    Quick text-based overview - useful for logging during training
    or when you don't have visualization capabilities.
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

# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def main():
    """
    Main demonstration of MLP forward pass with comprehensive analysis.
    
    This function showcases a complete workflow from initialization to analysis.
    It's designed to teach through example - run this to see all concepts in action.
    """
    print("=== Multi-Layer Perceptron (MLP) Forward Pass Demonstration ===")
    print("This example demonstrates core deep learning concepts with analysis tools.\n")
    
    # ===== 1. MODEL CONFIGURATION =====
    # These hyperparameters define our network architecture
    # Small values for demonstration - real networks often have 100s or 1000s of neurons
    input_dim = 16      # Number of input features (e.g., pixels, measurements)
    hidden_dim = 32     # Number of neurons in hidden layers
    output_dim = 8      # Number of outputs (e.g., classes, predictions)
    batch_size = 16     # Number of samples processed together
    
    print(f"Model Configuration:")
    print(f"  Input dimension: {input_dim} (e.g., features in a dataset)")
    print(f"  Hidden dimensions: {hidden_dim} (internal representation size)")
    print(f"  Output dimension: {output_dim} (e.g., number of classes)")
    print(f"  Batch size: {batch_size} (samples processed together)\n")
    
    # ===== 2. INITIALIZE MODEL =====
    # Create the learnable parameters (weights and biases)
    params = init_mlp(input_dim, hidden_dim, output_dim)
    print(f"✓ Initialized 3-layer MLP with Xavier initialization.")
    
    # ===== 3. CREATE SYNTHETIC DATA =====
    # Generate random data that mimics real dataset batches
    # In practice, you'd load actual data here (images, text, sensor readings, etc.)
    x = np.random.randn(batch_size, input_dim)
    print(f"✓ Generated synthetic input data: shape {x.shape}")
    
    # ===== 4. FORWARD PASS =====
    # Pass data through the network - this is "inference" or "prediction"
    output, (x_in, z1, h1, z2, h2, out) = mlp_forward(x, params)
    print(f"✓ Forward pass completed. Output shape: {output.shape}")
    
    # ===== 5. ORGANIZE ACTIVATIONS FOR ANALYSIS =====
    # Collect all layer outputs for visualization and analysis
    activations = {
        'Input': x_in,
        'Layer 1 (Linear)': z1,    # Linear combination before ReLU
        'Layer 1 (ReLU)': h1,      # After ReLU activation
        'Layer 2 (Linear)': z2,
        'Layer 2 (ReLU)': h2,
        'Output': out              # Final network prediction
    }
    
    # ===== 6. CALCULATE MODEL COMPLEXITY =====
    # How big and computationally expensive is our model?
    total_params = sum([
        layer_param_count(params['W1'], params['b1']),
        layer_param_count(params['W2'], params['b2']),
        layer_param_count(params['W3'], params['b3'])
    ])
    
    total_flops = sum([
        layer_flops(input_dim, hidden_dim) * batch_size,
        layer_flops(hidden_dim, hidden_dim) * batch_size,
        layer_flops(hidden_dim, output_dim) * batch_size
    ])
    
    print(f"\nModel Complexity Analysis:")
    print(f"  Total parameters: {total_params:,} (affects model capacity and memory)")
    print(f"  Total FLOPs per batch: {total_flops:,} (affects computation time)")
    
    # ===== 7. TEXT-BASED ANALYSIS =====
    # Quick look at activation statistics
    print_activation_summary(activations)
    
    # ===== 8. VISUALIZATIONS =====
    # Visual analysis provides intuitive understanding
    print("\nGenerating visualizations for deeper analysis...")
    print("  - Activation Distributions: Shows how values are distributed across layers")
    print("  - Sparsity Patterns: Reveals ReLU behavior and potential dead neurons")
    print("  - Model Complexity: Parameters and computational cost per layer\n")
    
    # Show activation distributions (histograms)
    visualize_activations(activations)
    
    # Show sparsity patterns (heatmaps)
    visualize_sparsity_patterns(activations)
    
    # Show model complexity (bar charts)
    visualize_model_complexity(params, input_dim, hidden_dim, output_dim, batch_size)
    
    print(f"\n✓ Final output shape: {output.shape}")
    print("✓ Demonstration complete. This workflow is typical for analyzing neural networks.")
    print("\nKey takeaways:")
    print("  1. Proper initialization prevents gradient issues")
    print("  2. Activation monitoring catches training problems early")
    print("  3. Complexity analysis informs deployment decisions")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # When this script is run directly (not imported), execute the main function
    # This is a common Python pattern - allows file to be both imported and run
    main()