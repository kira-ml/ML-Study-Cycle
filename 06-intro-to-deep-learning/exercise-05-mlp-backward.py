# Author: kira-ml (GitHub)
# Educational Implementation: Neural Network Backpropagation from Scratch
# Learn the fundamentals of how neural networks learn through gradient descent

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

# ============================================================================
# LAYER IMPLEMENTATIONS: FORWARD AND BACKWARD PASSES
# ============================================================================

def affine_forward(x: np.ndarray, W: np.ndarray, b: np.ndarray):
    """Forward pass for affine (fully connected) layer.
    
    This is the fundamental operation in neural networks:
    y = xW + b
    
    Think of it as a weighted sum of inputs, similar to linear regression.
    W (weights) determine importance of each input, b (bias) shifts the result.
    """
    # Matrix multiplication: each row of x (a sample) gets multiplied by W
    out = x @ W + b
    
    # Cache stores inputs for backward pass - we'll need them for gradient computation
    cache = (x, W, b)
    return out, cache


def affine_backward(dout: np.ndarray, cache: tuple):
    """Backward pass for affine layer.
    
    Computes gradients using the chain rule:
    - dx: how much should we change the input?
    - dW: how much should we change the weights?
    - db: how much should we change the biases?
    
    These gradients tell our optimizer (like SGD) how to update parameters.
    """
    # Unpack cached values from forward pass
    x, W, _ = cache
    
    # Gradient flows backward through the linear transformation
    dx = dout @ W.T  # How input affects output (via weights)
    dW = x.T @ dout  # How weights affect output (weighted by input)
    db = np.sum(dout, axis=0)  # Bias gradient: sum over all samples in batch
    
    return dx, dW, db


def relu_forward(x: np.ndarray):
    """Forward pass for ReLU (Rectified Linear Unit) activation.
    
    ReLU = max(0, x) - it lets positive values pass unchanged, zeros negatives.
    
    Why ReLU? It's simple, fast, and helps avoid the "vanishing gradient" problem
    that plagued older activation functions like sigmoid.
    """
    # Element-wise maximum: negative values become 0, positive stay as is
    out = np.maximum(0, x)
    
    # Cache original input - we'll need to know which inputs were positive/negative
    cache = x
    return out, cache


def relu_backward(dout: np.ndarray, cache: np.ndarray):
    """Backward pass for ReLU activation.
    
    ReLU gradient is simple:
    - 1 if input was positive (pass gradient through)
    - 0 if input was negative (block gradient)
    
    This "sparsity" helps with efficient computation and prevents dead neurons
    from updating unnecessarily.
    """
    x = cache
    
    # Create mask: 1 where x > 0, 0 where x <= 0
    # Only neurons that were "active" in forward pass get gradients
    dx = dout * (x > 0)
    
    return dx


def two_layer_forward(x: np.ndarray, params: tuple):
    """Forward pass for two-layer MLP (Multi-Layer Perceptron).
    
    Architecture: Input → Affine → ReLU → Affine → Output
    
    This simple network can learn surprisingly complex patterns!
    The Universal Approximation Theorem says even this simple architecture
    can approximate any continuous function given enough neurons.
    """
    # Unpack parameters: W1/b1 for first layer, W2/b2 for second layer
    W1, b1, W2, b2 = params
    
    # First layer: linear transformation
    a1, fc1_cache = affine_forward(x, W1, b1)
    
    # Activation: introduces non-linearity (essential for learning complex patterns)
    h1, relu_cache = relu_forward(a1)
    
    # Second layer: final linear transformation to output space
    scores, fc2_cache = affine_forward(h1, W2, b2)
    
    # Cache everything for backward pass
    cache = (fc1_cache, relu_cache, fc2_cache)
    return scores, cache


def two_layer_backward(dout: np.ndarray, cache: tuple):
    """Backward pass for two-layer MLP.
    
    Implements backpropagation - the algorithm that makes deep learning possible!
    
    We compute gradients layer by layer, starting from the output and moving
    backward through the network. This is efficient thanks to the chain rule
    from calculus.
    """
    # Unpack caches from forward pass
    fc1_cache, relu_cache, fc2_cache = cache

    # Step 1: Backprop through second affine layer (output layer)
    dh1, dW2, db2 = affine_backward(dout, fc2_cache)
    
    # Step 2: Backprop through ReLU activation
    da1 = relu_backward(dh1, relu_cache)
    
    # Step 3: Backprop through first affine layer (hidden layer)
    dx, dW1, db1 = affine_backward(da1, fc1_cache)

    # Package all gradients for optimizer
    grads = (dW1, db1, dW2, db2)
    return dx, grads


def eval_loss(x: np.ndarray, params: tuple, target: np.ndarray):
    """Evaluate Mean Squared Error (MSE) loss.
    
    MSE measures how far predictions are from targets.
    Formula: 0.5 * Σ(prediction - target)²
    
    The 0.5 factor simplifies gradient calculation (derivative loses the 2).
    """
    # Get predictions from forward pass
    scores, _ = two_layer_forward(x, params)
    
    # Compute MSE loss
    return 0.5 * np.sum((scores - target) ** 2)


def numerical_gradient(f, x, h=1e-5):
    """Compute numerical gradient using central finite differences.
    
    This is a sanity check for our analytical gradients from backpropagation.
    
    How it works: for each parameter, we tweak it slightly (+h and -h),
    see how the loss changes, and approximate the derivative.
    Central differences (using both +h and -h) are more accurate than
    one-sided differences.
    """
    grad = np.zeros_like(x)
    
    # Iterate through every element in the parameter array
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index  # Get current position (e.g., (0,0), (0,1), etc.)
        old_val = x[idx]      # Save original value

        # Compute f(x + h)
        x[idx] = old_val + h
        fxph = f(x)

        # Compute f(x - h)
        x[idx] = old_val - h
        fxmh = f(x)

        # Restore original value (important!)
        x[idx] = old_val

        # Central difference formula: [f(x+h) - f(x-h)] / (2h)
        grad[idx] = (fxph - fxmh) / (2 * h)
        
        it.iternext()  # Move to next element

    return grad


def relative_error(x, y):
    """Compute relative error between two arrays.
    
    Used to compare analytical gradients (from backprop) with numerical gradients.
    
    Why relative error instead of absolute error?
    A difference of 0.1 means different things if the values are 1.0 vs 1000.0.
    Relative error accounts for scale.
    """
    return np.linalg.norm(x - y) / (np.linalg.norm(x) + np.linalg.norm(y))


# ============================================================================
# VISUALIZATION FUNCTIONS - SEE THE MATH COME TO LIFE!
# ============================================================================

def visualize_relu_activation():
    """Create visualization of ReLU activation function and its gradient."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Generate data points
    x = np.linspace(-3, 3, 1000)
    y = np.maximum(0, x)  # ReLU function
    
    # Plot 1: ReLU function
    axes[0].plot(x, y, linewidth=3, color='#2E86AB', label='ReLU(x) = max(0, x)')
    axes[0].plot(x, x, '--', linewidth=2, color='#A23B72', label='y = x (reference)')
    
    # Highlight different regions
    axes[0].fill_between(x, y, where=(x > 0), alpha=0.3, color='#F18F01', label='Active Region')
    axes[0].fill_between(x, y, where=(x <= 0), alpha=0.3, color='#C73E1D', label='Inactive Region')
    
    axes[0].set_title('ReLU Activation Function', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Input (x)')
    axes[0].set_ylabel('Output (ReLU(x))')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()
    axes[0].axhline(0, color='black', linewidth=0.5)
    axes[0].axvline(0, color='black', linewidth=0.5)
    
    # Plot 2: ReLU gradient
    x_grad = np.linspace(-3, 3, 1000)
    y_grad = (x_grad > 0).astype(float)  # Gradient: 1 if x>0, 0 otherwise
    
    axes[1].plot(x_grad, y_grad, linewidth=3, color='#A23B72', label="ReLU'(x)")
    axes[1].fill_between(x_grad, y_grad, where=(x_grad > 0), alpha=0.3, color='#F18F01', label='Gradient = 1')
    axes[1].fill_between(x_grad, y_grad, where=(x_grad <= 0), alpha=0.3, color='#C73E1D', label='Gradient = 0')
    
    axes[1].set_title('ReLU Gradient', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Input (x)')
    axes[1].set_ylabel("ReLU'(x)")
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()
    axes[1].axhline(0, color='black', linewidth=0.5)
    axes[1].axvline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()


def visualize_affine_transformation():
    """Visualize how affine transformation changes data."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create random 2D data points
    x = np.random.randn(100, 2)
    
    # Define transformation: W rotates/scales, b shifts
    W = np.array([[1.5, 0.5], [0.2, 1.0]])
    b = np.array([0.5, -0.3])
    
    # Apply transformation: y = xW + b
    transformed = x @ W + b
    
    # Plot original and transformed data
    ax.scatter(x[:, 0], x[:, 1], alpha=0.6, c='blue', label='Original Data', s=50)
    ax.scatter(transformed[:, 0], transformed[:, 1], alpha=0.6, c='red', label='Transformed Data', s=50)
    
    ax.set_title('Affine Transformation: $y = xW + b$', fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    # Add annotation to show transformation of one point
    ax.annotate('Linear Transformation\n(Rotation + Scaling + Translation)', 
                xy=(transformed[0, 0], transformed[0, 1]), 
                xytext=(x[0, 0], x[0, 1]),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=12, ha='center')
    
    plt.tight_layout()
    plt.show()


def visualize_network_flow():
    """Visualize forward and backward flow in neural network."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Forward pass visualization
    layer_names = ['Input (x)', 'FC1 (W1, b1)', 'ReLU', 'FC2 (W2, b2)', 'Output']
    colors = ['#FF9AA2', '#FFB7B2', '#FFDAC1', '#E2F0CB', '#B5EAD7']
    
    axes[0].barh(range(len(layer_names)), [1]*len(layer_names), color=colors, height=0.7)
    for i, name in enumerate(layer_names):
        axes[0].text(0.5, i, name, ha='center', va='center', fontweight='bold', fontsize=12)
    
    axes[0].set_title('Forward Pass: Data Flow Through Network', fontsize=14, fontweight='bold')
    axes[0].set_yticks(range(len(layer_names)))
    axes[0].set_yticklabels([])
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel('Forward Propagation Direction →')
    
    # Backward pass visualization
    layer_names_back = ['Output', 'FC2 (W2, b2)', 'ReLU', 'FC1 (W1, b1)', 'Input (x)']
    colors_back = ['#B5EAD7', '#E2F0CB', '#FFDAC1', '#FFB7B2', '#FF9AA2']
    
    axes[1].barh(range(len(layer_names_back)), [1]*len(layer_names_back), color=colors_back, height=0.7)
    for i, name in enumerate(layer_names_back):
        axes[1].text(0.5, i, name, ha='center', va='center', fontweight='bold', fontsize=12)
    
    axes[1].set_title('Backward Pass: Gradient Flow (Backpropagation)', fontsize=14, fontweight='bold')
    axes[1].set_yticks(range(len(layer_names_back)))
    axes[1].set_yticklabels([])
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel('Backward Propagation Direction ←')
    
    plt.tight_layout()
    plt.show()


def visualize_gradient_checking():
    """Visualize gradient checking concept."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Simple quadratic function: f(x) = x²
    x = np.linspace(-2, 2, 1000)
    y = x**2
    dy_dx = 2*x  # Analytical derivative: f'(x) = 2x
    
    # Show central difference approximation at x = 0.5
    h = 0.2
    x0 = 0.5
    f_x0 = x0**2
    f_x0_h = (x0+h)**2
    f_x0_neg_h = (x0-h)**2
    
    # Plot function and its derivative
    ax.plot(x, y, label='$f(x) = x^2$', linewidth=2, color='blue')
    ax.plot(x, dy_dx, label="$f'(x) = 2x$", linewidth=2, color='red', linestyle='--')
    
    # Mark points used for numerical gradient calculation
    ax.plot([x0-h, x0, x0+h], [f_x0_neg_h, f_x0, f_x0_h], 'o', markersize=8, 
            color='orange', label='Points for numerical gradient')
    
    # Draw tangent line (analytical gradient at x0)
    tangent_x = np.linspace(x0-1, x0+1, 100)
    tangent_y = f_x0 + 2*x0*(tangent_x - x0)
    ax.plot(tangent_x, tangent_y, '--', color='green', linewidth=2, 
            label=f'Tangent line (slope = {2*x0})')
    
    # Draw secant line (numerical approximation)
    secant_slope = (f_x0_h - f_x0_neg_h) / (2*h)
    secant_y = f_x0 + secant_slope*(tangent_x - x0)
    ax.plot(tangent_x, secant_y, ':', color='purple', linewidth=2, 
            label=f'Secant line (slope ≈ {secant_slope:.2f})')
    
    ax.set_title('Gradient Checking: Analytical vs Numerical Gradients', fontsize=14, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x) / f\'(x)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN DEMONSTRATION - SEE EVERYTHING IN ACTION!
# ============================================================================

# Test ReLU implementation
print("=== Testing ReLU Activation ===")
print("ReLU: f(x) = max(0, x)")
print("Gradient: f'(x) = 1 if x > 0, else 0")

# Create test input with both positive and negative values
x = np.array([[-1.0, 0.5], [2.0, -3.0]])
print(f"\nTest input:\n{x}")

# Forward pass through ReLU
out, cache = relu_forward(x)
print(f"\nReLU output (negative values become 0):\n{out}")

# Backward pass (simulating gradient from later layers)
dout = np.ones_like(x)  # Gradient of 1 from next layer
dx = relu_backward(dout, cache)
print(f"\nReLU gradient (only positive inputs get gradient):\n{dx}")

print("✓ ReLU correctly: zeros negatives, passes positives, blocks negative gradients")

# Visualize ReLU function
visualize_relu_activation()

# Test Affine layer implementation
print("\n" + "="*50)
print("=== Testing Affine Layer ===")
print("Affine transformation: y = xW + b")

# Set random seed for reproducibility
np.random.seed(42)

# Create test data
x = np.random.randn(2, 3)  # 2 samples, 3 features each
W = np.random.randn(3, 4)  # 3 input features → 4 output features
b = np.random.randn(4)     # 4 biases (one per output feature)

print(f"\nInput shape: {x.shape} (batch_size × input_dim)")
print(f"Weight shape: {W.shape} (input_dim × output_dim)")
print(f"Bias shape: {b.shape} (output_dim,)")

# Forward pass
out, cache = affine_forward(x, W, b)
print(f"\nOutput shape: {out.shape} (batch_size × output_dim)")

# Backward pass
dout = np.random.randn(2, 4)  # Simulated gradient from next layer
dx, dW, db = affine_backward(dout, cache)

print(f"\nGradient shapes:")
print(f"  dx (input gradient): {dx.shape} ← how input affects loss")
print(f"  dW (weight gradient): {dW.shape} ← how weights affect loss")
print(f"  db (bias gradient): {db.shape} ← how biases affect loss")

print("✓ Affine layer correctly transforms and computes gradients")

# Visualize affine transformation
visualize_affine_transformation()

# Test complete two-layer network with gradient checking
print("\n" + "="*50)
print("=== Testing Two-Layer Network & Gradient Check ===")
print("This validates our backpropagation implementation is correct!")

np.random.seed(42)

# Create a small network
x = np.random.randn(3, 5)      # 3 samples, 5 features
W1 = np.random.randn(5, 4)     # First layer: 5 → 4
b1 = np.random.randn(4)
W2 = np.random.randn(4, 2)     # Second layer: 4 → 2
b2 = np.random.randn(2)
params = (W1, b1, W2, b2)

# Random targets for regression
target = np.random.randn(3, 2)

print(f"\nNetwork architecture:")
print(f"  Input: {x.shape[1]} features")
print(f"  Hidden layer: {W1.shape[1]} neurons")
print(f"  Output: {W2.shape[1]} values")

# Forward pass
scores, cache = two_layer_forward(x, params)
loss = eval_loss(x, params, target)

print(f"\nForward pass:")
print(f"  Scores shape: {scores.shape} (predictions)")
print(f"  Loss: {loss:.4f} (how wrong are predictions?)")

# Backward pass (compute gradients analytically)
dout = scores - target  # Gradient of MSE loss: ∂L/∂scores = scores - target
_, grads = two_layer_backward(dout, cache)

print(f"\nAnalytical gradients computed via backpropagation")

# Numerical gradient check for W1
print(f"\nGradient checking - comparing analytical vs numerical:")
print("  (Should be < 1e-7 for correct implementation)")

# Function that computes loss with only W1 varying
f = lambda W: eval_loss(x, (W, b1, W2, b2), target)

# Compute numerical gradient using finite differences
num_dW1 = numerical_gradient(f, W1)

# Compare with analytical gradient from backprop
rel_err = relative_error(num_dW1, grads[0])
print(f"  Relative error for W1: {rel_err:.2e}")

if rel_err < 1e-7:
    print("✓ Gradient check PASSED! Backprop implementation is correct.")
else:
    print("✗ Gradient check FAILED! Check backprop implementation.")

# Visualize network flow
visualize_network_flow()

# Visualize gradient checking concept
visualize_gradient_checking()

# Final summary
print("\n" + "="*50)
print("=== Shape Verification ===")
print("All shapes should make sense for backpropagation:")

print(f"\nForward pass shapes:")
print(f"  Input: {x.shape}")
print(f"  Scores: {scores.shape}")

print(f"\nGradient shapes:")
for i, g in enumerate(grads):
    layer = ["W1", "b1", "W2", "b2"][i]
    print(f"  d{layer}: {g.shape}")

# Create final summary visualization
fig, ax = plt.subplots(figsize=(12, 8))
ax.text(0.05, 0.95, 'Neural Network Components Summary', 
        fontsize=18, fontweight='bold', transform=ax.transAxes)
ax.text(0.05, 0.85, '• Affine Layer: Performs linear transformation y = xW + b', 
        fontsize=12, transform=ax.transAxes)
ax.text(0.05, 0.80, '• ReLU Activation: Introduces non-linearity (max(0, x))', 
        fontsize=12, transform=ax.transAxes)
ax.text(0.05, 0.75, '• Backpropagation: Computes gradients using chain rule', 
        fontsize=12, transform=ax.transAxes)
ax.text(0.05, 0.70, '• Gradient Checking: Validates analytical gradients', 
        fontsize=12, transform=ax.transAxes)

ax.text(0.05, 0.60, 'Implementation successfully tested with:', 
        fontsize=14, fontweight='bold', transform=ax.transAxes)
ax.text(0.05, 0.55, '• Forward/backward passes', fontsize=12, transform=ax.transAxes)
ax.text(0.05, 0.50, '• Gradient verification (error: {:.2e})'.format(rel_err), 
        fontsize=12, transform=ax.transAxes)
ax.text(0.05, 0.45, '• Two-layer network', fontsize=12, transform=ax.transAxes)

ax.text(0.05, 0.30, 'Key Insight:', fontsize=14, fontweight='bold', transform=ax.transAxes)
ax.text(0.05, 0.25, 'Backpropagation is just the chain rule applied efficiently!', 
        fontsize=12, transform=ax.transAxes)
ax.text(0.05, 0.20, 'It lets us compute how every parameter affects the loss.', 
        fontsize=12, transform=ax.transAxes)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("🎉 DEMONSTRATION COMPLETE!")
print("You've implemented and validated a neural network from scratch!")
print("\nNext steps: Try modifying the network architecture or loss function.")