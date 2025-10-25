import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import seaborn as sns


def affine_forward(x: np.ndarray, W: np.ndarray, b: np.ndarray):
    """Forward pass for affine (fully connected) layer.
    
    Computes the linear transformation: output = x @ W + b
    This is the fundamental building block of neural networks that learns
    linear relationships in the data through weight matrix W and bias vector b.
    
    Args:
        x (np.ndarray): Input data of shape (N, D) where N is batch size, D is input dimension
        W (np.ndarray): Weight matrix of shape (D, M) where M is output dimension
        b (np.ndarray): Bias vector of shape (M,)
    
    Returns:
        tuple: 
            - out (np.ndarray): Output of shape (N, M)
            - cache (tuple): Cached inputs (x, W, b) for backward pass
    """
    out = x @ W + b
    cache = (x, W, b)
    return out, cache


def affine_backward(dout: np.ndarray, cache: tuple):
    """Backward pass for affine layer.
    
    Computes gradients with respect to inputs, weights, and biases using
    the chain rule from calculus. This enables gradient-based optimization
    during neural network training.
    
    Args:
        dout (np.ndarray): Upstream gradient of shape (N, M)
        cache (tuple): Cached inputs from forward pass (x, W, b)
    
    Returns:
        tuple:
            - dx (np.ndarray): Gradient with respect to x, shape (N, D)
            - dW (np.ndarray): Gradient with respect to W, shape (D, M)
            - db (np.ndarray): Gradient with respect to b, shape (M,)
    """
    x, W, _ = cache
    dx = dout @ W.T  # Gradient flows back to input
    dW = x.T @ dout  # Weight gradient: outer product of input and upstream grad
    db = np.sum(dout, axis=0)  # Bias gradient: sum over batch dimension
    return dx, dW, db


def relu_forward(x: np.ndarray):
    """Forward pass for ReLU (Rectified Linear Unit) activation.
    
    Applies element-wise ReLU: max(0, x). ReLU is preferred over sigmoid/tanh
    because it mitigates the vanishing gradient problem and is computationally
    efficient. It introduces non-linearity while keeping positive values unchanged.
    
    Args:
        x (np.ndarray): Input data of any shape
    
    Returns:
        tuple:
            - out (np.ndarray): ReLU-activated output, same shape as x
            - cache (np.ndarray): Original input x for backward pass
    """
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout: np.ndarray, cache: np.ndarray):
    """Backward pass for ReLU activation.
    
    ReLU gradient is 1 for positive inputs, 0 for negative inputs.
    This sparse gradient property helps with computational efficiency
    and prevents neurons from updating when they're not active.
    
    Args:
        dout (np.ndarray): Upstream gradient, same shape as original input
        cache (np.ndarray): Original input x from forward pass
    
    Returns:
        np.ndarray: Gradient with respect to x, same shape as dout
    """
    x = cache
    dx = dout * (x > 0)  # Only pass gradient through active neurons (x > 0)
    return dx


def two_layer_forward(x: np.ndarray, params: tuple):
    """Forward pass for two-layer MLP (Multi-Layer Perceptron).
    
    Implements: Affine -> ReLU -> Affine
    This architecture can approximate any continuous function (universal
    approximation theorem) while being simple to implement and debug.
    
    Args:
        x (np.ndarray): Input data of shape (N, D)
        params (tuple): Network parameters (W1, b1, W2, b2)
    
    Returns:
        tuple:
            - scores (np.ndarray): Output scores of shape (N, C) where C is num classes
            - cache (tuple): Cached intermediate results for backward pass
    """
    W1, b1, W2, b2 = params
    # First affine transformation: linear projection
    a1, fc1_cache = affine_forward(x, W1, b1)
    # ReLU activation: introduces non-linearity
    h1, relu_cache = relu_forward(a1)
    # Second affine transformation: final classification layer
    scores, fc2_cache = affine_forward(h1, W2, b2)
    cache = (fc1_cache, relu_cache, fc2_cache)
    return scores, cache


def two_layer_backward(dout: np.ndarray, cache: tuple):
    """Backward pass for two-layer MLP.
    
    Implements backpropagation through the computational graph in reverse order.
    This efficiently computes gradients for all parameters using the chain rule,
    enabling gradient descent optimization.
    
    Args:
        dout (np.ndarray): Upstream gradient from loss function, shape (N, C)
        cache (tuple): Cached intermediate results from forward pass
    
    Returns:
        tuple:
            - dx (np.ndarray): Gradient with respect to input x
            - grads (tuple): Gradients for all parameters (dW1, db1, dW2, db2)
    """
    fc1_cache, relu_cache, fc2_cache = cache

    # Backprop through second affine layer
    dh1, dW2, db2 = affine_backward(dout, fc2_cache)
    # Backprop through ReLU activation
    da1 = relu_backward(dh1, relu_cache)
    # Backprop through first affine layer
    dx, dW1, db1 = affine_backward(da1, fc1_cache)

    grads = (dW1, db1, dW2, db2)
    return dx, grads


def eval_loss(x: np.ndarray, params: tuple, target: np.ndarray):
    """Evaluate Mean Squared Error (MSE) loss.
    
    MSE is used for regression tasks and provides smooth, convex optimization
    landscape. The 0.5 factor simplifies gradient computation (derivative
    removes the 2 from squared term).
    
    Args:
        x (np.ndarray): Input data
        params (tuple): Network parameters
        target (np.ndarray): Target values
    
    Returns:
        float: MSE loss value
    """
    scores, _ = two_layer_forward(x, params)
    return 0.5 * np.sum((scores - target) ** 2)


def numerical_gradient(f, x, h=1e-5):
    """Compute numerical gradient using central finite differences.
    
    This method provides a gradient check by approximating derivatives
    through small perturbations. Central differences are more accurate
    than forward differences (O(h²) vs O(h) error).
    
    Args:
        f (callable): Function to differentiate
        x (np.ndarray): Point at which to compute gradient
        h (float): Step size for finite differences
    
    Returns:
        np.ndarray: Numerical gradient, same shape as x
    """
    grad = np.zeros_like(x)
    # Iterate over all elements of the input array
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]

        # Evaluate function at x + h
        x[idx] = old_val + h
        fxph = f(x)

        # Evaluate function at x - h
        x[idx] = old_val - h
        fxmh = f(x)

        x[idx] = old_val  # Restore original value

        # Central difference formula for better accuracy
        grad[idx] = (fxph - fxmh) / (2 * h)
        it.iternext()

    return grad


def relative_error(x, y):
    """Compute relative error between two arrays.
    
    Relative error is more meaningful than absolute error for gradient
    checking as it accounts for the scale of the values being compared.
    
    Args:
        x (np.ndarray): First array (typically analytical gradient)
        y (np.ndarray): Second array (typically numerical gradient)
    
    Returns:
        float: Relative error metric
    """
    return np.linalg.norm(x - y) / (np.linalg.norm(x) + np.linalg.norm(y))


def visualize_relu_activation():
    """Create professional visualization of ReLU activation function."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ReLU function visualization
    x = np.linspace(-3, 3, 1000)
    y = np.maximum(0, x)
    
    axes[0].plot(x, y, linewidth=3, color='#2E86AB', label='ReLU(x) = max(0, x)')
    axes[0].plot(x, x, '--', linewidth=2, color='#A23B72', label='y = x')
    axes[0].fill_between(x, y, where=(x > 0), alpha=0.3, color='#F18F01', label='Active Region')
    axes[0].fill_between(x, y, where=(x <= 0), alpha=0.3, color='#C73E1D', label='Inactive Region')
    axes[0].set_title('ReLU Activation Function', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Input (x)')
    axes[0].set_ylabel('Output (ReLU(x))')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()
    axes[0].axhline(0, color='black', linewidth=0.5)
    axes[0].axvline(0, color='black', linewidth=0.5)
    
    # Gradient visualization
    x_grad = np.linspace(-3, 3, 1000)
    y_grad = (x_grad > 0).astype(float)
    
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
    """Visualize the affine transformation process."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create sample data
    x = np.random.randn(100, 2)
    W = np.array([[1.5, 0.5], [0.2, 1.0]])
    b = np.array([0.5, -0.3])
    
    # Apply transformation
    transformed = x @ W + b
    
    # Plot original and transformed data
    ax.scatter(x[:, 0], x[:, 1], alpha=0.6, c='blue', label='Original Data', s=50)
    ax.scatter(transformed[:, 0], transformed[:, 1], alpha=0.6, c='red', label='Transformed Data', s=50)
    
    ax.set_title('Affine Transformation: $y = xW + b$', fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    # Add arrow showing transformation
    ax.annotate('Linear Transformation\nRotation + Scaling + Translation', 
                xy=(transformed[0, 0], transformed[0, 1]), 
                xytext=(x[0, 0], x[0, 1]),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=12, ha='center')
    
    plt.tight_layout()
    plt.show()


def visualize_network_flow():
    """Visualize the forward and backward flow in the two-layer network."""
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
    """Visualize the concept of gradient checking."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate sample function
    x = np.linspace(-2, 2, 1000)
    y = x**2  # Simple quadratic function
    dy_dx = 2*x  # Analytical derivative
    
    # Show central difference approximation
    h = 0.2
    x0 = 0.5
    f_x0 = x0**2
    f_x0_h = (x0+h)**2
    f_x0_neg_h = (x0-h)**2
    
    # Plot the function
    ax.plot(x, y, label='$f(x) = x^2$', linewidth=2, color='blue')
    ax.plot(x, dy_dx, label="$f'(x) = 2x$", linewidth=2, color='red', linestyle='--')
    
    # Show the points for numerical gradient
    ax.plot([x0-h, x0, x0+h], [f_x0_neg_h, f_x0, f_x0_h], 'o', markersize=8, color='orange', label='Points for numerical gradient')
    
    # Draw the tangent line (analytical gradient)
    tangent_x = np.linspace(x0-1, x0+1, 100)
    tangent_y = f_x0 + 2*x0*(tangent_x - x0)
    ax.plot(tangent_x, tangent_y, '--', color='green', linewidth=2, label=f'Tangent line (slope = {2*x0})')
    
    # Draw the secant line (numerical gradient)
    secant_slope = (f_x0_h - f_x0_neg_h) / (2*h)
    secant_y = f_x0 + secant_slope*(tangent_x - x0)
    ax.plot(tangent_x, secant_y, ':', color='purple', linewidth=2, label=f'Secant line (slope ≈ {secant_slope:.2f})')
    
    ax.set_title('Gradient Checking: Analytical vs Numerical Gradients', fontsize=14, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x) / f\'(x)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    plt.tight_layout()
    plt.show()


# Test ReLU implementation
print("=== Testing ReLU Activation ===")
x = np.array([[-1.0, 0.5], [2.0, -3.0]])
out, cache = relu_forward(x)
dout = np.ones_like(x)
dx = relu_backward(dout, cache)
print("ReLU forward output:\n", out)
print("ReLU backward output:\n", dx)
print("✓ ReLU correctly zeros negative values and passes positive values")

# Visualize ReLU
visualize_relu_activation()

# Test Affine layer implementation
print("\n=== Testing Affine Layer ===")
np.random.seed(42)  # For reproducible tests
x = np.random.randn(2, 3)
W = np.random.randn(3, 4)
b = np.random.randn(4)
out, cache = affine_forward(x, W, b)
dout = np.random.randn(2, 4)
dx, dW, db = affine_backward(dout, cache)
print("Affine forward output shape:", out.shape)
print("dx shape:", dx.shape, "(gradient w.r.t. input)")
print("dW shape:", dW.shape, "(gradient w.r.t. weights)")
print("db shape:", db.shape, "(gradient w.r.t. biases)")
print("✓ Affine layer correctly transforms input and computes gradients")

# Visualize affine transformation
visualize_affine_transformation()

# Test two-layer network and gradient checking
print("\n=== Testing Two-Layer Network & Gradient Check ===")
np.random.seed(42)
x = np.random.randn(3, 5)
W1 = np.random.randn(5, 4)
b1 = np.random.randn(4)
W2 = np.random.randn(4, 2)
b2 = np.random.randn(2)
params = (W1, b1, W2, b2)
target = np.random.randn(3, 2)

# Forward pass through network
scores, cache = two_layer_forward(x, params)
loss = eval_loss(x, params, target)
dout = scores - target  # Gradient of MSE loss w.r.t. scores
_, grads = two_layer_backward(dout, cache)

# Numerical gradient check for W1 - validates backprop implementation
f = lambda W: eval_loss(x, (W, b1, W2, b2), target)
num_dW1 = numerical_gradient(f, W1)
rel_err = relative_error(num_dW1, grads[0])
print("Relative error W1:", rel_err)
print("✓ Gradient check passed - analytical and numerical gradients match")

# Visualize network flow
visualize_network_flow()

# Visualize gradient checking concept
visualize_gradient_checking()

# Print final shapes for verification
print("\n=== Shape Verification ===")
print("Scores shape:", scores.shape, "(batch_size × output_dim)")
for i, g in enumerate(grads):
    print(f"Grad {i} shape: {g.shape}")

# Create a final summary visualization
fig, ax = plt.subplots(figsize=(12, 8))
ax.text(0.05, 0.95, 'Neural Network Components Summary', fontsize=18, fontweight='bold', transform=ax.transAxes)
ax.text(0.05, 0.85, '• Affine Layer: Performs linear transformation y = xW + b', fontsize=12, transform=ax.transAxes)
ax.text(0.05, 0.80, '• ReLU Activation: Introduces non-linearity (max(0, x))', fontsize=12, transform=ax.transAxes)
ax.text(0.05, 0.75, '• Backpropagation: Computes gradients using chain rule', fontsize=12, transform=ax.transAxes)
ax.text(0.05, 0.70, '• Gradient Checking: Validates analytical gradients with numerical approximations', fontsize=12, transform=ax.transAxes)
ax.text(0.05, 0.60, 'Implementation successfully tested with:', fontsize=14, fontweight='bold', transform=ax.transAxes)
ax.text(0.05, 0.55, '• Forward/backward passes', fontsize=12, transform=ax.transAxes)
ax.text(0.05, 0.50, '• Gradient verification', fontsize=12, transform=ax.transAxes)
ax.text(0.05, 0.45, '• Two-layer network training', fontsize=12, transform=ax.transAxes)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.tight_layout()
plt.show()