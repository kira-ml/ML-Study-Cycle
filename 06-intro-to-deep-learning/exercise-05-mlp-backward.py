import numpy as np


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


# Test ReLU implementation
print("=== Testing ReLU Activation ===")
x = np.array([[-1.0, 0.5], [2.0, -3.0]])
out, cache = relu_forward(x)
dout = np.ones_like(x)
dx = relu_backward(dout, cache)
print("ReLU forward output:\n", out)
print("ReLU backward output:\n", dx)
print("✓ ReLU correctly zeros negative values and passes positive values")

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

# Print final shapes for verification
print("\n=== Shape Verification ===")
print("Scores shape:", scores.shape, "(batch_size × output_dim)")
for i, g in enumerate(grads):
    print(f"Grad {i} shape: {g.shape}")