import numpy as np


def affine_forward(x: np.ndarray, W: np.ndarray, b: np.ndarray):
    """Forward pass for affine (fully connected) layer."""
    out = x @ W + b
    cache = (x, W, b)
    return out, cache


def affine_backward(dout: np.ndarray, cache: tuple):
    """Backward pass for affine layer."""
    x, W, _ = cache
    dx = dout @ W.T
    dW = x.T @ dout
    db = np.sum(dout, axis=0)
    return dx, dW, db


def relu_forward(x: np.ndarray):
    """Forward pass for ReLU activation."""
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout: np.ndarray, cache: np.ndarray):
    """Backward pass for ReLU activation."""
    x = cache
    dx = dout * (x > 0)
    return dx


def two_layer_forward(x: np.ndarray, params: tuple):
    """Forward pass for two-layer MLP."""
    W1, b1, W2, b2 = params
    a1, fc1_cache = affine_forward(x, W1, b1)
    h1, relu_cache = relu_forward(a1)
    scores, fc2_cache = affine_forward(h1, W2, b2)
    cache = (fc1_cache, relu_cache, fc2_cache)
    return scores, cache


def two_layer_backward(dout: np.ndarray, cache: tuple):
    """Backward pass for two-layer MLP."""
    fc1_cache, relu_cache, fc2_cache = cache

    dh1, dW2, db2 = affine_backward(dout, fc2_cache)
    da1 = relu_backward(dh1, relu_cache)
    dx, dW1, db1 = affine_backward(da1, fc1_cache)

    grads = (dW1, db1, dW2, db2)
    return dx, grads


def eval_loss(x: np.ndarray, params: tuple, target: np.ndarray):
    """Evaluate loss (MSE)."""
    scores, _ = two_layer_forward(x, params)
    return 0.5 * np.sum((scores - target) ** 2)


def numerical_gradient(f, x, h=1e-5):
    """Compute numerical gradient using finite differences."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]

        x[idx] = old_val + h
        fxph = f(x)

        x[idx] = old_val - h
        fxmh = f(x)

        x[idx] = old_val  # restore

        grad[idx] = (fxph - fxmh) / (2 * h)
        it.iternext()

    return grad


def relative_error(x, y):
    """Compute relative error between two arrays."""
    return np.linalg.norm(x - y) / (np.linalg.norm(x) + np.linalg.norm(y))


# Test ReLU
x = np.array([[-1.0, 0.5], [2.0, -3.0]])
out, cache = relu_forward(x)
dout = np.ones_like(x)
dx = relu_backward(dout, cache)
print("ReLU backward output:\n", dx)
print("ReLU forward output:\n", out)

# Test Affine
np.random.seed(42)
x = np.random.randn(2, 3)
W = np.random.randn(3, 4)
b = np.random.randn(4)
out, cache = affine_forward(x, W, b)
dout = np.random.randn(2, 4)
dx, dW, db = affine_backward(dout, cache)
print("Affine forward output:\n", out)
print("dx shape:", dx.shape)
print("dW shape:", dW.shape)
print("db shape:", db.shape)

# Test two-layer network and gradient check
np.random.seed(42)
x = np.random.randn(3, 5)
W1 = np.random.randn(5, 4)
b1 = np.random.randn(4)
W2 = np.random.randn(4, 2)
b2 = np.random.randn(2)
params = (W1, b1, W2, b2)
target = np.random.randn(3, 2)

scores, cache = two_layer_forward(x, params)
loss = eval_loss(x, params, target)
dout = scores - target
_, grads = two_layer_backward(dout, cache)

# Numerical gradient check for W1
f = lambda W: eval_loss(x, (W, b1, W2, b2), target)
num_dW1 = numerical_gradient(f, W1)
rel_err = relative_error(num_dW1, grads[0])
print("Relative error W1:", rel_err)

# Print shapes
print("Scores shape:", scores.shape)
for g in grads:
    print("Grad shape:", g.shape)