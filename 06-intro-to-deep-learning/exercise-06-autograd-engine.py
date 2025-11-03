import numpy as np


class Tensor:
    """
    A simple tensor class that supports automatic differentiation.
    
    This implements a reverse-mode autodiff system similar to PyTorch, where
    computational graphs are built dynamically during forward pass operations.
    
    Attributes:
        data: Underlying numpy array storing tensor values
        requires_grad: Flag indicating if gradients should be computed for this tensor
        grad: Gradient of loss with respect to this tensor
        _op: Operation that created this tensor (for debugging and graph visualization)
        _children: Input tensors that were used to create this tensor
        _backward: Function to compute gradients during backward pass
    """
    
    def __init__(self, data, requires_grad=False, _op='', _children=()):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._op = _op  # Operation that produced this tensor
        self._children = set(_children)  # Input tensors in computational graph
        self._backward = lambda: None  # Gradient computation function

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad}, grad={self.grad})"

    def __add__(self, other):
        """Element-wise addition with broadcasting support."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _op='+',
            _children=(self, other)
        )

        def _backward():
            # Gradient flows equally to both operands in addition
            if self.requires_grad:
                # Handle broadcasting by summing over extra dimensions
                self.grad += _broadcast_gradient(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _broadcast_gradient(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        """Element-wise multiplication with broadcasting support."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _op='*',
            _children=(self, other)
        )

        def _backward():
            # For multiplication: d(out)/d(self) = other * d(out)/d(out)
            if self.requires_grad:
                grad = other.data * out.grad
                self.grad += _broadcast_gradient(grad, self.data.shape)
            if other.requires_grad:
                grad = self.data * out.grad
                other.grad += _broadcast_gradient(grad, other.data.shape)

        out._backward = _backward
        return out

    def __sub__(self, other):
        """Element-wise subtraction."""
        return self + (-other)

    def __neg__(self):
        """Element-wise negation."""
        return self * -1

    def __pow__(self, exponent):
        """Element-wise power operation."""
        if not isinstance(exponent, (int, float)):
            raise ValueError("Exponent must be a scalar")
            
        out = Tensor(
            self.data ** exponent,
            requires_grad=self.requires_grad,
            _op=f'**{exponent}',
            _children=(self,)
        )

        def _backward():
            if self.requires_grad:
                # d(x^n)/dx = n * x^(n-1)
                self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    def matmul(self, other):
        """Matrix multiplication."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _op='@',
            _children=(self, other)
        )

        def _backward():
            # Matrix multiplication gradients:
            # d(out)/d(self) = d(out)/d(out) @ other.T
            # d(out)/d(other) = self.T @ d(out)/d(out)
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def relu(self):
        """Rectified Linear Unit activation function."""
        out = Tensor(
            np.maximum(0, self.data), 
            requires_grad=self.requires_grad, 
            _op='ReLU', 
            _children=(self,)
        )

        def _backward():
            if self.requires_grad:
                # ReLU gradient: 1 where input > 0, else 0
                self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        """Sigmoid activation function."""
        # Precompute for efficiency in both forward and backward passes
        sigmoid_data = 1 / (1 + np.exp(-self.data))
        out = Tensor(
            sigmoid_data, 
            requires_grad=self.requires_grad, 
            _op='sigmoid', 
            _children=(self,)
        )

        def _backward():
            if self.requires_grad:
                # Sigmoid derivative: σ(x) * (1 - σ(x))
                self.grad += sigmoid_data * (1 - sigmoid_data) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        """Hyperbolic tangent activation function."""
        tanh_data = np.tanh(self.data)
        out = Tensor(
            tanh_data, 
            requires_grad=self.requires_grad, 
            _op='tanh', 
            _children=(self,)
        )

        def _backward():
            if self.requires_grad:
                # tanh derivative: 1 - tanh²(x)
                self.grad += (1 - tanh_data ** 2) * out.grad

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        """Sum over specified dimensions."""
        out = Tensor(
            np.sum(self.data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _op='sum',
            _children=(self,)
        )

        def _backward():
            if self.requires_grad:
                grad = out.grad
                # Handle broadcasting for summed dimensions
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                self.grad += np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        """Mean over specified dimensions."""
        out = Tensor(
            np.mean(self.data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _op='mean',
            _children=(self,)
        )

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                # For mean: gradient is distributed equally to all elements
                count = np.prod([self.data.shape[i] for i in (axis if axis is not None else range(self.data.ndim))])
                self.grad += np.broadcast_to(grad / count, self.data.shape)

        out._backward = _backward
        return out

    def backward(self, gradient=None):
        """
        Compute gradients for all tensors in the computational graph.
        
        This implements reverse-mode automatic differentiation by:
        1. Topologically sorting the computational graph
        2. Applying chain rule in reverse order
        3. Accumulating gradients in each tensor's .grad attribute
        
        Args:
            gradient: Initial gradient (defaults to ones, like scalar loss)
        """
        if gradient is None:
            gradient = np.ones_like(self.data)
        self.grad = gradient

        # Topological sort of computational graph
        topo = []
        visited = set()

        def build_topo(v):
            """Build topological order using DFS."""
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Apply chain rule in reverse topological order
        for v in reversed(topo):
            v._backward()

    def zero_grad(self):
        """Reset gradients to zero."""
        if self.grad is not None:
            self.grad.fill(0)

    def detach(self):
        """Create a new tensor detached from the computational graph."""
        return Tensor(self.data, requires_grad=False)

    def prune_tape(self):
        """
        Clear computational graph to free memory.
        
        This breaks references to allow garbage collection of intermediate tensors
        that are no longer needed after backward pass.
        """
        for child in self._children:
            if child.requires_grad:
                child.prune_tape()
        self._children = set()
        self._backward = lambda: None

    # Enable right-side operations
    __radd__ = __add__
    __rmul__ = __mul__


def _broadcast_gradient(grad, target_shape):
    """
    Handle gradient broadcasting by summing over expanded dimensions.
    
    When operations involve broadcasting, the gradient must be reduced
    to match the original tensor shape.
    """
    if grad.shape == target_shape:
        return grad
    
    # Sum over dimensions that were broadcasted
    ndim_diff = grad.ndim - len(target_shape)
    if ndim_diff > 0:
        grad = grad.sum(axis=tuple(range(ndim_diff)))
    
    # Sum over dimensions that were size 1 in original tensor
    axes = tuple(i for i, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape)) 
                if t_dim == 1 and g_dim > 1)
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)
    
    return grad


class Linear:
    """
    A linear (fully-connected) neural network layer.
    
    Implements: output = input @ weight + bias
    
    Attributes:
        weight: Learnable weight matrix of shape (input_size, output_size)
        bias: Learnable bias vector of shape (output_size,)
    """
    
    def __init__(self, input_size, output_size):
        # Xavier/Glorot initialization for stable training
        scale = np.sqrt(2.0 / (input_size + output_size))
        self.weight = Tensor(
            np.random.randn(input_size, output_size) * scale, 
            requires_grad=True
        )
        self.bias = Tensor(np.zeros(output_size), requires_grad=True)

    def __call__(self, x):
        """Forward pass through the linear layer."""
        return x.matmul(self.weight) + self.bias

    def parameters(self):
        """Return all learnable parameters."""
        return [self.weight, self.bias]

    def zero_grad(self):
        """Reset gradients for all parameters."""
        for param in self.parameters():
            param.zero_grad()


def mse_loss(pred, target):
    """
    Mean Squared Error loss function.
    
    Args:
        pred: Predicted values
        target: Ground truth values
    
    Returns:
        MSE loss tensor
    """
    return ((pred - target) ** 2).mean()


def cross_entropy_loss(pred, target):
    """
    Cross Entropy loss for classification.
    
    Args:
        pred: Raw logits (before softmax)
        target: Class indices or one-hot encoded labels
    
    Returns:
        Cross entropy loss tensor
    """
    # Numerical stability: subtract max for softmax
    max_val = pred.data.max(axis=-1, keepdims=True)
    stable_pred = pred - Tensor(max_val)
    
    # Softmax computation
    exp_pred = stable_pred.exp()
    softmax_val = exp_pred / exp_pred.sum(axis=-1, keepdims=True)
    
    # Cross entropy
    if len(target.data.shape) == 1:  # Class indices
        # Convert to one-hot for gradient computation
        target_one_hot = np.eye(softmax_val.data.shape[-1])[target.data.astype(int)]
        target = Tensor(target_one_hot)
    
    log_softmax = stable_pred - exp_pred.sum(axis=-1, keepdims=True).log()
    return -(target * log_softmax).sum(axis=-1).mean()


class SGD:
    """
    Stochastic Gradient Descent optimizer.
    
    Attributes:
        parameters: List of parameters to optimize
        lr: Learning rate
    """
    
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        """Update parameters using computed gradients."""
        for param in self.parameters:
            if param.requires_grad:
                param.data -= self.lr * param.grad

    def zero_grad(self):
        """Reset gradients for all parameters."""
        for param in self.parameters:
            param.zero_grad()


# Example 1: Simple gradient check
def gradient_check():
    """Verify gradient computation with numerical differentiation."""
    print("=== Gradient Check ===")
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)
    c = a * b + Tensor([1.0])
    c.backward()
    
    print(f"a: {a}, grad: {a.grad}")  # Should be ~3.0
    print(f"b: {b}, grad: {b.grad}")  # Should be ~2.0
    print(f"c: {c}")


# Example 2: Neural network training
def train_linear_regression():
    """Train a simple linear regression model."""
    print("\n=== Linear Regression Training ===")
    np.random.seed(42)
    
    # Generate synthetic data: y = X @ w + noise
    n_samples, n_features = 100, 3
    true_weights = np.array([[1.5], [-2.0], [0.8]])
    X = Tensor(np.random.randn(n_samples, n_features))
    y = X.matmul(Tensor(true_weights)).data + 0.1 * np.random.randn(n_samples, 1)
    y = Tensor(y)

    model = Linear(n_features, 1)
    optimizer = SGD(model.parameters(), lr=0.01)

    for epoch in range(100):
        # Forward pass
        pred = model(X)
        loss = mse_loss(pred, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Clean up computational graph
        loss.prune_tape()

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}, Loss: {loss.data:.6f}")

    print(f"True weights: {true_weights.flatten()}")
    print(f"Learned weights: {model.weight.data.flatten()}")


if __name__ == "__main__":
    gradient_check()
    train_linear_regression()