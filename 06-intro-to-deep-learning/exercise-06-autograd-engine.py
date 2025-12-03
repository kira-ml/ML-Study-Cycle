"""
🚀 TinyTorch: A Minimal Autodiff Framework
Author: Kira-ML
Purpose: Educational implementation of automatic differentiation
Concept: Build computational graphs dynamically, compute gradients via chain rule
"""

import numpy as np


# 🔧 HELPER FUNCTION: Broadcasting Gradient Reduction
def _broadcast_gradient(grad, target_shape):
    """
    📐 Adjust gradients after broadcasting operations.
    
    When we add [3x1] + [1x3], numpy broadcasts to [3x3].
    During backprop, the [3x3] gradient must be reduced to match 
    the original [3x1] and [1x3] shapes.
    
    Broadcasting Example:
        Original: [3, 1] + [1, 3] → [3, 3] (broadcast happens)
        Gradient: [3, 3] → sum over appropriate axes → [3, 1] or [1, 3]
    
    Args:
        grad: Gradient from output (may be broadcasted shape)
        target_shape: Shape we need to match for gradient accumulation
    
    Returns:
        Reduced gradient matching target_shape
    """
    # Quick exit if shapes already match
    if grad.shape == target_shape:
        return grad
    
    # Case 1: Extra dimensions were added (e.g., scalar → vector)
    ndim_diff = grad.ndim - len(target_shape)
    if ndim_diff > 0:
        # Sum over the extra dimensions that were added
        grad = grad.sum(axis=tuple(range(ndim_diff)))
    
    # Case 2: Dimensions were expanded from size 1 to size N
    axes = tuple(i for i, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape)) 
                if t_dim == 1 and g_dim > 1)
    if axes:
        # Sum over dimensions that were broadcasted
        grad = grad.sum(axis=axes, keepdims=True)
    
    return grad


# 🧮 CORE CLASS: Tensor with Autodiff
class Tensor:
    """
    📦 A tensor that remembers how it was created for automatic differentiation.
    
    Think of tensors as LEGO blocks that remember:
    1. Their value (self.data)
    2. If they need gradients (self.requires_grad)
    3. Their parent tensors in the computational graph (self._children)
    4. How to compute their gradient (self._backward)
    
    Key Insight: Every operation builds a computational graph!
    Example: c = a + b creates:
        c._children = {a, b}
        c._backward = function that tells a.grad and b.grad how to update
    
    Attributes:
        data: Numerical values stored as numpy array
        requires_grad: Flag for gradient computation (True for parameters)
        grad: Accumulated gradient (dLoss/dThisTensor)
        _op: Operation symbol for debugging (+, *, ReLU, etc.)
        _children: Input tensors that created this tensor
        _backward: Function to propagate gradient to children
    """
    
    def __init__(self, data, requires_grad=False, _op='', _children=()):
        """
        Initialize a new tensor.
        
        Args:
            data: Can be number, list, or numpy array
            requires_grad: Track gradients? (True for learnable parameters)
            _op: Operation that created this (for visualization/debugging)
            _children: Parent tensors in computational graph
        """
        # Convert input to numpy array for consistent operations
        self.data = np.array(data, dtype=np.float32)
        
        # Gradient tracking
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        
        # Computational graph metadata
        self._op = _op  # What operation made me?
        self._children = set(_children)  # Who are my parents?
        self._backward = lambda: None  # How do I tell my parents their gradients?
    
    def __repr__(self):
        """Pretty print for debugging."""
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad}, grad={self.grad})"
    
    # ➕ ADDITION OPERATION
    def __add__(self, other):
        """
        Element-wise addition: self + other
        
        Gradient Insight: Addition distributes gradient equally to both inputs.
        If c = a + b, then:
            dc/da = 1
            dc/db = 1
        So during backward: a.grad += c.grad, b.grad += c.grad
        """
        # Convert non-tensor to tensor (e.g., Tensor(5) + 3 works)
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        # Create output tensor (child of self and other)
        out = Tensor(
            self.data + other.data,  # Forward pass: compute values
            requires_grad=self.requires_grad or other.requires_grad,  # Need grad if any parent does
            _op='+',
            _children=(self, other)  # Remember parents
        )
        
        # Define how to propagate gradients backward
        def _backward():
            # Addition rule: gradient flows unchanged to both parents
            if self.requires_grad:
                # Handle broadcasting if shapes don't match
                self.grad += _broadcast_gradient(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _broadcast_gradient(out.grad, other.data.shape)
        
        # Attach the backward function to the output tensor
        out._backward = _backward
        return out
    
    # ✖️ MULTIPLICATION OPERATION
    def __mul__(self, other):
        """
        Element-wise multiplication: self * other
        
        Gradient Insight: Multiplication scales the gradient by the other input.
        If c = a * b, then:
            dc/da = b
            dc/db = a
        So during backward: a.grad += b * c.grad, b.grad += a * c.grad
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _op='*',
            _children=(self, other)
        )
        
        def _backward():
            # Chain rule: d(out)/d(self) = other * d(out)/d(out)
            if self.requires_grad:
                grad = other.data * out.grad  # Scale by other's value
                self.grad += _broadcast_gradient(grad, self.data.shape)
            if other.requires_grad:
                grad = self.data * out.grad  # Scale by self's value
                other.grad += _broadcast_gradient(grad, other.data.shape)
        
        out._backward = _backward
        return out
    
    # ➖ SUBTRACTION & NEGATION
    def __sub__(self, other):
        """Subtraction: self - other = self + (-other)"""
        return self + (-other)
    
    def __neg__(self):
        """Negation: -self = self * -1"""
        return self * -1
    
    # 💪 POWER OPERATION
    def __pow__(self, exponent):
        """
        Power: self ** exponent
        
        Gradient Insight: Power rule from calculus!
        If y = x^n, then dy/dx = n * x^(n-1)
        """
        if not isinstance(exponent, (int, float)):
            raise ValueError("Exponent must be a scalar (int or float)")
            
        out = Tensor(
            self.data ** exponent,
            requires_grad=self.requires_grad,
            _op=f'**{exponent}',
            _children=(self,)
        )
        
        def _backward():
            if self.requires_grad:
                # Power rule: n * x^(n-1) * d(out)/d(out)
                self.grad += exponent * (self.data ** (exponent - 1)) * out.grad
        
        out._backward = _backward
        return out
    
    # 🔷 MATRIX MULTIPLICATION
    def matmul(self, other):
        """
        Matrix multiplication: self @ other
        
        Gradient Insight: For matrix multiplication A @ B = C:
            dC/dA = dC/dC @ B.T  (gradient flows through B transposed)
            dC/dB = A.T @ dC/dC   (gradient flows through A transposed)
        This is why we need .T (transpose) in the backward pass!
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        out = Tensor(
            self.data @ other.data,  # @ is numpy's matmul operator
            requires_grad=self.requires_grad or other.requires_grad,
            _op='@',
            _children=(self, other)
        )
        
        def _backward():
            # Matrix gradient rules (derived from matrix calculus)
            if self.requires_grad:
                # d(out)/d(self) = out.grad @ other^T
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                # d(out)/d(other) = self^T @ out.grad
                other.grad += self.data.T @ out.grad
        
        out._backward = _backward
        return out
    
    # 🧠 ACTIVATION FUNCTIONS
    
    def relu(self):
        """
        Rectified Linear Unit: max(0, x)
        
        Why ReLU? It helps with vanishing gradient problem!
        Gradient: 1 if x > 0, else 0 (very simple!)
        
        Visual:
           │      /
           │     /
           │____/______
           │   /
           │  /
           │ /
        """
        out = Tensor(
            np.maximum(0, self.data),  # Element-wise: x if x > 0, else 0
            requires_grad=self.requires_grad,
            _op='ReLU',
            _children=(self,)
        )
        
        def _backward():
            if self.requires_grad:
                # ReLU derivative: 1 where input > 0, else 0
                self.grad += (out.data > 0) * out.grad
        
        out._backward = _backward
        return out
    
    def sigmoid(self):
        """
        Sigmoid: 1 / (1 + exp(-x))
        
        Maps any input to (0, 1) range. Great for probabilities!
        Gradient: σ(x) * (1 - σ(x)) - peaks at 0.25 when x=0
        
        Pro tip: Store sigmoid_data for efficiency in backward pass!
        """
        # Compute once, use twice (forward and backward)
        sigmoid_data = 1 / (1 + np.exp(-self.data))
        
        out = Tensor(
            sigmoid_data,
            requires_grad=self.requires_grad,
            _op='sigmoid',
            _children=(self,)
        )
        
        def _backward():
            if self.requires_grad:
                # Beautiful property: derivative uses the forward result!
                # σ'(x) = σ(x) * (1 - σ(x))
                self.grad += sigmoid_data * (1 - sigmoid_data) * out.grad
        
        out._backward = _backward
        return out
    
    def tanh(self):
        """
        Hyperbolic tangent: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        
        Maps to (-1, 1) range. Often better than sigmoid!
        Gradient: 1 - tanh²(x) - flows better than sigmoid
        """
        tanh_data = np.tanh(self.data)
        
        out = Tensor(
            tanh_data,
            requires_grad=self.requires_grad,
            _op='tanh',
            _children=(self,)
        )
        
        def _backward():
            if self.requires_grad:
                # Another nice property: derivative uses forward result
                # tanh'(x) = 1 - tanh²(x)
                self.grad += (1 - tanh_data ** 2) * out.grad
        
        out._backward = _backward
        return out
    
    def exp(self):
        """Exponential: exp(x) - derivative is exp(x) itself!"""
        exp_data = np.exp(self.data)
        
        out = Tensor(
            exp_data,
            requires_grad=self.requires_grad,
            _op='exp',
            _children=(self,)
        )
        
        def _backward():
            if self.requires_grad:
                # exp'(x) = exp(x) - how elegant!
                self.grad += exp_data * out.grad
        
        out._backward = _backward
        return out
    
    def log(self):
        """Natural log: log(x) - derivative is 1/x"""
        out = Tensor(
            np.log(self.data),
            requires_grad=self.requires_grad,
            _op='log',
            _children=(self,)
        )
        
        def _backward():
            if self.requires_grad:
                # log'(x) = 1/x
                self.grad += (1 / self.data) * out.grad
        
        out._backward = _backward
        return out
    
    # 📊 REDUCTION OPERATIONS
    
    def sum(self, axis=None, keepdims=False):
        """
        Sum elements along specified axis.
        
        Gradient Insight: Sum distributes the output gradient 
        to all input elements equally.
        If y = sum(x), then dy/dx_i = 1 for all i
        """
        out = Tensor(
            np.sum(self.data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _op='sum',
            _children=(self,)
        )
        
        def _backward():
            if self.requires_grad:
                grad = out.grad
                # If we summed dimensions away, we need to add them back
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                # Broadcast gradient to original shape
                self.grad += np.broadcast_to(grad, self.data.shape)
        
        out._backward = _backward
        return out
    
    def mean(self, axis=None, keepdims=False):
        """
        Mean (average) along specified axis.
        
        Gradient Insight: Mean divides gradient equally among inputs.
        If y = mean(x) and x has n elements, then dy/dx_i = 1/n
        """
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
                # Count how many elements were averaged
                if axis is None:
                    count = np.prod(self.data.shape)  # All elements
                else:
                    # Product of dimensions along the reduction axes
                    count = np.prod([self.data.shape[i] for i in 
                                    (axis if isinstance(axis, tuple) else (axis,))])
                # Distribute gradient: each element gets grad/count
                self.grad += np.broadcast_to(grad / count, self.data.shape)
        
        out._backward = _backward
        return out
    
    # ⚙️ AUTODIFF ENGINE
    
    def backward(self, gradient=None):
        """
        🎯 The Magic Function: Backpropagate gradients through entire graph!
        
        This implements reverse-mode automatic differentiation:
        1. Build topological order of computational graph (forward to backward)
        2. Apply chain rule in reverse order
        3. Accumulate gradients in each tensor's .grad attribute
        
        Chain Rule Visual:
            If L = loss, and c = a + b, then:
            dL/da = dL/dc * dc/da
                   ↑        ↑
            (from c.grad)  (from c._backward)
        
        Args:
            gradient: Initial gradient (defaults to 1, assuming scalar output)
        """
        # If no gradient provided, assume scalar output (like loss)
        if gradient is None:
            gradient = np.ones_like(self.data)
        self.grad = gradient  # Start gradient at output
        
        # Step 1: Topological sort - order tensors from inputs to output
        topo = []  # Will store tensors in topological order
        visited = set()
        
        def build_topo(v):
            """
            Depth-first search to build topological order.
            
            Why topological sort? We need to process children before parents!
            Example: For c = a + b, we need a and b before c in backward pass.
            """
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)  # Visit children first
                topo.append(v)  # Add parent after children
        
        build_topo(self)  # Start from output tensor
        
        # Step 2: Backward pass in reverse topological order
        for v in reversed(topo):
            v._backward()  # Apply chain rule at this node
        
        print(f"✓ Backward pass complete. Processed {len(topo)} tensors.")
    
    # 🧹 UTILITY METHODS
    
    def zero_grad(self):
        """Reset gradients to zero before next backward pass."""
        if self.grad is not None:
            self.grad.fill(0)
    
    def detach(self):
        """Create a copy without gradient tracking."""
        return Tensor(self.data, requires_grad=False)
    
    def prune_tape(self):
        """
        Free memory by clearing computational graph references.
        
        Why prune? After backward pass, we don't need intermediate
        tensors anymore. This helps prevent memory leaks in big models.
        """
        for child in self._children:
            if child.requires_grad:
                child.prune_tape()
        self._children = set()  # Break references for garbage collection
        self._backward = lambda: None  # Clear backward function
    
    # Enable operations like 3 + tensor (right-side operations)
    __radd__ = __add__  # Enables: 3 + tensor
    __rmul__ = __mul__  # Enables: 3 * tensor


# 🧱 NEURAL NETWORK LAYERS

class Linear:
    """
    📐 A fully-connected (dense) neural network layer.
    
    Formula: output = input @ weight + bias
    
    Where:
        @ = matrix multiplication
        input shape: [batch_size, input_size]
        weight shape: [input_size, output_size]
        bias shape: [output_size]
        output shape: [batch_size, output_size]
    
    Example: If input_size=10, output_size=5, batch_size=32:
        input: [32, 10]
        weight: [10, 5]
        output: [32, 5] = [32, 10] @ [10, 5] + [5]
    """
    
    def __init__(self, input_size, output_size):
        """
        Initialize weights with Xavier/Glorot initialization.
        
        Why Xavier initialization? It keeps variance of activations 
        consistent across layers, helping with gradient flow.
        
        Scale factor: sqrt(2 / (input_size + output_size))
        """
        # Heuristic for ReLU-like activations
        scale = np.sqrt(2.0 / (input_size + output_size))
        
        # Weight matrix - the main learnable parameters
        self.weight = Tensor(
            np.random.randn(input_size, output_size) * scale,
            requires_grad=True  # These need gradients!
        )
        
        # Bias vector - also learnable
        self.bias = Tensor(
            np.zeros(output_size),
            requires_grad=True
        )
    
    def __call__(self, x):
        """Forward pass through the layer."""
        # y = Wx + b
        return x.matmul(self.weight) + self.bias
    
    def parameters(self):
        """Return all learnable parameters (for optimizer)."""
        return [self.weight, self.bias]
    
    def zero_grad(self):
        """Reset gradients for all parameters."""
        for param in self.parameters():
            param.zero_grad()


# 📉 LOSS FUNCTIONS

def mse_loss(pred, target):
    """
    📏 Mean Squared Error Loss for regression tasks.
    
    Formula: MSE = mean((pred - target)²)
    
    Properties:
        - Always non-negative
        - Penalizes large errors heavily (quadratic)
        - Derivative: 2 * (pred - target) / n
        
    Use for: Predicting continuous values (price, temperature, etc.)
    """
    # Square the differences, then average
    return ((pred - target) ** 2).mean()


def cross_entropy_loss(pred, target):
    """
    🎯 Cross Entropy Loss for classification tasks.
    
    Formula: CE = -Σ(target * log(softmax(pred)))
    
    Numerical stability trick: Subtract max before softmax
    This prevents overflow in exp() while keeping results identical.
    
    Use for: Multi-class classification (cat/dog/bird, etc.)
    """
    # Step 1: Numerical stability - subtract max
    # This doesn't change probabilities but prevents huge exp() values
    max_val = pred.data.max(axis=-1, keepdims=True)
    stable_pred = pred - Tensor(max_val)
    
    # Step 2: Softmax = exp(x) / sum(exp(x))
    exp_pred = stable_pred.exp()
    softmax_val = exp_pred / exp_pred.sum(axis=-1, keepdims=True)
    
    # Step 3: Handle target format (indices or one-hot)
    if len(target.data.shape) == 1:  # Class indices [0, 2, 1, ...]
        # Convert to one-hot for gradient computation
        n_classes = softmax_val.data.shape[-1]
        target_one_hot = np.eye(n_classes)[target.data.astype(int)]
        target = Tensor(target_one_hot)
    
    # Step 4: Compute log softmax more stably
    # log_softmax = x - log(sum(exp(x)))
    log_softmax = stable_pred - exp_pred.sum(axis=-1, keepdims=True).log()
    
    # Step 5: Cross entropy = -Σ(target * log_softmax)
    return -(target * log_softmax).sum(axis=-1).mean()


# 🏃 OPTIMIZER

class SGD:
    """
    🏃 Stochastic Gradient Descent Optimizer.
    
    Update rule: param = param - learning_rate * gradient
    
    This is the simplest optimizer - just follow the negative gradient!
    More advanced optimizers (Adam, RMSProp) add momentum, adaptive rates, etc.
    
    Visual: Imagine rolling down a hill - always step in the steepest 
            downhill direction (negative gradient).
    """
    
    def __init__(self, parameters, lr=0.01):
        """
        Args:
            parameters: List of tensors with requires_grad=True
            lr: Learning rate - how big of a step to take
                Too high → overshoot minimum
                Too low → slow training
        """
        self.parameters = parameters
        self.lr = lr
    
    def step(self):
        """Take one optimization step: update all parameters."""
        for param in self.parameters:
            if param.requires_grad:
                # The core of gradient descent!
                param.data -= self.lr * param.grad
    
    def zero_grad(self):
        """Clear gradients before next forward-backward pass."""
        for param in self.parameters:
            param.zero_grad()


# 🧪 EXAMPLE 1: GRADIENT CHECKING
def gradient_check():
    """
    🧪 Sanity check: Compare our autodiff gradients with numerical gradients.
    
    Numerical gradient formula:
        f'(x) ≈ (f(x + h) - f(x - h)) / (2h)  [central difference]
    
    This validates our backward() implementation is correct!
    """
    print("=" * 50)
    print("🧪 GRADIENT CHECK: Verify Autodiff Implementation")
    print("=" * 50)
    
    # Simple computation: c = a * b
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)
    
    print(f"a = {a.data[0]}, b = {b.data[0]}")
    print(f"Computing c = a * b...")
    
    # Forward pass
    c = a * b
    
    # Backward pass (autodiff)
    c.backward()
    
    print(f"\n🤖 AUTODIFF GRADIENTS:")
    print(f"  dc/da = {a.grad[0]:.6f} (should be b = 3.0)")
    print(f"  dc/db = {b.grad[0]:.6f} (should be a = 2.0)")
    
    # Numerical gradient check (finite differences)
    print(f"\n🧮 NUMERICAL GRADIENT CHECK (h=0.0001):")
    h = 0.0001
    
    # dc/da ≈ (c(a+h) - c(a-h)) / (2h)
    c_plus = Tensor([(2.0 + h) * 3.0])
    c_minus = Tensor([(2.0 - h) * 3.0])
    numeric_da = (c_plus.data[0] - c_minus.data[0]) / (2 * h)
    print(f"  dc/da ≈ {numeric_da:.6f}")
    
    # dc/db ≈ (c(b+h) - c(b-h)) / (2h)
    c_plus = Tensor([2.0 * (3.0 + h)])
    c_minus = Tensor([2.0 * (3.0 - h)])
    numeric_db = (c_plus.data[0] - c_minus.data[0]) / (2 * h)
    print(f"  dc/db ≈ {numeric_db:.6f}")
    
    # Verify match
    da_error = abs(a.grad[0] - numeric_da)
    db_error = abs(b.grad[0] - numeric_db)
    print(f"\n✅ ERROR: da = {da_error:.8f}, db = {db_error:.8f}")
    print("   (Should be < 0.000001 for correct implementation)")


# 🚂 EXAMPLE 2: LINEAR REGRESSION TRAINING
def train_linear_regression():
    """
    🚂 End-to-end training example: Linear regression.
    
    We'll:
    1. Generate synthetic data: y = X @ w_true + noise
    2. Initialize a linear model
    3. Train using gradient descent
    4. Watch as learned weights approach true weights!
    
    This demonstrates the full ML pipeline with our framework.
    """
    print("\n" + "=" * 50)
    print("🚂 LINEAR REGRESSION TRAINING")
    print("=" * 50)
    
    # Fix random seed for reproducibility
    np.random.seed(42)
    print("✓ Random seed fixed for reproducibility")
    
    # Generate synthetic data
    n_samples = 100     # Number of data points
    n_features = 3      # Number of features per sample
    print(f"\n📊 Generating {n_samples} samples with {n_features} features...")
    
    # True weights we want to learn
    true_weights = np.array([[1.5], [-2.0], [0.8]])
    print(f"True weights: {true_weights.flatten()}")
    
    # Create feature matrix X
    X = Tensor(np.random.randn(n_samples, n_features))
    
    # Create labels: y = X @ w + noise
    y_true = X.matmul(Tensor(true_weights))
    noise = 0.1 * np.random.randn(n_samples, 1)  # Small Gaussian noise
    y = Tensor(y_true.data + noise)
    print(f"Added Gaussian noise (σ=0.1) to simulate real data")
    
    # Initialize model and optimizer
    print(f"\n🏗️  Initializing model...")
    model = Linear(n_features, 1)  # Input: n_features, Output: 1 (regression)
    optimizer = SGD(model.parameters(), lr=0.01)
    print(f"Learning rate: {optimizer.lr}")
    
    # Training loop
    print(f"\n🎯 Starting training for 100 epochs:")
    print("-" * 40)
    
    for epoch in range(101):  # 0 to 100 inclusive
        # --- FORWARD PASS ---
        pred = model(X)  # y_pred = X @ w + b
        loss = mse_loss(pred, y)  # L = mean((y_pred - y)²)
        
        # --- BACKWARD PASS ---
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Compute new gradients
        optimizer.step()       # Update weights: w = w - lr * gradient
        
        # Clean up to save memory
        loss.prune_tape()
        
        # Print progress
        if epoch % 20 == 0 or epoch == 100:
            print(f"Epoch {epoch:3d} | Loss: {loss.data[0]:.6f}")
    
    # Final results
    print("-" * 40)
    print(f"\n🎉 TRAINING COMPLETE!")
    print(f"\n📈 Final weights:")
    print(f"  True:      {true_weights.flatten()}")
    print(f"  Learned:   {model.weight.data.flatten()}")
    
    # Calculate error
    error = np.abs(true_weights.flatten() - model.weight.data.flatten())
    print(f"\n📏 Mean absolute error: {np.mean(error):.4f}")
    print("   (Close to 0 means we learned the true relationship!)")


# 🏁 MAIN EXECUTION
if __name__ == "__main__":
    """
    🏁 Run both examples when script is executed directly.
    
    Try modifying:
    1. Learning rate in SGD (try 0.1, 0.001)
    2. Number of training epochs
    3. Noise level in synthetic data
    4. Add more operations to gradient_check()
    """
    
    print("✨ Welcome to TinyTorch - Educational Autodiff Framework ✨")
    print("Created by Kira-ML for ML education\n")
    
    # Run examples
    gradient_check()
    train_linear_regression()
    
    print("\n" + "=" * 50)
    print("🎓 Learning Complete!")
    print("=" * 50)
    print("\nNext steps to explore:")
    print("1. Add more operations (division, transpose, convolution)")
    print("2. Implement more optimizers (Adam, RMSProp)")
    print("3. Build a multi-layer neural network")
    print("4. Add saving/loading of model weights")
    print("\nHappy coding, future ML engineer! 💻🚀")