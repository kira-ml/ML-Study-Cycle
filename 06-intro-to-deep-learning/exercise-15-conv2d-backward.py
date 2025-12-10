# exercise-15-conv2d-backward.py
"""Optimized Conv2D implementation with im2col for efficient computation."""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange
from typing import Tuple, Optional
import time

# ============================================================================
# Optimized im2col functions with Numba JIT compilation
# ============================================================================

@jit(nopython=True, parallel=True, cache=True)
def im2col_numba(x: np.ndarray, kh: int, kw: int, pad: int = 0, stride: int = 1) -> np.ndarray:
    """Optimized im2col using Numba for speed."""
    if pad > 0:
        x_padded = np.zeros((x.shape[0], x.shape[1], x.shape[2] + 2*pad, x.shape[3] + 2*pad), dtype=x.dtype)
        x_padded[:, :, pad:-pad, pad:-pad] = x
        x = x_padded
    
    N, C, H, W = x.shape
    out_h = (H - kh) // stride + 1
    out_w = (W - kw) // stride + 1
    
    # Pre-allocate output array
    cols = np.zeros((kh * kw * C, N * out_h * out_w), dtype=x.dtype)
    
    for n in prange(N):
        for c in range(C):
            for h in range(out_h):
                for w in range(out_w):
                    col_idx = n * out_h * out_w + h * out_w + w
                    row_start = c * kh * kw
                    
                    for i in range(kh):
                        for j in range(kw):
                            row_idx = row_start + i * kw + j
                            cols[row_idx, col_idx] = x[n, c, h*stride + i, w*stride + j]
    
    return cols

@jit(nopython=True, parallel=True, cache=True)
def col2im_numba(cols: np.ndarray, x_shape: Tuple, kh: int, kw: int, pad: int = 0, stride: int = 1) -> np.ndarray:
    """Optimized col2im using Numba for speed."""
    N, C, H, W = x_shape
    H_pad, W_pad = H + 2*pad, W + 2*pad
    x_padded = np.zeros((N, C, H_pad, W_pad), dtype=cols.dtype)
    
    out_h = (H_pad - kh) // stride + 1
    out_w = (W_pad - kw) // stride + 1
    
    for n in prange(N):
        for c in range(C):
            for h in range(out_h):
                for w in range(out_w):
                    col_idx = n * out_h * out_w + h * out_w + w
                    row_start = c * kh * kw
                    
                    for i in range(kh):
                        for j in range(kw):
                            row_idx = row_start + i * kw + j
                            x_padded[n, c, h*stride + i, w*stride + j] += cols[row_idx, col_idx]
    
    return x_padded if pad == 0 else x_padded[:, :, pad:-pad, pad:-pad]

# ============================================================================
# Vectorized im2col functions (alternative implementation)
# ============================================================================

def im2col_vectorized(x: np.ndarray, kh: int, kw: int, pad: int = 0, stride: int = 1) -> np.ndarray:
    """Vectorized im2col using as_strided for maximum performance."""
    if pad > 0:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    
    N, C, H, W = x.shape
    out_h = (H - kh) // stride + 1
    out_w = (W - kw) // stride + 1
    
    # Create view into array with striding
    shape = (N, C, out_h, out_w, kh, kw)
    strides = (x.strides[0], x.strides[1], stride * x.strides[2], stride * x.strides[3], 
               x.strides[2], x.strides[3])
    
    windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides, writeable=False)
    
    # Reshape to column format
    cols = windows.transpose(1, 4, 5, 0, 2, 3).reshape(C * kh * kw, N * out_h * out_w)
    
    return cols

def col2im_vectorized(cols: np.ndarray, x_shape: Tuple, kh: int, kw: int, pad: int = 0, stride: int = 1) -> np.ndarray:
    """Vectorized col2im using numpy operations."""
    N, C, H, W = x_shape
    H_pad, W_pad = H + 2*pad, W + 2*pad
    
    # Reshape columns back
    out_h = (H_pad - kh) // stride + 1
    out_w = (W_pad - kw) // stride + 1
    
    cols_reshaped = cols.reshape(C, kh, kw, N, out_h, out_w)
    
    # Create output array
    x_padded = np.zeros((N, C, H_pad, W_pad), dtype=cols.dtype)
    
    for c in range(C):
        for i in range(kh):
            for j in range(kw):
                x_padded[:, c, i:i+out_h*stride:stride, j:j+out_w*stride:stride] += cols_reshaped[c, i, j]
    
    return x_padded if pad == 0 else x_padded[:, :, pad:-pad, pad:-pad]

# ============================================================================
# Optimized Conv2D Layer
# ============================================================================

class Conv2D:
    """Optimized 2D Convolution layer with multiple acceleration options."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, use_numba: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding
        self.use_numba = use_numba
        
        # He initialization for better training
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        scale = np.sqrt(2.0 / fan_in)
        self.W = np.random.randn(out_channels, in_channels, *self.kernel_size) * scale
        self.b = np.zeros((out_channels, 1))
        
        # Cache
        self.x_shape = None
        self.x_col = None
        self.output_shape = None
        
        # Choose im2col implementation
        self.im2col_fn = im2col_numba if use_numba else im2col_vectorized
        self.col2im_fn = col2im_numba if use_numba else col2im_vectorized
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass using optimized im2col."""
        self.x_shape = x.shape
        N, C, H, W = x.shape
        
        # Calculate output dimensions
        out_h = (H + 2*self.padding - self.kernel_size[0]) // self.stride + 1
        out_w = (W + 2*self.padding - self.kernel_size[1]) // self.stride + 1
        self.output_shape = (N, self.out_channels, out_h, out_w)
        
        # Convert input to column format
        self.x_col = self.im2col_fn(x, self.kernel_size[0], self.kernel_size[1], 
                                   self.padding, self.stride)
        
        # Reshape weights for matrix multiplication
        w_row = self.W.reshape(self.out_channels, -1)
        
        # Matrix multiplication
        out = w_row @ self.x_col + self.b
        
        # Reshape to output format
        return out.reshape(self.out_channels, out_h, out_w, N).transpose(3, 0, 1, 2)
    
    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backward pass with optimized gradients."""
        N, F, out_h, out_w = dout.shape
        
        # Reshape gradient
        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(F, -1)
        
        # Bias gradient (sum over all axes except output channels)
        db = np.sum(dout, axis=(0, 2, 3), keepdims=True).reshape(F, 1)
        
        # Weight gradient
        dW = dout_reshaped @ self.x_col.T
        dW = dW.reshape(self.W.shape)
        
        # Input gradient
        w_reshaped = self.W.reshape(F, -1)
        dx_col = w_reshaped.T @ dout_reshaped
        dx = self.col2im_fn(dx_col, self.x_shape, self.kernel_size[0], 
                          self.kernel_size[1], self.padding, self.stride)
        
        return dx, dW, db
    
    def update(self, dW: np.ndarray, db: np.ndarray, lr: float = 0.01) -> None:
        """Update parameters with learning rate."""
        self.W -= lr * dW
        self.b -= lr * db

# ============================================================================
# Optimized Numerical Gradient
# ============================================================================

@jit(nopython=True, parallel=True, cache=True)
def numerical_gradient_fast(f, x: np.ndarray, h: float = 1e-4) -> np.ndarray:
    """Parallel numerical gradient computation."""
    grad = np.zeros_like(x)
    n_elements = x.size
    
    for i in prange(n_elements):
        # Compute flat index
        idx = np.unravel_index(i, x.shape)
        
        # Save original value
        orig = x[idx]
        
        # f(x + h)
        x[idx] = orig + h
        f_plus = f(x)
        
        # f(x - h)
        x[idx] = orig - h
        f_minus = f(x)
        
        # Central difference
        grad[idx] = (f_plus - f_minus) / (2 * h)
        
        # Restore original value
        x[idx] = orig
    
    return grad

# ============================================================================
# Optimized Visualization
# ============================================================================

def visualize_convolution_process():
    """Optimized convolution visualization."""
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle('Convolution Process Visualization', fontsize=16, fontweight='bold')
    
    # Input and kernel
    input_img = np.arange(1, 17).reshape(1, 1, 4, 4)
    kernel = np.array([[[1, 0], [-1, 0]]])
    
    # Pre-compute convolution using vectorized operations
    patches = np.lib.stride_tricks.sliding_window_view(input_img[0, 0], (2, 2)).reshape(9, 2, 2)
    conv_result = np.sum(patches * kernel[0], axis=(1, 2)).reshape(3, 3)
    
    # Plotting configurations
    plots_config = [
        (input_img[0, 0], 'viridis', 'Input Image\n(4x4)', axes[0, 0]),
        (kernel[0], 'RdBu', 'Convolution Kernel\n(2x2)', axes[0, 1]),
        (conv_result, 'viridis', 'Convolved Output\n(3x3)', axes[0, 2])
    ]
    
    for data, cmap, title, ax in plots_config:
        im = ax.imshow(data, cmap=cmap, interpolation='none', 
                      vmin=-1 if cmap == 'RdBu' else None, 
                      vmax=1 if cmap == 'RdBu' else None)
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(range(data.shape[1]))
        ax.set_yticks(range(data.shape[0]))
        
        # Add text annotations
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                color = 'white' if (cmap == 'RdBu' and data[i, j] < 0) or data[i, j] > 8 else 'black'
                ax.text(j, i, f'{data[i, j]:.0f}', ha='center', va='center', color=color)
        ax.grid(color='white', linewidth=1)
    
    # im2col transformation
    x_col = im2col_vectorized(input_img, 2, 2)
    axes[0, 3].imshow(x_col, cmap='Blues', aspect='auto')
    axes[0, 3].set_title('Im2Col Transformation\n(4x9)', fontweight='bold')
    
    # Gradient visualizations
    gradient_info = [
        ('dx Computation', 'Input Gradient\nBackpropagation', 'lightblue', axes[1, 0]),
        ('dW Computation', 'Weight Gradient\nBackpropagation', 'lightgreen', axes[1, 1]),
        ('db Computation', 'Bias Gradient\nBackpropagation', 'lightcoral', axes[1, 2]),
        ('Gradient Checking', 'Numerical vs Analytical\nGradient Comparison', 'yellow', axes[1, 3])
    ]
    
    for title, text, color, ax in gradient_info:
        ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color))
        ax.set_title(title, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# Optimized Testing Functions
# ============================================================================

def test_conv2d_gradients(use_numba: bool = False):
    """Optimized gradient testing with timing."""
    print(f"Testing Conv2D gradients ({'Numba' if use_numba else 'Vectorized'})...")
    
    # Create test data
    np.random.seed(42)  # For reproducibility
    x = np.random.randn(2, 3, 8, 8) * 0.1  # Larger batch for better timing
    
    # Create layer
    layer = Conv2D(in_channels=3, out_channels=4, kernel_size=3, 
                  stride=1, padding=1, use_numba=use_numba)
    
    # Time forward pass
    start = time.perf_counter()
    out = layer.forward(x)
    forward_time = time.perf_counter() - start
    
    # Loss function closure
    def create_loss_fn(layer, x):
        def loss_fn(params):
            orig_W, orig_b = layer.W.copy(), layer.b.copy()
            layer.W, layer.b = params['W'], params['b']
            loss = np.sum(layer.forward(x) ** 2)
            layer.W, layer.b = orig_W, orig_b
            return loss
        return loss_fn
    
    loss_fn = create_loss_fn(layer, x)
    
    # Analytical gradients
    dout = 2 * out  # Gradient of sum of squares
    start = time.perf_counter()
    dx, dW, db = layer.backward(dout)
    backward_time = time.perf_counter() - start
    
    # Numerical gradients with timing
    params = {'W': layer.W, 'b': layer.b}
    
    start = time.perf_counter()
    num_dW = numerical_gradient_fast(lambda w: loss_fn({'W': w, 'b': params['b']}), layer.W)
    num_db = numerical_gradient_fast(lambda b: loss_fn({'W': params['W'], 'b': b}), layer.b)
    num_dx = numerical_gradient_fast(lambda x_in: np.sum(layer.forward(x_in) ** 2), x)
    numerical_time = time.perf_counter() - start
    
    # Calculate errors
    def relative_error(a, b):
        return np.mean(np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-8))
    
    errors = {
        'W': (np.max(np.abs(dW - num_dW)), relative_error(dW, num_dW)),
        'b': (np.max(np.abs(db - num_db)), relative_error(db, num_db)),
        'x': (np.max(np.abs(dx - num_dx)), relative_error(dx, num_dx))
    }
    
    # Print results
    print(f"Forward pass: {forward_time*1000:.2f} ms")
    print(f"Backward pass: {backward_time*1000:.2f} ms")
    print(f"Numerical gradients: {numerical_time*1000:.2f} ms")
    
    for param, (max_err, rel_err) in errors.items():
        print(f"{param}: max error={max_err:.6e}, relative error={rel_err:.6e}")
    
    # Verify gradients
    tolerance = 1e-5
    all_pass = all(max_err < tolerance for max_err, _ in errors.values())
    
    if all_pass:
        print("✓ All gradient checks passed!")
    else:
        print("✗ Some gradient checks failed!")
    
    return errors

# ============================================================================
# Optimized Training Function
# ============================================================================

class SimpleCNN:
    """Optimized simple CNN for toy classification."""
    
    def __init__(self, in_channels=1, conv_channels=4, kernel_size=2, num_classes=2):
        self.conv = Conv2D(in_channels, conv_channels, kernel_size)
        self.W_class = np.random.randn(num_classes, conv_channels * 3 * 3) * 0.1
        self.b_class = np.zeros((num_classes, 1))
    
    def forward(self, x, return_conv=False):
        conv_out = self.conv.forward(x)
        flat = conv_out.reshape(conv_out.shape[0], -1)
        scores = self.W_class @ flat.T + self.b_class
        return (scores.T, conv_out) if return_conv else scores.T
    
    def backward(self, x, dscores):
        conv_out = self.conv.forward(x)  # Recompute forward pass
        flat = conv_out.reshape(conv_out.shape[0], -1)
        
        # Classifier gradients
        dW_class = dscores.T @ flat
        db_class = np.sum(dscores.T, axis=1, keepdims=True)
        
        # Convolution gradients
        dconv_out = (dscores @ self.W_class).reshape(conv_out.shape)
        dx, dW_conv, db_conv = self.conv.backward(dconv_out)
        
        return dx, dW_conv, db_conv, dW_class, db_class
    
    def update(self, grads, lr=0.01):
        dW_conv, db_conv, dW_class, db_class = grads
        self.conv.update(dW_conv, db_conv, lr)
        self.W_class -= lr * dW_class
        self.b_class -= lr * db_class

def train_toy_classifier(epochs=100, lr=0.01, batch_size=5):
    """Optimized training with vectorized operations."""
    print(f"\nTraining CNN classifier...")
    
    # Create dataset efficiently
    X = np.zeros((10, 1, 4, 4))
    y = np.zeros(10, dtype=int)
    
    # Horizontal lines (class 0)
    X[:5, 0, 1:3, :] = 1.0
    # Vertical lines (class 1)
    X[5:, 0, :, 1:3] = 1.0
    y[5:] = 1
    
    # Initialize model
    model = SimpleCNN()
    
    # Training storage
    losses = np.zeros(epochs)
    accuracies = np.zeros(epochs)
    
    # Pre-allocate arrays
    onehot_y = np.eye(2)[y]
    
    for epoch in range(epochs):
        # Forward pass
        scores = model.forward(X)
        
        # Softmax with numerical stability
        scores -= np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        # Loss and accuracy
        losses[epoch] = -np.mean(np.log(probs[np.arange(10), y] + 1e-8))
        accuracies[epoch] = np.mean(np.argmax(scores, axis=1) == y)
        
        # Backward pass
        dscores = probs - onehot_y
        dscores /= len(y)
        
        grads = model.backward(X, dscores)
        model.update(grads[1:], lr)  # Skip dx
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: Loss={losses[epoch]:.4f}, Acc={accuracies[epoch]:.4f}")
    
    # Final evaluation
    final_preds = np.argmax(model.forward(X), axis=1)
    final_acc = np.mean(final_preds == y)
    print(f"Final accuracy: {final_acc:.4f}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(losses, 'b-', linewidth=2)
    ax1.set_title('Training Loss', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(accuracies, 'g-', linewidth=2)
    ax2.set_title('Training Accuracy', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return final_acc

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 15: Optimized Conv Layer Backward and Gradients")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 1. Visualization
    visualize_convolution_process()
    
    # 2. Test both implementations
    print("\n" + "=" * 60)
    print("Testing Vectorized Implementation")
    print("=" * 60)
    errors_vec = test_conv2d_gradients(use_numba=False)
    
    print("\n" + "=" * 60)
    print("Testing Numba Implementation")
    print("=" * 60)
    errors_numba = test_conv2d_gradients(use_numba=True)
    
    # 3. Train classifier
    print("\n" + "=" * 60)
    print("Training Toy Classifier")
    print("=" * 60)
    accuracy = train_toy_classifier(epochs=100, lr=0.01)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✓ Convolution process visualized")
    print("✓ Gradient checks completed for both implementations")
    print(f"✓ Toy classifier trained to {accuracy:.2%} accuracy")
    print("=" * 60)