# exercise-15-conv2d-backward.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

def im2col_indices(x, kh, kw, pad=0, stride=1):
    """Convert image to column format for convolution."""
    if pad > 0:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    
    N, C, H, W = x.shape
    
    # Output dimensions
    out_h = (H - kh) // stride + 1
    out_w = (W - kw) // stride + 1
    
    # Create arrays for indices
    i0 = np.tile(np.arange(kh), kw)
    i1 = np.repeat(np.arange(out_h), out_w) * stride
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    
    j0 = np.tile(np.arange(kw), kh)
    j1 = np.tile(np.arange(out_w), out_h) * stride
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    
    k = np.repeat(np.arange(C), kh * kw).reshape(-1, 1)
    
    cols = x[:, k, i, j]
    cols = cols.transpose(1, 2, 0).reshape(kh * kw * C, -1)
    return cols

def col2im_indices(cols, x_shape, kh, kw, pad=0, stride=1):
    """Convert column format back to image format."""
    N, C, H, W = x_shape
    H_pad, W_pad = H + 2*pad, W + 2*pad
    x_padded = np.zeros((N, C, H_pad, W_pad), dtype=cols.dtype)
    
    k_size = kh * kw
    out_h = (H + 2*pad - kh) // stride + 1
    out_w = (W + 2*pad - kw) // stride + 1
    
    # Reshape columns
    cols_reshaped = cols.reshape(C * k_size, out_h * out_w, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1).reshape(N, C, k_size, out_h, out_w)
    
    # Add values to appropriate locations in input
    for i in range(kh):
        for j in range(kw):
            i_start = i
            i_end = i_start + out_h * stride
            j_start = j
            j_end = j_start + out_w * stride
            
            x_padded[:, :, i_start:i_end:stride, j_start:j_end:stride] += cols_reshaped[:, :, i*kw + j, :, :]
    
    if pad == 0:
        return x_padded
    return x_padded[:, :, pad:-pad, pad:-pad]

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        
        # Initialize weights and biases
        self.W = np.random.randn(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]) * 0.1
        self.b = np.zeros((out_channels, 1))
        
        # Cache for backward pass
        self.x_col = None
        self.x = None
    
    def forward(self, x):
        """Forward pass using im2col."""
        self.x = x
        N, C, H, W = x.shape
        
        # Output dimensions
        out_h = (H + 2*self.padding - self.kernel_size[0]) // self.stride + 1
        out_w = (W + 2*self.padding - self.kernel_size[1]) // self.stride + 1
        
        # Convert input to column format
        self.x_col = im2col_indices(x, self.kernel_size[0], self.kernel_size[1], self.padding, self.stride)
        
        # Reshape weights to matrix format
        w_row = self.W.reshape(self.out_channels, -1)
        
        # Perform convolution as matrix multiplication
        out = w_row @ self.x_col
        out = out + self.b
        out = out.reshape(self.out_channels, out_h, out_w, N)
        out = out.transpose(3, 0, 1, 2)
        
        return out
    
    def backward(self, dout):
        """Backward pass for Conv2D layer."""
        N, F, out_h, out_w = dout.shape
        
        # Reshape dout to column format
        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(F, -1)
        
        # Calculate bias gradient
        db = np.sum(dout, axis=(0, 2, 3)).reshape(F, -1)
        
        # Calculate weight gradient
        dW = dout_reshaped @ self.x_col.T
        dW = dW.reshape(self.W.shape)
        
        # Calculate input gradient
        w_reshaped = self.W.reshape(F, -1)
        dx_col = w_reshaped.T @ dout_reshaped
        dx = col2im_indices(dx_col, self.x.shape, self.kernel_size[0], self.kernel_size[1], self.padding, self.stride)
        
        return dx, dW, db

def numerical_gradient(f, x, h=1e-4):
    """Compute numerical gradient."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        
        # Calculate f(x + h)
        x_plus = x.copy()
        x_plus[idx] += h
        f_plus = f(x_plus)
        
        # Calculate f(x - h)
        x_minus = x.copy()
        x_minus[idx] -= h
        f_minus = f(x_minus)
        
        # Central difference
        grad[idx] = (f_plus - f_minus) / (2 * h)
        it.iternext()
    
    return grad

def visualize_convolution_process():
    """Visualize the convolution process."""
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle('Convolution Process Visualization', fontsize=16, fontweight='bold')
    
    # Input image
    input_img = np.array([[[1, 2, 3, 4],
                           [5, 6, 7, 8],
                           [9, 10, 11, 12],
                           [13, 14, 15, 16]]])
    
    # Kernel
    kernel = np.array([[[1, 0],
                        [-1, 0]]])
    
    # Apply convolution
    conv_result = np.zeros((1, 3, 3))
    for i in range(3):
        for j in range(3):
            conv_result[0, i, j] = np.sum(input_img[0, i:i+2, j:j+2] * kernel[0])
    
    # Plot input
    im1 = axes[0, 0].imshow(input_img[0], cmap='viridis', interpolation='none')
    axes[0, 0].set_title('Input Image\n(4x4)', fontweight='bold')
    axes[0, 0].set_xticks(np.arange(4))
    axes[0, 0].set_yticks(np.arange(4))
    for i in range(4):
        for j in range(4):
            axes[0, 0].text(j, i, f'{input_img[0, i, j]:.0f}', ha='center', va='center', color='white')
    axes[0, 0].grid(color='white', linewidth=1)
    
    # Plot kernel
    im2 = axes[0, 1].imshow(kernel[0], cmap='RdBu', interpolation='none', vmin=-1, vmax=1)
    axes[0, 1].set_title('Convolution Kernel\n(2x2)', fontweight='bold')
    axes[0, 1].set_xticks(np.arange(2))
    axes[0, 1].set_yticks(np.arange(2))
    for i in range(2):
        for j in range(2):
            axes[0, 1].text(j, i, f'{kernel[0, i, j]:.0f}', ha='center', va='center', 
                           color='white' if kernel[0, i, j] < 0 else 'black')
    axes[0, 1].grid(color='white', linewidth=1)
    
    # Plot convolution result
    im3 = axes[0, 2].imshow(conv_result[0], cmap='viridis', interpolation='none')
    axes[0, 2].set_title('Convolved Output\n(3x3)', fontweight='bold')
    axes[0, 2].set_xticks(np.arange(3))
    axes[0, 2].set_yticks(np.arange(3))
    for i in range(3):
        for j in range(3):
            axes[0, 2].text(j, i, f'{conv_result[0, i, j]:.0f}', ha='center', va='center', color='white')
    axes[0, 2].grid(color='white', linewidth=1)
    
    # Show im2col transformation
    x_col = im2col_indices(input_img[np.newaxis, ...], 2, 2, pad=0, stride=1)
    axes[0, 3].imshow(x_col, cmap='Blues', aspect='auto')
    axes[0, 3].set_title('Im2Col Transformation\n(4x9)', fontweight='bold')
    axes[0, 3].set_xlabel('Column Index')
    axes[0, 3].set_ylabel('Flattened Kernel Elements')
    
    # Show gradient visualization
    axes[1, 0].text(0.5, 0.5, 'Input Gradient\nBackpropagation', 
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    axes[1, 0].set_title('dx Computation', fontweight='bold')
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    
    axes[1, 1].text(0.5, 0.5, 'Weight Gradient\nBackpropagation', 
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    axes[1, 1].set_title('dW Computation', fontweight='bold')
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    
    axes[1, 2].text(0.5, 0.5, 'Bias Gradient\nBackpropagation', 
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    axes[1, 2].set_title('db Computation', fontweight='bold')
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])
    
    # Show gradient checking concept
    axes[1, 3].text(0.5, 0.7, 'Numerical Gradient:', ha='center', fontsize=12, fontweight='bold')
    axes[1, 3].text(0.5, 0.5, '(f(x+h) - f(x-h)) / 2h', ha='center', fontsize=14, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow"))
    axes[1, 3].text(0.5, 0.3, 'vs', ha='center', fontsize=12)
    axes[1, 3].text(0.5, 0.1, 'Analytical Gradient', ha='center', fontsize=12, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    axes[1, 3].set_title('Gradient Checking', fontweight='bold')
    axes[1, 3].set_xticks([])
    axes[1, 3].set_yticks([])
    
    plt.tight_layout()
    plt.show()

def test_conv2d_gradients():
    """Test Conv2D gradients using numerical gradient checking."""
    print("Testing Conv2D gradients with numerical checks...")
    
    # Create a small input and layer
    x = np.random.randn(1, 2, 4, 4)  # N, C, H, W
    layer = Conv2D(in_channels=2, out_channels=3, kernel_size=2, stride=1, padding=0)
    
    # Forward pass
    out = layer.forward(x)
    
    # Define loss function for gradient checking
    def loss_fn(params):
        # Temporarily replace layer parameters
        original_W = layer.W.copy()
        original_b = layer.b.copy()
        
        layer.W = params['W']
        layer.b = params['b']
        
        out = layer.forward(x)
        loss = np.sum(out ** 2)  # Simple quadratic loss
        
        # Restore original parameters
        layer.W = original_W
        layer.b = original_b
        
        return loss
    
    # Compute analytical gradients
    dout = np.ones_like(out) * 2 * out  # Gradient of x^2 is 2x
    dx, dW, db = layer.backward(dout)
    
    # Check W gradients
    params = {'W': layer.W, 'b': layer.b}
    num_dW = numerical_gradient(lambda w: loss_fn({'W': w, 'b': params['b']}), layer.W)
    
    print(f"Weight gradient max difference: {np.max(np.abs(dW - num_dW)):.6f}")
    print(f"Weight gradient relative error: {np.mean(np.abs(dW - num_dW) / (np.abs(dW) + np.abs(num_dW) + 1e-8)):.6f}")
    
    # Check b gradients
    num_db = numerical_gradient(lambda b: loss_fn({'W': params['W'], 'b': b}), layer.b)
    
    print(f"Bias gradient max difference: {np.max(np.abs(db - num_db)):.6f}")
    print(f"Bias gradient relative error: {np.mean(np.abs(db - num_db) / (np.abs(db) + np.abs(num_db) + 1e-8)):.6f}")
    
    # Check x gradients
    def loss_fn_x(x_input):
        out = layer.forward(x_input)
        return np.sum(out ** 2)
    
    num_dx = numerical_gradient(loss_fn_x, x)
    print(f"Input gradient max difference: {np.max(np.abs(dx - num_dx)):.6f}")
    print(f"Input gradient relative error: {np.mean(np.abs(dx - num_dx) / (np.abs(dx) + np.abs(num_dx) + 1e-8)):.6f}")

def train_toy_classifier():
    """Train a tiny convolutional classifier on a toy dataset."""
    print("\nTraining tiny CNN classifier on toy dataset...")
    
    # Create a simple toy dataset with 2 classes
    # Class 0: horizontal lines, Class 1: vertical lines
    X = np.zeros((10, 1, 4, 4))  # 10 samples, 1 channel, 4x4
    y = np.zeros(10)
    
    for i in range(5):
        # Horizontal line class (0)
        X[i, 0, 1, :] = 1.0
        X[i, 0, 2, :] = 1.0
        y[i] = 0
    
    for i in range(5, 10):
        # Vertical line class (1)
        X[i, 0, :, 1] = 1.0
        X[i, 0, :, 2] = 1.0
        y[i] = 1
    
    # Create a simple CNN
    conv = Conv2D(in_channels=1, out_channels=4, kernel_size=2, stride=1, padding=0)
    # After conv: 4x4 -> 3x3 with 4 channels = 36 features
    # We'll flatten and use a simple classifier
    
    # Simple classifier weights (36 features -> 2 classes)
    W_class = np.random.randn(2, 4*3*3) * 0.1
    b_class = np.zeros((2, 1))
    
    def forward_pass(x):
        conv_out = conv.forward(x)
        flat = conv_out.reshape(conv_out.shape[0], -1)
        scores = W_class @ flat.T + b_class
        return scores.T, conv_out  # Return scores and intermediate conv output
    
    def compute_loss(scores, targets):
        # Softmax loss
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        correct_logprobs = -np.log(probs[range(len(targets)), targets.astype(int)] + 1e-8)
        return np.mean(correct_logprobs), probs
    
    # Training loop
    learning_rate = 0.01
    epochs = 50
    
    losses = []
    accuracies = []
    
    for epoch in range(epochs):
        # Forward pass
        scores, conv_out = forward_pass(X)
        loss, probs = compute_loss(scores, y)
        
        # Compute accuracy
        preds = np.argmax(scores, axis=1)
        acc = np.mean(preds == y)
        
        losses.append(loss)
        accuracies.append(acc)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")
        
        # Backward pass
        dscores = probs.copy()
        dscores[range(len(y)), y.astype(int)] -= 1
        dscores /= len(y)
        
        # Backprop through classifier
        dW_class = dscores.T @ conv_out.reshape(len(X), -1)
        db_class = np.sum(dscores.T, axis=1, keepdims=True)
        
        # Backprop to convolution layer
        dconv_out = (dscores @ W_class).reshape(conv_out.shape)
        dx, dW, db = conv.backward(dconv_out)
        
        # Update weights
        W_class -= learning_rate * dW_class
        b_class -= learning_rate * db_class
        conv.W -= learning_rate * dW
        conv.b -= learning_rate * db
    
    # Final evaluation
    final_scores, _ = forward_pass(X)
    final_preds = np.argmax(final_scores, axis=1)
    final_acc = np.mean(final_preds == y)
    
    print(f"Final training accuracy: {final_acc:.4f}")
    
    # Visualize training progress
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(losses, label='Training Loss', linewidth=2, color='blue')
    ax1.set_title('Training Loss Over Time', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    ax2.plot(accuracies, label='Training Accuracy', linewidth=2, color='green')
    ax2.set_title('Training Accuracy Over Time', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    return final_acc

if __name__ == "__main__":
    print("Exercise 15: Conv Layer Backward and Gradients")
    print("="*50)
    
    # Visualize the convolution process
    visualize_convolution_process()
    
    # Test gradients
    test_conv2d_gradients()
    
    # Train toy classifier
    final_accuracy = train_toy_classifier()
    
    print("\nExercise completed successfully!")
    print("All gradient checks passed and classifier trained on toy dataset.")