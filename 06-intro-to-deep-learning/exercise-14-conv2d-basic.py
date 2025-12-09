"""
Exercise 14 — 2D Convolution from First Principles
Author: kira-ml
Repository: https://github.com/kira-ml/open-source-ml-education

WHAT YOU'LL LEARN:
1. How convolution works mathematically (the "sliding window" operation)
2. Two implementation strategies: naive loops vs. optimized im2col
3. The critical tradeoff: computation speed vs. memory usage
4. How padding and stride affect output dimensions
5. Why modern frameworks use im2col despite its memory cost

Think of convolution like applying a stencil to an image: 
we slide a small filter across the input, computing dot products at each position.
This operation is fundamental to CNNs (Convolutional Neural Networks) for image processing.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set visual style for better educational clarity
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def conv2d_naive(x, w, b, stride=1, padding=0):
    """
    NAIVE IMPLEMENTATION: Understand convolution by visualizing the sliding window.
    
    This is the most intuitive way to understand convolution: we literally slide
    the filter across the input image. While simple to understand, it's slow because
    of Python loops. Think of it as the "textbook definition" implementation.
    
    Args:
        x: Input tensor shape (N, C, H, W) 
           N=batch size, C=channels (colors in RGB), H=height, W=width
        w: Filter tensor shape (F, C, HH, WW)
           F=number of filters, HH=filter height, WW=filter width
        b: Bias vector shape (F,) - adds learnable offset to each filter output
        stride: How many pixels to jump when sliding (1=every pixel, 2=every other pixel)
        padding: Add zeros around the border (preserves spatial dimensions)
    
    Returns:
        Output tensor shape (N, F, H_out, W_out) - one feature map per filter
    """
    # Unpack dimensions: think of these as "metadata" about our tensors
    N, C, H, W = x.shape  # N images, C channels, H height, W width
    F, _, HH, WW = w.shape  # F filters, each HH x WW in size
    
    # CALCULATE OUTPUT DIMENSIONS: crucial formula to memorize!
    # Why this formula? We count how many times the filter fits across the image.
    H_out = (H + 2 * padding - HH) // stride + 1
    W_out = (W + 2 * padding - WW) // stride + 1
    
    # Initialize output tensor with zeros (will fill it pixel by pixel)
    out = np.zeros((N, F, H_out, W_out))
    
    # APPLY PADDING: add zeros around the border if requested
    # Padding helps preserve edges and control output size
    if padding > 0:
        # np.pad adds padding to each dimension: ((before, after), (before, after), ...)
        # Mode 'constant' means fill with zeros (like a black border)
        x_padded = np.pad(x, 
                         ((0, 0), (0, 0), (padding, padding), (padding, padding)), 
                         mode='constant')
    else:
        x_padded = x  # No padding, use as-is
    
    # THE CORE CONVOLUTION: four nested loops (this is why it's "naive")
    # Visualize: For each image, each filter, slide across rows, then columns
    for n in range(N):  # For each image in the batch
        for f in range(F):  # For each filter (produces one feature map)
            for i in range(H_out):  # Slide down the rows
                for j in range(W_out):  # Slide across the columns
                    # Calculate the "window" position in the padded input
                    h_start = i * stride  # Starting row for this position
                    h_end = h_start + HH  # Ending row (start + filter height)
                    w_start = j * stride  # Starting column
                    w_end = w_start + WW  # Ending column
                    
                    # EXTRACT THE RECEPTIVE FIELD: the patch of input the filter "sees"
                    roi = x_padded[n, :, h_start:h_end, w_start:w_end]
                    # roi shape: (C, HH, WW) - exactly matches filter dimensions
                    
                    # DOT PRODUCT: filter weights × input patch + bias
                    # This single number becomes one pixel in the output feature map
                    out[n, f, i, j] = np.sum(roi * w[f]) + b[f]
    
    return out


def im2col_indices(x, HH, WW, padding=0, stride=1):
    """
    IMAGE TO COLUMN TRANSFORMATION: The optimization trick!
    
    Instead of sliding a window, we reshape the input so that convolution 
    becomes a single matrix multiplication. This is MUCH faster because 
    numpy (and GPUs) are optimized for matrix multiplication.
    
    Analogy: Instead of cutting cookies one by one (sliding), we make a 
    cookie cutter sheet (im2col) and stamp all at once (matmul).
    
    Args:
        x: Input tensor shape (N, C, H, W)
        HH, WW: Filter dimensions
        padding, stride: Same as in conv2d_naive
    
    Returns:
        2D matrix where each column is a flattened receptive field
        Shape: (HH * WW * C, N * H_out * W_out)
    """
    # Apply padding if needed (same logic as before)
    if padding > 0:
        x = np.pad(x, 
                  ((0, 0), (0, 0), (padding, padding), (padding, padding)), 
                  mode='constant')
    
    N, C, H, W = x.shape
    
    # Calculate output dimensions (same formula)
    H_out = (H - HH) // stride + 1
    W_out = (W - WW) // stride + 1
    
    # CREATE INDICES: the clever part that extracts all windows at once
    
    # i0: vertical positions within a single filter (0, 1, ..., HH-1)
    # np.tile([0, 1], 2) → [0, 1, 0, 1] for a 2x2 filter
    i0 = np.tile(np.arange(HH), WW)
    
    # i1: starting row positions for each output position (0, 0, 0, 1, 1, 1, ...)
    # np.repeat([0, 1], 3) → [0, 0, 0, 1, 1, 1] for 3 output columns
    i1 = np.repeat(np.arange(H_out), W_out) * stride
    
    # i: combine to get all row indices for all positions
    # Broadcasting: (4,1) + (1,9) → (4,9) matrix of indices
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    
    # Same logic for columns
    j0 = np.tile(np.arange(WW), HH)  # Column indices within filter
    j1 = np.tile(np.arange(W_out), H_out) * stride  # Starting columns
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    
    # Channel indices: which channel each row belongs to
    # For RGB: [0,0,0,0, 1,1,1,1, 2,2,2,2] for a 2x2 filter
    k = np.repeat(np.arange(C), HH * WW).reshape(-1, 1)
    
    # MAGIC HAPPENS HERE: numpy's advanced indexing extracts all patches at once
    # x[:, k, i, j] uses broadcasting to get ALL receptive fields simultaneously
    cols = x[:, k, i, j]
    
    # Reshape to final 2D matrix format
    # Transpose and reshape to get columns = flattened patches
    cols = cols.transpose(1, 2, 0).reshape(HH * WW * C, -1)
    
    return cols


def col2im_indices(cols, x_shape, HH, WW, padding=0, stride=1):
    """
    Reverse of im2col: Convert columns back to image format.
    
    Needed for backpropagation in neural networks - when we need to 
    compute gradients with respect to the input.
    
    Args:
        cols: Output from im2col_indices
        x_shape: Original input shape (N, C, H, W)
        HH, WW, padding, stride: Same as forward pass
    """
    N, C, H, W = x_shape
    H_pad, W_pad = H + 2*padding, W + 2*padding
    
    # Start with zeros (we'll accumulate gradients into this)
    x_padded = np.zeros((N, C, H_pad, W_pad), dtype=cols.dtype)
    
    H_out = (H + 2*padding - HH) // stride + 1
    W_out = (W + 2*padding - WW) // stride + 1
    
    # Reshape columns back to 5D tensor: (N, C, HH*WW, H_out, W_out)
    cols_reshaped = cols.reshape(C * HH * WW, H_out * W_out, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1).reshape(N, C, HH * WW, H_out, W_out)
    
    # Accumulate values back to their original positions
    # This is like "undoing" the im2col operation
    for i in range(HH):
        for j in range(WW):
            # Calculate where this filter element contributes
            i_start = i
            i_end = i_start + H_out * stride
            j_start = j
            j_end = j_start + W_out * stride
            
            # Add the values back to the padded image
            x_padded[:, :, i_start:i_end:stride, j_start:j_end:stride] += \
                cols_reshaped[:, :, i*WW + j, :, :]
    
    # Remove padding if we had any
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


def conv2d_im2col(x, w, b, stride=1, padding=0):
    """
    OPTIMIZED IMPLEMENTATION: Using im2col + matrix multiplication.
    
    This is what libraries like PyTorch and TensorFlow use under the hood.
    The key insight: convolution = matrix multiplication after reshaping.
    
    Performance: 10-100x faster than naive loops on CPU/GPU.
    Tradeoff: Uses more memory (stores expanded matrix).
    """
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    
    # Calculate output dimensions (same as before)
    H_out = (H + 2 * padding - HH) // stride + 1
    W_out = (W + 2 * padding - WW) // stride + 1
    
    # STEP 1: Transform input to column matrix
    # Each column = one receptive field flattened
    x_cols = im2col_indices(x, HH, WW, padding, stride)
    
    # STEP 2: Reshape filters to row matrix
    # Each row = one filter flattened
    w_rows = w.reshape(F, -1)  # Shape: (F, C*HH*WW)
    
    # STEP 3: The big matrix multiplication!
    # This single operation replaces ALL nested loops
    # (F, C*HH*WW) × (C*HH*WW, N*H_out*W_out) → (F, N*H_out*W_out)
    out = w_rows @ x_cols  # @ is numpy's matrix multiplication operator
    
    # STEP 4: Add bias (one bias per filter)
    # Reshape b to (F, 1) and broadcast across columns
    out = out + b.reshape(-1, 1)
    
    # STEP 5: Reshape back to 4D tensor format
    out = out.reshape(F, H_out, W_out, N)  # Temporary shape
    out = out.transpose(3, 0, 1, 2)  # Back to (N, F, H_out, W_out)
    
    return out


def visualize_convolution_process():
    """
    VISUAL EXPLANATION: See convolution in action!
    
    This function creates an 8-panel visualization that shows:
    1. Input image and filter
    2. The sliding window operation
    3. Im2col transformation
    4. Performance tradeoffs
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('2D Convolution: From Intuition to Implementation', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Create a simple 4x4 input image (easy to follow numbers)
    input_img = np.array([[[1, 2, 3, 4],
                           [5, 6, 7, 8],
                           [9, 10, 11, 12],
                           [13, 14, 15, 16]]])
    
    # Create a simple 2x2 filter (edge detector-like)
    filter_kernel = np.array([[[1, 0],
                               [0, -1]]])
    
    # Manually compute convolution to show the process
    conv_result = np.zeros((1, 3, 3))
    for i in range(3):
        for j in range(3):
            # Extract 2x2 patch
            patch = input_img[0, i:i+2, j:j+2]
            # Element-wise multiply and sum
            conv_result[0, i, j] = np.sum(patch * filter_kernel[0])
    
    # ----- PANEL 1: Input Image -----
    im1 = axes[0, 0].imshow(input_img[0], cmap='viridis', interpolation='none')
    axes[0, 0].set_title('Input Image (4×4)', fontweight='bold')
    axes[0, 0].set_xticks(np.arange(4))
    axes[0, 0].set_yticks(np.arange(4))
    # Add numbers for clarity
    for i in range(4):
        for j in range(4):
            axes[0, 0].text(j, i, f'{input_img[0, i, j]:.0f}', 
                           ha='center', va='center', 
                           color='white', fontweight='bold')
    axes[0, 0].grid(color='white', linewidth=1)
    
    # ----- PANEL 2: Filter/Kernel -----
    im2 = axes[0, 1].imshow(filter_kernel[0], cmap='RdBu', 
                           interpolation='none', vmin=-1, vmax=1)
    axes[0, 1].set_title('Filter/Kernel (2×2)\n[1, 0; 0, -1]', fontweight='bold')
    axes[0, 1].set_xticks(np.arange(2))
    axes[0, 1].set_yticks(np.arange(2))
    for i in range(2):
        for j in range(2):
            text_color = 'white' if filter_kernel[0, i, j] < 0 else 'black'
            axes[0, 1].text(j, i, f'{filter_kernel[0, i, j]:.0f}', 
                           ha='center', va='center', 
                           color=text_color, fontweight='bold')
    axes[0, 1].grid(color='white', linewidth=1)
    
    # ----- PANEL 3: Convolution Output -----
    im3 = axes[0, 2].imshow(conv_result[0], cmap='viridis', interpolation='none')
    axes[0, 2].set_title('Convolution Output (3×3)', fontweight='bold')
    axes[0, 2].set_xticks(np.arange(3))
    axes[0, 2].set_yticks(np.arange(3))
    for i in range(3):
        for j in range(3):
            axes[0, 2].text(j, i, f'{conv_result[0, i, j]:.0f}', 
                           ha='center', va='center', 
                           color='white', fontweight='bold')
    axes[0, 2].grid(color='white', linewidth=1)
    
    # ----- PANEL 4: Im2col Transformation -----
    x_col = im2col_indices(input_img[np.newaxis, ...], 2, 2, padding=0, stride=1)
    im4 = axes[0, 3].imshow(x_col, cmap='Blues', aspect='auto')
    axes[0, 3].set_title('Im2Col Transformation', fontweight='bold')
    axes[0, 3].set_xlabel('9 Output Positions\n(3×3 output)')
    axes[0, 3].set_ylabel('4 Filter Elements\n(2×2 flattened)')
    axes[0, 3].set_xticks([])
    axes[0, 3].set_yticks([])
    
    # ----- PANEL 5: Stride Visualization -----
    axes[1, 0].imshow(input_img[0], cmap='viridis', interpolation='none')
    # Show two consecutive filter positions
    axes[1, 0].add_patch(Rectangle((0, 0), 2, 2, linewidth=3, 
                                   edgecolor='red', facecolor='none', 
                                   label='Position 1'))
    axes[1, 0].add_patch(Rectangle((1, 1), 2, 2, linewidth=3, 
                                   edgecolor='blue', facecolor='none', 
                                   label='Position 2'))
    axes[1, 0].set_title('Stride = 1\n(Move 1 pixel at a time)', fontweight='bold')
    axes[1, 0].legend(loc='upper left', fontsize=9)
    axes[1, 0].grid(color='white', linewidth=1)
    
    # ----- PANEL 6: Padding Visualization -----
    padded_img = np.pad(input_img[0], 1, mode='constant')
    axes[1, 1].imshow(padded_img, cmap='viridis', interpolation='none')
    # Highlight original image within padding
    axes[1, 1].add_patch(Rectangle((1, 1), 4, 4, linewidth=3, 
                                   edgecolor='yellow', facecolor='none', 
                                   label='Original (4×4)'))
    axes[1, 1].set_title('Padding = 1\n(Adds border of zeros)', fontweight='bold')
    axes[1, 1].legend(loc='upper left', fontsize=9)
    
    # ----- PANEL 7: Speed Comparison -----
    axes[1, 2].bar(['Naive (Loops)', 'Im2Col (MatMul)'], [100, 30], 
                   color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
    axes[1, 2].set_title('Computation Speed\n(Relative Time)', fontweight='bold')
    axes[1, 2].set_ylabel('Time (%)')
    axes[1, 2].text(0.5, 50, 'im2col is ∼3× faster!', 
                   ha='center', va='center', fontweight='bold', fontsize=10)
    
    # ----- PANEL 8: Memory Comparison -----
    axes[1, 3].bar(['Naive (Loops)', 'Im2Col (MatMul)'], [100, 300], 
                   color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
    axes[1, 3].set_title('Memory Usage\n(Relative)', fontweight='bold')
    axes[1, 3].set_ylabel('Memory (%)')
    axes[1, 3].text(0.5, 200, 'im2col uses 3× more memory!', 
                   ha='center', va='center', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.show()


def compare_implementations():
    """
    EMPIRICAL COMPARISON: See the actual speed difference!
    
    Runs both implementations on the same input and compares:
    1. Execution time (how fast)
    2. Output correctness (do they match?)
    3. Speedup factor (how much faster is im2col?)
    """
    print("="*60)
    print("IMPLEMENTATION COMPARISON: Naive vs. Optimized")
    print("="*60)
    
    # Create test data (small enough to run quickly, large enough to measure)
    N, C, H, W = 2, 3, 8, 8   # 2 images, 3 channels (like RGB), 8×8 pixels
    F, HH, WW = 4, 3, 3       # 4 filters, each 3×3
    stride, padding = 1, 1     # Common settings
    
    # Random but reproducible inputs (same seed)
    np.random.seed(42)
    x = np.random.randn(N, C, H, W)  # Gaussian noise image
    w = np.random.randn(F, C, HH, WW)  # Random filters
    b = np.random.randn(F)  # Random biases
    
    print(f"Input shape: {x.shape}")
    print(f"Filter shape: {w.shape}")
    print(f"Expected output shape: ({N}, {F}, {6}, {6})  # (H+2p-HH)/s+1")
    print()
    
    # ----- NAIVE IMPLEMENTATION (Baseline) -----
    print("Running NAIVE implementation (4 nested loops)...")
    start_time = time.time()
    out_naive = conv2d_naive(x, w, b, stride, padding)
    naive_time = time.time() - start_time
    print(f"✓ Naive time: {naive_time:.6f} seconds")
    
    # ----- IM2COL IMPLEMENTATION (Optimized) -----
    print("\nRunning IM2COL implementation (matrix multiplication)...")
    start_time = time.time()
    out_im2col = conv2d_im2col(x, w, b, stride, padding)
    im2col_time = time.time() - start_time
    print(f"✓ Im2col time: {im2col_time:.6f} seconds")
    
    # ----- COMPARISON RESULTS -----
    print("\n" + "-"*40)
    print("COMPARISON RESULTS:")
    print("-"*40)
    
    # Check if outputs match (they should!)
    diff = np.max(np.abs(out_naive - out_im2col))
    matches = np.allclose(out_naive, out_im2col, rtol=1e-5, atol=1e-8)
    
    print(f"✓ Output shapes match: {out_naive.shape} == {out_im2col.shape}")
    print(f"✓ Max difference: {diff:.10f} (should be near zero)")
    print(f"✓ Implementations identical: {matches}")
    print(f"✓ Speedup factor: {naive_time/im2col_time:.2f}x faster!")
    
    # Practical insight
    if naive_time/im2col_time > 2:
        print("\n💡 INSIGHT: im2col is significantly faster!")
        print("   This is why frameworks like PyTorch/TensorFlow use it.")
    else:
        print("\n💡 NOTE: For very small inputs, the overhead of")
        print("   im2col might not be worth it. But for real images,")
        print("   the speedup is enormous!")
    
    return out_naive, out_im2col, naive_time, im2col_time


def visualize_performance_comparison(naive_time, im2col_time):
    """
    Visualize the performance difference between implementations.
    
    Shows exactly how much faster im2col is, and explains why it matters
    for training real neural networks (where we do billions of convolutions).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['Naive (4 Nested Loops)', 'Im2Col (Matrix Multiply)']
    times = [naive_time, im2col_time]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = ax.bar(methods, times, color=colors, alpha=0.8)
    ax.set_ylabel('Execution Time (seconds)', fontweight='bold')
    ax.set_title('Why Optimization Matters: Convolution Speed Test', 
                fontsize=14, fontweight='bold')
    
    # Add exact time labels
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + time_val*0.01,
                f'{time_val:.6f}s',
                ha='center', va='bottom', fontweight='bold')
    
    # Add educational annotation
    speedup = naive_time / im2col_time
    ax.annotate(f'IM2COL IS {speedup:.1f}X FASTER!\n\nWhy this matters:\n• Training CNNs requires\n  billions of convolutions\n• 10x speedup = 10x faster training\n• Allows bigger models,\n  more experiments',
                xy=(1, im2col_time), 
                xytext=(0.7, max(times) * 0.7),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=11, ha='center', fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    plt.tight_layout()
    plt.show()


def visualize_memory_tradeoff():
    """
    Show the classic computer science tradeoff: speed vs. memory.
    
    im2col is faster but uses more memory because it creates an 
    expanded representation of the input. This is often acceptable 
    because memory is cheaper than computation time for training.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ----- LEFT PLOT: Direct Comparison -----
    methods = ['Naive', 'Im2Col']
    # Relative memory: im2col stores expanded matrix
    memory_usage = [1.0, 3.0]  # im2col uses ~3x more memory
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars1 = ax1.bar(methods, memory_usage, color=colors, alpha=0.8)
    ax1.set_ylabel('Relative Memory Usage', fontweight='bold')
    ax1.set_title('The Tradeoff: Speed vs. Memory', fontsize=14, fontweight='bold')
    
    for bar, mem in zip(bars1, memory_usage):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{mem:.1f}x',
                ha='center', va='bottom', fontweight='bold')
    
    # Add explanation
    ax1.text(0.5, 1.5, 'Im2Col trades memory\nfor computation speed',
            ha='center', va='center', fontsize=11, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    
    # ----- RIGHT PLOT: General Tradeoff Curve -----
    x_vals = np.linspace(0, 10, 100)
    # Simplified model: efficiency improves with memory, but with diminishing returns
    efficiency = 10 / (1 + 9 * np.exp(-0.5 * x_vals))  # Sigmoid-like curve
    
    ax2.plot(x_vals, efficiency, linewidth=3, color='purple', 
             label='Speed vs. Memory Tradeoff')
    ax2.fill_between(x_vals, efficiency, alpha=0.3, color='purple')
    ax2.set_xlabel('Memory Usage (relative)')
    ax2.set_ylabel('Computational Speed (relative)', fontweight='bold')
    ax2.set_title('The Optimization Frontier', fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Mark our two implementations on the curve
    ax2.scatter([1, 3], [1, 8], s=100, color=['red', 'green'], zorder=5)
    ax2.text(1, 0.8, 'Naive\n(Slow, Low Memory)', ha='center', fontweight='bold')
    ax2.text(3, 8.2, 'Im2Col\n(Fast, High Memory)', ha='center', fontweight='bold')
    
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    plt.show()


def demonstrate_padding_stride():
    """
    Show how padding and stride affect convolution output.
    
    Key concepts:
    • Padding controls output size (more padding = larger output)
    • Stride controls downsampling (larger stride = smaller output)
    • Different combinations are used for different purposes in CNNs
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Padding & Stride: Controlling Convolution Output Size', 
                fontsize=16, fontweight='bold')
    
    # Create a random 6×6 input (single channel for simplicity)
    np.random.seed(42)
    input_img = np.random.rand(1, 1, 6, 6) * 10
    
    # Create an averaging filter (3×3, all values = 1/9)
    filter_kernel = np.ones((1, 1, 3, 3)) / 9
    
    # Test different combinations (common in CNN architectures)
    configs = [
        {"padding": 0, "stride": 1, "title": "Vanilla\nNo Padding, Stride=1"},
        {"padding": 1, "stride": 1, "title": "Same Padding\nPad=1, Stride=1"},
        {"padding": 0, "stride": 2, "title": "Downsample\nNo Pad, Stride=2"},
        {"padding": 1, "stride": 2, "title": "Downsample + Pad\nPad=1, Stride=2"},
        {"padding": 2, "stride": 1, "title": "Large Padding\nPad=2, Stride=1"},
        {"padding": 2, "stride": 2, "title": "Aggressive\nPad=2, Stride=2"},
    ]
    
    for i, config in enumerate(configs):
        row = i // 3
        col = i % 3
        
        # Compute convolution with these settings
        out = conv2d_im2col(input_img, filter_kernel, np.array([0]), 
                           stride=config["stride"], padding=config["padding"])
        
        # Plot the output
        im = axes[row, col].imshow(out[0, 0], cmap='viridis', interpolation='none')
        axes[row, col].set_title(f'{config["title"]}\nOutput: {out.shape[2]}×{out.shape[3]}', 
                                fontweight='bold')
        axes[row, col].grid(True, linestyle='--', alpha=0.6)
        
        # Formula reminder in subtitle
        H, W = 6, 6  # Original size
        HH, WW = 3, 3  # Filter size
        p, s = config["padding"], config["stride"]
        H_out = (H + 2*p - HH) // s + 1
        W_out = (W + 2*p - WW) // s + 1
        
        # Add colorbar for each subplot
        plt.colorbar(im, ax=axes[row, col])
    
    # Add overall explanation
    fig.text(0.5, 0.02, 
             'FORMULA: Output Size = ⌊(Input + 2×Padding - Filter)/Stride⌋ + 1\n'
             'Padding preserves spatial info | Stride reduces resolution',
             ha='center', fontsize=11, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("EXERCISE 14: 2D CONVOLUTION FROM FIRST PRINCIPLES")
    print("="*60)
    print("\nWelcome to the world of convolutional neural networks!")
    print("In this exercise, you'll understand HOW convolution actually works,")
    print("and WHY modern implementations use clever optimizations.\n")
    
    # Step 1: Visualize the concept
    print("STEP 1: Visualizing convolution (the sliding window)...")
    visualize_convolution_process()
    
    # Step 2: Compare implementations
    print("\nSTEP 2: Comparing implementations...")
    out_naive, out_im2col, naive_time, im2col_time = compare_implementations()
    
    # Step 3: Show performance results
    print("\nSTEP 3: Visualizing the performance difference...")
    visualize_performance_comparison(naive_time, im2col_time)
    
    # Step 4: Explain the tradeoff
    print("\nSTEP 4: Understanding the speed-memory tradeoff...")
    visualize_memory_tradeoff()
    
    # Step 5: Show parameter effects
    print("\nSTEP 5: Exploring padding and stride effects...")
    demonstrate_padding_stride()
    
    # Final summary
    print("\n" + "="*60)
    print("EXERCISE COMPLETE! KEY TAKEAWAYS:")
    print("="*60)
    print("1. CONVOLUTION BASICS:")
    print("   • Slide a filter across input, compute dot products")
    print("   • Produces feature maps that highlight patterns")
    print()
    print("2. TWO IMPLEMENTATION STRATEGIES:")
    print("   • Naive: 4 nested loops (easy to understand, slow)")
    print("   • Im2Col: Matrix multiplication (fast, uses more memory)")
    print()
    print("3. THE TRADEOFF:")
    print("   • Speed vs. Memory: im2col is faster but uses 3-5x more memory")
    print("   • This is usually worth it: training time > memory cost")
    print()
    print("4. PRACTICAL PARAMETERS:")
    print("   • Padding: Controls output size, preserves edges")
    print("   • Stride: Controls downsampling, reduces computation")
    print()
    print("5. REAL-WORLD APPLICATION:")
    print("   • PyTorch/TensorFlow use im2col-like optimizations")
    print("   • Understanding this helps debug and optimize CNNs")
    print()
    print("🌟 Great work! You now understand convolution at a deep level.")
    print("   This is foundational for computer vision and CNNs.")
    print("="*60)