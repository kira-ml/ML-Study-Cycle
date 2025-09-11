# Author: kira-ml (GitHub)
"""
NumPy Broadcasting Operations for Machine Learning

This module demonstrates fundamental NumPy operations that are essential for 
understanding broadcasting in machine learning contexts. It includes implementations
of common tensor operations such as dimension expansion, tiling, axis permutation,
and fused scale-shift operations.

The implementations contrast explicit loop-based approaches with vectorized 
NumPy operations to illustrate performance differences and best practices
in numerical computing for ML applications.

Key Concepts Demonstrated:
- Broadcasting rules and their applications
- Tensor shape manipulation for ML pipelines
- Performance implications of vectorization
- Common patterns in deep learning frameworks
"""

import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from matplotlib.patches import Rectangle
import matplotlib.patches as patches


def unsqueeze_to(tensor: np.ndarray, target_ndim: int) -> np.ndarray:
    """
    Expand tensor dimensions to match target number of dimensions.
    
    In ML, we often need to add dimensions to enable broadcasting between
    tensors of different ranks. This is commonly used when combining
    batch dimensions with parameter tensors.
    
    Args:
        tensor: Input tensor to expand
        target_ndim: Desired number of dimensions
        
    Returns:
        np.ndarray: Tensor with expanded dimensions
    """
    while tensor.ndim < target_ndim:
        tensor = np.expand_dims(tensor, axis=0)
    return tensor


def tile_like(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Replicate source array to match the shape of reference array.
    
    Used in ML for broadcasting operations where we need to repeat
    parameters across batch dimensions or feature channels.
    
    Args:
        source: Array to replicate
        reference: Target shape reference
        
    Returns:
        np.ndarray: Replicated array matching reference shape
        
    Raises:
        ValueError: If source and reference have different dimensions
    """
    if source.ndim != reference.ndim:
        raise ValueError(f"Source and reference must have same number of dimensions. "
                         f"Got {source.ndim} and {reference.ndim}")
    
    reps = tuple(ref_dim // src_dim for ref_dim, src_dim in zip(reference.shape, source.shape))
    return np.tile(source, reps)


def fused_scale_shift(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Perform element-wise fused scaling and shifting operation: a * b + c.
    
    This pattern is fundamental in ML - it's the core computation in batch
    normalization, layer normalization, and many activation functions.
    NumPy's vectorization makes this extremely efficient.
    
    Args:
        a: Input tensor
        b: Scale factors
        c: Shift values
        
    Returns:
        np.ndarray: Scaled and shifted result
    """
    return a * b + c


def permute_axes(tensor: np.ndarray, order: tuple) -> np.ndarray:
    """
    Reorder tensor axes according to the specified permutation.
    
    Essential for aligning tensor dimensions when connecting different
    layers in neural networks or preparing data for specific operations.
    
    Args:
        tensor: Input tensor
        order: New axis order (e.g., (2, 0, 1) to move last axis to front)
        
    Returns:
        np.ndarray: Tensor with reordered axes
    """
    return np.transpose(tensor, axes=order)


def loop_multiply_add(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Multiply and add using explicit loops (educational implementation).
    
    This demonstrates what happens under the hood but is inefficient.
    Included for pedagogical purposes to show why vectorization matters.
    
    Args:
        a: Input tensor (2D)
        b: Scale factors (1D)
        c: Shift values (1D)
        
    Returns:
        np.ndarray: Result of a * b + c computed with loops
    """
    result = np.zeros_like(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            result[i, j] = a[i, j] * b[j] + c[j]
    return result


def vectorized_multiply_add(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Multiply and add using vectorized NumPy operations (production implementation).
    
    Leverages NumPy's broadcasting to perform the same operation as loop_multiply_add
    but much more efficiently. This is how you'd implement it in production ML code.
    
    Args:
        a: Input tensor (2D)
        b: Scale factors (1D)
        c: Shift values (1D)
        
    Returns:
        np.ndarray: Result of a * b + c computed with vectorization
    """
    return a * b + c


# === VISUALIZATION FUNCTIONS ===

def visualize_broadcasting_concept():
    """Visualize how broadcasting works conceptually."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Original arrays
    a = np.array([[1, 2, 3],
                  [4, 5, 6]])  # Shape (2, 3)
    b = np.array([10, 20, 30])   # Shape (3,)
    
    # Visualize array a
    im1 = axes[0].imshow(a, cmap='Blues', aspect='auto')
    axes[0].set_title('Array A: Shape (2, 3)')
    axes[0].set_xlabel('Columns')
    axes[0].set_ylabel('Rows')
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            axes[0].text(j, i, str(a[i, j]), ha='center', va='center')
    
    # Visualize array b
    b_reshaped = b.reshape(1, -1)
    im2 = axes[1].imshow(b_reshaped, cmap='Reds', aspect='auto')
    axes[1].set_title('Array B: Shape (3,) → Broadcasted')
    axes[1].set_xlabel('Columns')
    axes[1].set_ylabel('Broadcast Dimension')
    for j in range(b.shape[0]):
        axes[1].text(j, 0, str(b[j]), ha='center', va='center')
    
    # Visualize result
    result = a + b
    im3 = axes[2].imshow(result, cmap='Greens', aspect='auto')
    axes[2].set_title('Result A + B: Shape (2, 3)')
    axes[2].set_xlabel('Columns')
    axes[2].set_ylabel('Rows')
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            axes[2].text(j, i, str(result[i, j]), ha='center', va='center')
    
    plt.tight_layout()
    plt.suptitle('Broadcasting Concept: (2,3) + (3,) → (2,3)', y=1.05)
    plt.show()


def visualize_unsqueeze_operation():
    """Visualize dimension expansion (unsqueeze) operation."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    
    # Original 1D array
    original = np.array([1, 2, 3])
    axes[0].bar(range(len(original)), original, color='blue', alpha=0.7)
    axes[0].set_title(f'Original: Shape {original.shape}')
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Value')
    
    # After one unsqueeze
    unsqueezed1 = np.expand_dims(original, axis=0)
    axes[1].imshow(unsqueezed1, cmap='Blues', aspect='auto')
    axes[1].set_title(f'After 1 unsqueeze: Shape {unsqueezed1.shape}')
    for j in range(unsqueezed1.shape[1]):
        axes[1].text(j, 0, str(unsqueezed1[0, j]), ha='center', va='center')
    
    # After two unsqueezes
    unsqueezed2 = np.expand_dims(np.expand_dims(original, axis=0), axis=0)
    axes[2].imshow(unsqueezed2, cmap='Blues', aspect='auto')
    axes[2].set_title(f'After 2 unsqueezes: Shape {unsqueezed2.shape}')
    for j in range(unsqueezed2.shape[2]):
        axes[2].text(j, 0, str(unsqueezed2[0, 0, j]), ha='center', va='center')
    
    plt.tight_layout()
    plt.suptitle('Unsqueeze Operation (Adding Dimensions)', y=1.05)
    plt.show()


def visualize_tiling_operation():
    """Visualize tiling operation."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Source array (2, 1)
    source = np.array([[1], [2]])
    im1 = axes[0].imshow(source, cmap='Reds', aspect='auto')
    axes[0].set_title(f'Source: Shape {source.shape}')
    for i in range(source.shape[0]):
        axes[0].text(0, i, str(source[i, 0]), ha='center', va='center')
    
    # Reference array (2, 3)
    reference = np.array([[0, 0, 0], [0, 0, 0]])  # Just for shape reference
    im2 = axes[1].imshow(reference, cmap='Greys', alpha=0.3, aspect='auto')
    axes[1].set_title(f'Reference Shape: {reference.shape}')
    axes[1].text(1, 0.5, 'Target\nShape', ha='center', va='center', transform=axes[1].transAxes)
    
    # Tiled result
    result = tile_like(source, np.ones((2, 3)))
    im3 = axes[2].imshow(result, cmap='Greens', aspect='auto')
    axes[2].set_title(f'Tiled Result: Shape {result.shape}')
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            axes[2].text(j, i, str(result[i, j]), ha='center', va='center')
    
    plt.tight_layout()
    plt.suptitle('Tiling Operation: Replicating Arrays', y=1.05)
    plt.show()


def visualize_axis_permutation():
    """Visualize axis permutation operation."""
    fig = plt.figure(figsize=(15, 5))
    
    # Original tensor (2, 3, 4)
    original = np.random.randint(1, 10, (2, 3, 4))
    
    # Create 3D visualization
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.voxels(np.ones_like(original), facecolors='blue', alpha=0.3)
    ax1.set_title(f'Original: Shape {original.shape}\n(2 batches, 3 rows, 4 cols)')
    ax1.set_xlabel('Cols (dim 2)')
    ax1.set_ylabel('Rows (dim 1)')
    ax1.set_zlabel('Batches (dim 0)')
    
    # Permuted tensor (4, 2, 3)
    permuted = permute_axes(original, (2, 0, 1))
    
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.voxels(np.ones_like(permuted), facecolors='red', alpha=0.3)
    ax2.set_title(f'Permuted: Shape {permuted.shape}\n(4 cols, 2 batches, 3 rows)')
    ax2.set_xlabel('Rows (dim 2)')
    ax2.set_ylabel('Batches (dim 1)')
    ax2.set_zlabel('Cols (dim 0)')
    
    # Show axis mapping
    ax3 = fig.add_subplot(133)
    mapping_data = np.array([
        [0, 2],  # dim 0 → dim 2
        [1, 0],  # dim 1 → dim 0
        [2, 1]   # dim 2 → dim 1
    ])
    ax3.scatter(mapping_data[:, 0], mapping_data[:, 1], s=100, c='green')
    for i, (src, dst) in enumerate(mapping_data):
        ax3.annotate(f'dim {src} → dim {dst}', (src, dst), xytext=(5, 5), 
                    textcoords='offset points')
    ax3.plot(mapping_data[:, 0], mapping_data[:, 1], 'g--', alpha=0.5)
    ax3.set_xlim(-0.5, 2.5)
    ax3.set_ylim(-0.5, 2.5)
    ax3.set_xlabel('Original Dimensions')
    ax3.set_ylabel('New Dimensions')
    ax3.set_title('Axis Permutation Mapping')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.suptitle('Axis Permutation Operation', y=1.05)
    plt.show()


def visualize_performance_comparison(loop_time, vec_time):
    """Visualize performance comparison between loop and vectorized approaches."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart comparison
    methods = ['Loops', 'Vectorized']
    times = [loop_time, vec_time]
    colors = ['red', 'green']
    
    bars = ax1.bar(methods, times, color=colors, alpha=0.7)
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Performance Comparison')
    ax1.set_yscale('log')
    
    # Add value labels on bars
    for bar, time_val in zip(bars, times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{time_val:.6f}s', ha='center', va='bottom')
    
    # Speedup visualization
    speedup = loop_time / vec_time
    ax2.bar(['Speedup'], [speedup], color='orange', alpha=0.7)
    ax2.set_ylabel('Times Faster')
    ax2.set_title(f'Vectorization Speedup: {speedup:.1f}x')
    ax2.text(0, speedup/2, f'{speedup:.1f}x', ha='center', va='center', 
             fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.suptitle('NumPy Vectorization Performance Benefits', y=1.02)
    plt.show()


def visualize_fused_scale_shift():
    """Visualize fused scale-shift operation."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Input data
    a = np.array([[1, 2, 3],
                  [4, 5, 6]])  # Shape (2, 3)
    b = np.array([10, 20, 30])   # Scale factors
    c = np.array([1, 1, 1])      # Shift values
    
    # Show input array
    im1 = axes[0, 0].imshow(a, cmap='Blues', aspect='auto')
    axes[0, 0].set_title('Input Array A')
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            axes[0, 0].text(j, i, str(a[i, j]), ha='center', va='center')
    
    # Show scale factors
    im2 = axes[0, 1].imshow(b.reshape(1, -1), cmap='Reds', aspect='auto')
    axes[0, 1].set_title('Scale Factors B')
    for j in range(b.shape[0]):
        axes[0, 1].text(j, 0, str(b[j]), ha='center', va='center')
    
    # Show shift values
    im3 = axes[1, 0].imshow(c.reshape(1, -1), cmap='Greens', aspect='auto')
    axes[1, 0].set_title('Shift Values C')
    for j in range(c.shape[0]):
        axes[1, 0].text(j, 0, str(c[j]), ha='center', va='center')
    
    # Show result
    result = fused_scale_shift(a, b, c)
    im4 = axes[1, 1].imshow(result, cmap='Purples', aspect='auto')
    axes[1, 1].set_title('Result: A * B + C')
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            axes[1, 1].text(j, i, str(result[i, j]), ha='center', va='center')
    
    plt.tight_layout()
    plt.suptitle('Fused Scale-Shift Operation: A * B + C', y=1.02)
    plt.show()


def main():
    """Demonstrate and benchmark NumPy broadcasting operations with visualizations."""
    
    print("=== Visualizing NumPy Broadcasting Concepts ===")
    
    # Show conceptual broadcasting
    visualize_broadcasting_concept()
    
    # Show unsqueeze operation
    visualize_unsqueeze_operation()
    
    # Show tiling operation
    visualize_tiling_operation()
    
    # Show axis permutation
    visualize_axis_permutation()
    
    # Show fused scale-shift
    visualize_fused_scale_shift()
    
    print("\n=== Performance Comparison: Loops vs Vectorization ===")
    
    # Typical ML batch size (N) and feature/channel count (C)
    N, C = 16, 32
    a = np.random.randn(N, C)  # Batch of samples
    b = np.random.randn(C)     # Per-channel scale parameters
    c = np.random.randn(C)     # Per-channel shift parameters

    # Benchmark loop-based implementation
    start = perf_counter()
    out1 = loop_multiply_add(a, b, c)
    loop_time = perf_counter() - start

    # Benchmark vectorized implementation
    start = perf_counter()
    out2 = vectorized_multiply_add(a, b, c)
    vec_time = perf_counter() - start

    print("Output equality check:", np.allclose(out1, out2))
    print(f"Loop implementation time: {loop_time:.6f}s")
    print(f"Vectorized implementation time: {vec_time:.6f}s")
    print(f"Performance improvement: {loop_time/vec_time:.1f}x faster")
    
    # Visualize performance comparison
    visualize_performance_comparison(loop_time, vec_time)
    
    # Demonstrate tensor manipulation functions
    print("\n=== Tensor Shape Manipulation Functions ===")
    
    # unsqueeze_to: Adding dimensions for broadcasting compatibility
    x = np.array([1, 2, 3])
    y = unsqueeze_to(x, 3)
    print(f"Original shape: {x.shape}")
    print(f"Unsqueezed to 3D: {y.shape}")
    
    # tile_like: Replicating arrays to match target shapes
    src = np.array([[1], [2]])                    # Shape: (2, 1)
    ref = np.array([[1, 2, 3], [4, 5, 6]])       # Shape: (2, 3)
    tiled = tile_like(src, ref)
    print(f"Source shape: {src.shape}")
    print(f"Reference shape: {ref.shape}")
    print(f"Tiled result shape: {tiled.shape}")
    print("Tiled array:\n", tiled)
    
    # fused_scale_shift: Common ML operation pattern
    a = np.array([[1, 2], [3, 4]])
    b = np.array([10, 20])  # Scale factors
    c = np.array([1, 1])    # Shift values
    result = fused_scale_shift(a, b, c)
    print("Fused scale-shift result:\n", result)
    
    # permute_axes: Reordering tensor dimensions
    x = np.random.randn(2, 3, 4)  # Example: (batch, height, width)
    y = permute_axes(x, (2, 0, 1)) # Reorder to (width, batch, height)
    print(f"Original shape: {x.shape}")
    print(f"Permuted shape: {y.shape}")
    
    # Additional verification of basic operations
    print("\n=== Basic NumPy Operations Verification ===")
    dot_product = np.dot(np.array([1, 2, 3]), np.array([4, 5, 6]))
    print(f"Dot product test: {dot_product}")


if __name__ == "__main__":
    main()