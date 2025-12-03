# Author: kira-ml (GitHub)
"""
NumPy Broadcasting Operations for Machine Learning
==================================================

Welcome to this interactive tutorial on NumPy broadcasting! 
If you're learning machine learning, you've probably encountered 
error messages about "shape mismatch" or "broadcasting rules". 
This module will demystify those concepts through clear explanations
and visual examples.

Key Concepts You'll Learn:
• Broadcasting: How NumPy handles operations between arrays of different shapes
• Dimension Expansion: Adding new axes to make arrays compatible
• Vectorization: Why it's crucial for ML performance
• Tensor Manipulation: Essential operations for preparing data in deep learning

Think of broadcasting like making ingredients compatible for a recipe:
If you have 3 eggs and 1 cup of flour, you can't add them directly. 
But if you "broadcast" the 1 cup to match the 3 eggs dimensionally,
you can combine them properly!
"""

import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from matplotlib.patches import Rectangle
import matplotlib.patches as patches


def unsqueeze_to(tensor: np.ndarray, target_ndim: int) -> np.ndarray:
    """
    Add dimensions to a tensor until it reaches the target number of dimensions.
    
    ML CONTEXT: In neural networks, we often need to add batch dimensions or
    channel dimensions to make tensors compatible. For example, a single image
    might have shape (height, width, channels), but when processing a batch,
    we need shape (batch_size, height, width, channels).
    
    ANALOGY: Think of adding empty boxes around your data to make it fit
    a standardized container shape.
    
    Args:
        tensor: Your input array (could be 1D, 2D, 3D, etc.)
        target_ndim: How many dimensions you want it to have
        
    Returns:
        Expanded tensor with the specified number of dimensions
    """
    # Keep adding dimensions at the beginning until we reach target
    while tensor.ndim < target_ndim:
        # np.expand_dims adds a new dimension at position 0 (the front)
        # This is like adding a new outer box around your data
        tensor = np.expand_dims(tensor, axis=0)
    return tensor


def tile_like(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Replicate (tile) an array to match the shape of another array.
    
    ML CONTEXT: Used when you need to repeat weights or parameters across
    different dimensions. For example, repeating bias terms across all
    batch examples, or repeating channel-wise normalization parameters.
    
    ANALOGY: Making copies of a small stamp to fill a larger sheet of paper.
    
    Args:
        source: The array you want to replicate
        reference: The array whose shape you want to match
        
    Returns:
        A tiled version of source that has the same shape as reference
    """
    # First check: source and reference must have same number of dimensions
    # This prevents confusing errors later
    if source.ndim != reference.ndim:
        raise ValueError(
            f"Dimension mismatch! Source has {source.ndim} dimensions, "
            f"but reference has {reference.ndim}. "
            "They must have the same number of dimensions for tiling."
        )
    
    # Calculate how many times to repeat along each dimension
    # For each dimension: reference_size / source_size = how many copies needed
    reps = tuple(
        ref_dim // src_dim 
        for ref_dim, src_dim in zip(reference.shape, source.shape)
    )
    
    # np.tile creates the repeated copies efficiently
    return np.tile(source, reps)


def fused_scale_shift(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Compute a * b + c in a single, efficient operation.
    
    ML CONTEXT: This pattern appears EVERYWHERE in deep learning:
    • Batch Normalization: (x - mean) * gamma / std + beta
    • Layer Normalization: Similar pattern
    • Attention mechanisms: q * k + bias
    • Activation functions with learnable parameters
    
    PERFORMANCE TIP: Doing this as one operation (fused) is faster than
    doing a*b first, then adding c, because it reduces memory access.
    
    Args:
        a: Input tensor (your data)
        b: Scale factors (multipliers)
        c: Shift values (additive terms)
        
    Returns:
        The scaled and shifted result
    """
    # The beauty of NumPy: This one line does ALL the broadcasting automatically!
    # If shapes don't match, NumPy tries to make them compatible using
    # broadcasting rules (explained in visualizations)
    return a * b + c


def permute_axes(tensor: np.ndarray, order: tuple) -> np.ndarray:
    """
    Reorder the dimensions (axes) of a tensor.
    
    ML CONTEXT: Different frameworks expect different dimension orders!
    • PyTorch: Usually (batch, channels, height, width)
    • TensorFlow: Usually (batch, height, width, channels)
    • When saving/loading models between frameworks, you need to permute axes.
    
    ANALOGY: Rearranging the shelves in a bookcase to organize books differently.
    
    Args:
        tensor: Your multi-dimensional array
        order: The new order of dimensions. Example: (2, 0, 1) means:
               - New dimension 0 = old dimension 2
               - New dimension 1 = old dimension 0
               - New dimension 2 = old dimension 1
        
    Returns:
        Tensor with rearranged dimensions
    """
    # np.transpose is the workhorse for dimension reordering
    # It creates a new "view" of the data with different dimension order
    # (Not a copy, so it's memory efficient!)
    return np.transpose(tensor, axes=order)


def loop_multiply_add(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Compute a * b + c using explicit Python loops (SLOW version).
    
    EDUCATIONAL PURPOSE: This shows you EXACTLY what happens when you
    do element-wise operations. But NEVER use this in real ML code!
    
    WHY IT'S SLOW:
    1. Python loops are slow (interpreted language overhead)
    2. Individual array element access is inefficient
    3. No CPU cache optimization
    
    Args:
        a: 2D input array
        b: 1D scale factors (will be broadcast across rows)
        c: 1D shift values (will be broadcast across rows)
        
    Returns:
        Same result as a * b + c, but computed inefficiently
    """
    # Create empty result array with same shape as input
    result = np.zeros_like(a)
    
    # Outer loop: iterate through rows (first dimension)
    for i in range(a.shape[0]):
        # Inner loop: iterate through columns (second dimension)
        for j in range(a.shape[1]):
            # For each element: multiply by scale factor, add shift
            # Note: b[j] and c[j] use same j index - this is broadcasting!
            result[i, j] = a[i, j] * b[j] + c[j]
    
    return result


def vectorized_multiply_add(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Compute a * b + c using NumPy vectorization (FAST version).
    
    PRODUCTION CODE: This is how you should ALWAYS write ML operations.
    
    WHY IT'S FAST:
    1. Operations happen in C/C++ (compiled, not interpreted)
    2. Uses CPU SIMD instructions (processes multiple elements at once)
    3. Optimized memory access patterns
    4. Automatically handles broadcasting
    
    Args:
        a: 2D input array
        b: 1D scale factors
        c: 1D shift values
        
    Returns:
        Result of a * b + c, computed efficiently
    """
    # Magic happens here: NumPy automatically broadcasts b and c
    # to match the shape of a, then does element-wise operations
    # This one line replaces ALL the loops above!
    return a * b + c


# === VISUALIZATION FUNCTIONS ===
# These help you SEE what broadcasting looks like

def visualize_broadcasting_concept():
    """
    Show how broadcasting works when adding arrays of different shapes.
    
    KEY INSIGHT: Broadcasting automatically expands smaller arrays
    to match larger ones, WITHOUT making copies in memory!
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Create example arrays
    a = np.array([[1, 2, 3],
                  [4, 5, 6]])  # Shape (2, 3) - like 2 samples, 3 features
    b = np.array([10, 20, 30])   # Shape (3,) - like 3 bias terms
    
    # --- Plot 1: Original array A ---
    axes[0].imshow(a, cmap='Blues', aspect='auto')
    axes[0].set_title('Array A: Shape (2, 3)\n(2 rows, 3 columns)')
    axes[0].set_xlabel('Feature dimension')
    axes[0].set_ylabel('Sample dimension')
    # Add text labels to each cell
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            axes[0].text(j, i, str(a[i, j]), ha='center', va='center', 
                        fontweight='bold')
    
    # --- Plot 2: Array B before/after broadcasting ---
    # Reshape b to 2D for visualization
    b_reshaped = b.reshape(1, -1)  # Now shape (1, 3)
    axes[1].imshow(b_reshaped, cmap='Reds', aspect='auto')
    axes[1].set_title('Array B: Shape (3,)\nBroadcasted to (2, 3)')
    axes[1].set_xlabel('Feature dimension')
    axes[1].set_ylabel('Broadcast dimension')
    # Show what happens during broadcasting
    for j in range(b.shape[0]):
        axes[1].text(j, 0, str(b[j]), ha='center', va='center', 
                    fontweight='bold')
        # Draw arrow to show repetition
        axes[1].annotate('', xy=(j, 0.5), xytext=(j, 0),
                        arrowprops=dict(arrowstyle='->', color='black', alpha=0.5))
    
    # --- Plot 3: Result of A + B ---
    result = a + b  # Broadcasting happens here!
    axes[2].imshow(result, cmap='Greens', aspect='auto')
    axes[2].set_title('Result: A + B\nShape (2, 3)')
    axes[2].set_xlabel('Feature dimension')
    axes[2].set_ylabel('Sample dimension')
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            axes[2].text(j, i, str(result[i, j]), ha='center', va='center',
                        fontweight='bold')
    
    plt.tight_layout()
    # Main title explaining the concept
    plt.suptitle('BROADCASTING: How (2,3) + (3,) becomes (2,3)\n'
                 'The 1D array B is repeated across rows to match A', 
                 y=1.05, fontsize=14, fontweight='bold')
    plt.show()


def visualize_unsqueeze_operation():
    """
    Show how adding dimensions (unsqueezing) works.
    
    ML CONTEXT: This is like adding "container" dimensions:
    sample → batch of samples → batch of sequences of samples
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    
    # Start with simple 1D array (like a single feature vector)
    original = np.array([1, 2, 3])
    
    # --- Plot 1: Original 1D array ---
    axes[0].bar(range(len(original)), original, color='blue', alpha=0.7)
    axes[0].set_title(f'1D Vector\nShape: {original.shape}')
    axes[0].set_xlabel('Feature index')
    axes[0].set_ylabel('Value')
    axes[0].set_ylim(0, 4)
    
    # --- Plot 2: After first unsqueeze (add batch dimension) ---
    unsqueezed1 = np.expand_dims(original, axis=0)  # Shape becomes (1, 3)
    axes[1].imshow(unsqueezed1, cmap='Blues', aspect='auto')
    axes[1].set_title(f'Add batch dimension\nShape: {unsqueezed1.shape}')
    axes[1].set_xlabel('Feature dimension')
    axes[1].set_ylabel('Batch dimension')
    # Label the single batch
    axes[1].text(1.5, 0, 'Batch 0', ha='center', va='center', 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    for j in range(unsqueezed1.shape[1]):
        axes[1].text(j, 0, str(unsqueezed1[0, j]), ha='center', va='center')
    
    # --- Plot 3: After second unsqueeze (add sequence dimension) ---
    unsqueezed2 = np.expand_dims(unsqueezed1, axis=0)  # Shape becomes (1, 1, 3)
    # We need to display 3D data in 2D - show it as a depth stack
    axes[2].imshow(unsqueezed2[0], cmap='Blues', aspect='auto')  # Take first "depth"
    axes[2].set_title(f'Add sequence dimension\nShape: {unsqueezed2.shape}')
    axes[2].set_xlabel('Feature dimension')
    axes[2].set_ylabel('Sequence dimension')
    axes[2].text(1.5, 0.5, 'Now 3D!\n(batch, sequence, features)', 
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    for j in range(unsqueezed2.shape[2]):
        axes[2].text(j, 0, str(unsqueezed2[0, 0, j]), ha='center', va='center')
    
    plt.tight_layout()
    plt.suptitle('UNSQUEEZE: Adding dimensions for compatibility\n'
                 'Common when preparing data for neural networks', 
                 y=1.05, fontsize=12)
    plt.show()


def visualize_tiling_operation():
    """
    Show how tiling replicates arrays to match target shapes.
    
    ML CONTEXT: Useful when you have per-channel parameters that need to
    be applied to every spatial location in an image.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Source: A 2x1 array (like 2 channels, 1 spatial location)
    source = np.array([[1], [2]])
    
    # --- Plot 1: Source array ---
    axes[0].imshow(source, cmap='Reds', aspect='auto')
    axes[0].set_title(f'Source array\nShape: {source.shape}')
    axes[0].set_xlabel('Spatial dimension')
    axes[0].set_ylabel('Channel dimension')
    for i in range(source.shape[0]):
        axes[0].text(0, i, str(source[i, 0]), ha='center', va='center',
                    fontweight='bold')
    axes[0].text(0.5, -0.2, 'Each channel has\none parameter value', 
                ha='center', transform=axes[0].transAxes, fontsize=9)
    
    # --- Plot 2: Target shape we want to match ---
    # Create a reference array with target shape (2, 3)
    reference = np.zeros((2, 3))  # Just for visualization
    axes[1].imshow(reference, cmap='Greys', alpha=0.3, aspect='auto')
    axes[1].set_title(f'Target shape\nShape: {reference.shape}')
    axes[1].set_xlabel('Spatial positions (3)')
    axes[1].set_ylabel('Channels (2)')
    axes[1].text(1, 0.5, 'We want to apply\nchannel parameters\nacross ALL positions', 
                ha='center', va='center', transform=axes[1].transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # --- Plot 3: Tiled result ---
    result = tile_like(source, np.ones((2, 3)))  # Tile to shape (2, 3)
    axes[2].imshow(result, cmap='Greens', aspect='auto')
    axes[2].set_title(f'Tiled result\nShape: {result.shape}')
    axes[2].set_xlabel('Spatial positions')
    axes[2].set_ylabel('Channels')
    # Show how values are repeated
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            axes[2].text(j, i, str(result[i, j]), ha='center', va='center',
                        fontweight='bold')
        # Draw arrows showing repetition
        axes[2].annotate('', xy=(2.5, i), xytext=(0.5, i),
                        arrowprops=dict(arrowstyle='->', color='black', alpha=0.3))
    
    plt.tight_layout()
    plt.suptitle('TILING: Repeating parameters across dimensions\n'
                 'Example: Applying channel-wise normalization across an image', 
                 y=1.05, fontsize=12)
    plt.show()


def visualize_axis_permutation():
    """
    Show how reordering tensor dimensions works.
    
    ML CONTEXT: Different ML frameworks expect different dimension orders.
    This operation changes the "perspective" without changing the data.
    """
    fig = plt.figure(figsize=(15, 5))
    
    # Create a 3D tensor: (batch=2, height=3, width=4)
    # Like 2 images, each 3 pixels tall, 4 pixels wide
    original = np.random.randint(1, 10, (2, 3, 4))
    
    # --- Plot 1: Original tensor visualization ---
    ax1 = fig.add_subplot(131, projection='3d')
    # Create a voxel representation (like a 3D checkerboard)
    ax1.voxels(np.ones_like(original), facecolors='blue', alpha=0.3)
    ax1.set_title(f'Original: Shape {original.shape}\n(batch, height, width)')
    ax1.set_xlabel('Width (4)')
    ax1.set_ylabel('Height (3)')
    ax1.set_zlabel('Batch (2)')
    ax1.view_init(elev=20, azim=-35)
    
    # --- Plot 2: Permuted tensor ---
    # Change order to (width, batch, height) - common in some operations
    permuted = permute_axes(original, (2, 0, 1))
    
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.voxels(np.ones_like(permuted), facecolors='red', alpha=0.3)
    ax2.set_title(f'Permuted: Shape {permuted.shape}\n(width, batch, height)')
    ax2.set_xlabel('Height (3)')
    ax2.set_ylabel('Batch (2)')
    ax2.set_zlabel('Width (4)')
    ax2.view_init(elev=20, azim=-35)
    
    # --- Plot 3: Dimension mapping diagram ---
    ax3 = fig.add_subplot(133)
    # Show which dimension goes where
    mapping_data = np.array([
        [0, 2],  # Original dim 0 (batch) → New dim 2
        [1, 0],  # Original dim 1 (height) → New dim 0
        [2, 1]   # Original dim 2 (width) → New dim 1
    ])
    
    # Plot the mapping
    ax3.scatter(mapping_data[:, 0], mapping_data[:, 1], s=200, 
                c=['blue', 'green', 'red'], edgecolors='black')
    
    # Add labels for each dimension
    dim_names = ['Batch', 'Height', 'Width']
    for i, (src, dst) in enumerate(mapping_data):
        ax3.annotate(f'{dim_names[i]}: {src} → {dst}', 
                    (src, dst), 
                    xytext=(10, 10), 
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', alpha=0.5))
    
    # Connect with lines to show the mapping
    ax3.plot(mapping_data[:, 0], mapping_data[:, 1], 'k--', alpha=0.3)
    
    ax3.set_xlim(-0.5, 2.5)
    ax3.set_ylim(-0.5, 2.5)
    ax3.set_xlabel('Original dimension index')
    ax3.set_ylabel('New dimension index')
    ax3.set_title('Dimension Permutation Mapping')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('AXIS PERMUTATION: Reordering tensor dimensions\n'
                 'Useful when switching between ML frameworks', 
                 y=1.05, fontsize=12)
    plt.show()


def visualize_performance_comparison(loop_time, vec_time):
    """
    Visualize the dramatic speed difference between loops and vectorization.
    
    KEY MESSAGE: Vectorization isn't just "a bit faster" - it's often
    100x or 1000x faster for ML-scale operations!
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Data for comparison
    methods = ['Python Loops', 'NumPy Vectorized']
    times = [loop_time, vec_time]
    colors = ['red', 'green']
    
    # --- Plot 1: Time comparison (log scale) ---
    bars = ax1.bar(methods, times, color=colors, alpha=0.7)
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Execution Time Comparison')
    ax1.set_yscale('log')  # Log scale because difference is huge!
    
    # Add time labels on bars
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height,
                f'{time_val:.6f}s', 
                ha='center', va='bottom',
                fontweight='bold')
    
    # Add a dramatic comparison note
    if loop_time > vec_time * 10:
        ax1.text(0.5, 0.9, '🚨 ORDER OF MAGNITUDE DIFFERENCE!', 
                ha='center', transform=ax1.transAxes,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # --- Plot 2: Speedup factor ---
    speedup = loop_time / vec_time
    ax2.bar(['Speedup'], [speedup], color='orange', alpha=0.7)
    ax2.set_ylabel('Times Faster')
    ax2.set_title(f'Vectorization Speedup\n{methods[0]} / {methods[1]}')
    
    # Big, bold speedup number
    ax2.text(0, speedup/2, f'{speedup:.1f}x', 
             ha='center', va='center', 
             fontsize=24, fontweight='bold', color='darkred')
    
    # Add context about what this means
    if speedup > 100:
        message = 'Game-changing for training!'
        color = 'red'
    elif speedup > 10:
        message = 'Significant impact on training time'
        color = 'orange'
    else:
        message = 'Noticeable improvement'
        color = 'green'
    
    ax2.text(0, speedup * 1.1, message, 
             ha='center', va='bottom', color=color, fontweight='bold')
    
    plt.tight_layout()
    plt.suptitle('WHY VECTORIZATION MATTERS FOR MACHINE LEARNING\n'
                 'Even simple operations show massive speed differences', 
                 y=1.02, fontsize=14, fontweight='bold')
    plt.show()


def visualize_fused_scale_shift():
    """
    Visualize the fused scale-shift operation that's everywhere in deep learning.
    
    ML CONTEXT: This pattern appears in batch norm, layer norm, attention,
    and many other places. Understanding it is crucial!
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Create example data that mimics real ML scenarios
    a = np.array([[1, 2, 3],
                  [4, 5, 6]])  # Shape (2, 3) - like 2 samples, 3 features
    b = np.array([10, 20, 30])   # Scale factors (gamma in batch norm)
    c = np.array([1, 1, 1])      # Shift values (beta in batch norm)
    
    # --- Top-left: Input data A ---
    im1 = axes[0, 0].imshow(a, cmap='Blues', aspect='auto')
    axes[0, 0].set_title('Input Data A\n(2 samples × 3 features)')
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            axes[0, 0].text(j, i, f'x={a[i, j]}', 
                           ha='center', va='center',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    axes[0, 0].set_xlabel('Feature index')
    axes[0, 0].set_ylabel('Sample index')
    
    # --- Top-right: Scale factors B ---
    im2 = axes[0, 1].imshow(b.reshape(1, -1), cmap='Reds', aspect='auto')
    axes[0, 1].set_title('Scale Factors B\n(per-feature multipliers)')
    for j in range(b.shape[0]):
        axes[0, 1].text(j, 0, f'×{b[j]}', 
                       ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    axes[0, 1].set_xlabel('Feature index')
    axes[0, 1].set_ylabel('(Broadcast dimension)')
    # Show broadcasting with arrows
    for j in range(b.shape[0]):
        axes[0, 1].annotate('', xy=(j, 0.3), xytext=(j, 0),
                           arrowprops=dict(arrowstyle='->', color='black', alpha=0.5))
    
    # --- Bottom-left: Shift values C ---
    im3 = axes[1, 0].imshow(c.reshape(1, -1), cmap='Greens', aspect='auto')
    axes[1, 0].set_title('Shift Values C\n(per-feature offsets)')
    for j in range(c.shape[0]):
        axes[1, 0].text(j, 0, f'+{c[j]}', 
                       ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    axes[1, 0].set_xlabel('Feature index')
    axes[1, 0].set_ylabel('(Broadcast dimension)')
    
    # --- Bottom-right: Result ---
    result = fused_scale_shift(a, b, c)
    im4 = axes[1, 1].imshow(result, cmap='Purples', aspect='auto')
    axes[1, 1].set_title('Result: A × B + C\n(scale then shift)')
    # Show the computation for a few cells
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            # Show the actual computation
            computation = f'{a[i,j]}×{b[j]}+{c[j]}={result[i,j]}'
            axes[1, 1].text(j, i, computation, 
                           ha='center', va='center', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    axes[1, 1].set_xlabel('Feature index')
    axes[1, 1].set_ylabel('Sample index')
    
    plt.tight_layout()
    plt.suptitle('FUSED SCALE-SHIFT: The workhorse of deep learning\n'
                 'Pattern: output = input × scale + shift  (per-feature operations)', 
                 y=1.02, fontsize=14, fontweight='bold')
    plt.show()


def main():
    """
    Main demonstration function - your guided tour through NumPy broadcasting!
    
    This runs all the visualizations and comparisons to give you a complete
    understanding of how broadcasting works and why it's essential for ML.
    """
    
    print("=" * 70)
    print("WELCOME TO NUMPY BROADCASTING FOR MACHINE LEARNING")
    print("=" * 70)
    print("\n🎯 Learning Objectives:")
    print("1. Understand how NumPy handles different-shaped arrays")
    print("2. Learn essential tensor manipulation operations")
    print("3. See why vectorization is crucial for ML performance")
    print("4. Recognize common patterns in deep learning code")
    
    print("\n" + "=" * 70)
    print("PART 1: VISUALIZING KEY CONCEPTS")
    print("=" * 70)
    
    print("\n📊 1. Broadcasting Concept")
    print("   How (2,3) + (3,) becomes (2,3)")
    visualize_broadcasting_concept()
    
    print("\n📐 2. Dimension Expansion (Unsqueeze)")
    print("   Adding dimensions for compatibility")
    visualize_unsqueeze_operation()
    
    print("\n🔁 3. Tiling Operation")
    print("   Repeating parameters across dimensions")
    visualize_tiling_operation()
    
    print("\n🔄 4. Axis Permutation")
    print("   Reordering tensor dimensions")
    visualize_axis_permutation()
    
    print("\n⚡ 5. Fused Scale-Shift Operation")
    print("   The fundamental pattern in deep learning")
    visualize_fused_scale_shift()
    
    print("\n" + "=" * 70)
    print("PART 2: PERFORMANCE COMPARISON - WHY VECTORIZATION MATTERS")
    print("=" * 70)
    
    # Typical ML scenario: batch of 16 samples, each with 32 features
    # (Small numbers for demonstration, real ML uses thousands!)
    N, C = 16, 32
    print(f"\nSimulating ML operation on {N} samples with {C} features each")
    
    # Create example data
    a = np.random.randn(N, C)  # Batch of samples (e.g., neural network activations)
    b = np.random.randn(C)     # Per-feature scale parameters (e.g., batch norm gamma)
    c = np.random.randn(C)     # Per-feature shift parameters (e.g., batch norm beta)
    
    print("\nTiming loop-based implementation (educational, not practical)...")
    start = perf_counter()
    loop_result = loop_multiply_add(a, b, c)
    loop_time = perf_counter() - start
    
    print("Timing vectorized implementation (production-ready)...")
    start = perf_counter()
    vec_result = vectorized_multiply_add(a, b, c)
    vec_time = perf_counter() - start
    
    # Verify both methods give same result (they should!)
    print(f"\n✅ Verification: Results match? {np.allclose(loop_result, vec_result)}")
    print(f"   (Small differences might occur due to floating-point rounding)")
    
    print(f"\n⏱️  Loop time:    {loop_time:.6f} seconds")
    print(f"⏱️  Vectorized:   {vec_time:.6f} seconds")
    
    speedup = loop_time / vec_time
    print(f"🚀 Speedup:       {speedup:.1f}x faster!")
    
    # Visualize the performance difference
    visualize_performance_comparison(loop_time, vec_time)
    
    print("\n" + "=" * 70)
    print("PART 3: PRACTICAL TENSOR MANIPULATIONS")
    print("=" * 70)
    
    print("\n🧩 1. Unsqueeze_to: Adding dimensions")
    x = np.array([1, 2, 3])
    print(f"   Original: shape {x.shape} = {x}")
    y = unsqueeze_to(x, 3)
    print(f"   After unsqueeze_to(3): shape {y.shape}")
    print(f"   Now compatible with 3D operations!")
    
    print("\n🧩 2. Tile_like: Replicating arrays")
    src = np.array([[1], [2]])
    ref = np.array([[1, 2, 3], [4, 5, 6]])
    tiled = tile_like(src, ref)
    print(f"   Source shape: {src.shape}")
    print(f"   Target shape: {ref.shape}")
    print(f"   Tiled shape:  {tiled.shape}")
    print(f"   Tiled array:\n{tiled}")
    
    print("\n🧩 3. Fused Scale-Shift: ML fundamental")
    a_small = np.array([[1, 2], [3, 4]])
    b_small = np.array([10, 20])
    c_small = np.array([1, 2])
    result = fused_scale_shift(a_small, b_small, c_small)
    print(f"   Input A:\n{a_small}")
    print(f"   Scale B: {b_small}")
    print(f"   Shift C: {c_small}")
    print(f"   Result A×B+C:\n{result}")
    
    print("\n🧩 4. Permute_axes: Dimension reordering")
    x_3d = np.random.randn(2, 3, 4)
    y_3d = permute_axes(x_3d, (2, 0, 1))
    print(f"   Original shape (batch, height, width): {x_3d.shape}")
    print(f"   Permuted shape (width, batch, height): {y_3d.shape}")
    
    print("\n" + "=" * 70)
    print("🎓 SUMMARY & KEY TAKEAWAYS")
    print("=" * 70)
    
    summary = """
    WHAT YOU'VE LEARNED:
    
    1. Broadcasting: NumPy automatically expands smaller arrays to match
       larger ones, enabling operations between different shapes.
       
    2. Dimension Manipulation:
       • unsqueeze_to: Add dimensions for compatibility
       • tile_like: Repeat arrays to match target shapes
       • permute_axes: Reorder dimensions for different frameworks
       
    3. Vectorization: Using NumPy's built-in operations instead of Python
       loops is NOT just "a bit faster" - it's often 100-1000x faster!
       
    4. ML Patterns: The fused scale-shift (a*b + c) appears everywhere in
       deep learning (batch norm, layer norm, attention, etc.).
    
    PRACTICAL ADVICE:
    • Always use vectorized operations in ML code
    • Understand broadcasting rules to avoid shape errors
    • Use dimension manipulation to prepare data correctly
    • Profile your code to identify bottlenecks
    """
    
    print(summary)
    
    print("\n" + "=" * 70)
    print("🚀 NEXT STEPS FOR YOUR ML JOURNEY")
    print("=" * 70)
    
    next_steps = """
    1. Practice: Try modifying the examples with your own arrays
    2. Explore: Look at how real ML libraries (PyTorch, TensorFlow) 
       use these operations
    3. Apply: Use these concepts in your own ML projects
    4. Deepen: Study NumPy's official documentation on broadcasting
    
    Remember: Understanding these fundamentals will make you a better
    ML practitioner and help you debug complex shape-related errors!
    """
    
    print(next_steps)
    print("\nHappy coding with NumPy! 🎯")


if __name__ == "__main__":
    # Run the complete tutorial
    main()