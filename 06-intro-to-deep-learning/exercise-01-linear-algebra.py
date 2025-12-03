"""
Linear Algebra Fundamentals for Machine Learning
================================================

This module demonstrates essential linear algebra operations that form the 
mathematical foundation of machine learning algorithms. Each function includes
practical ML applications and visual explanations.

Key Concepts Covered:
1. Dot Products → Similarity measurement, neural network layers
2. Outer Products → Attention mechanisms, feature interactions
3. Batch Operations → Parallel processing in deep learning
4. Normalization → Feature scaling, gradient stability
5. Matrix Analysis → Numerical stability, model diagnostics

Author: kira-ml (GitHub)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional

# ============================================================================
# CORE LINEAR ALGEBRA OPERATIONS
# ============================================================================

def vector_dot(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the dot product between two vectors.
    
    📚 ML CONCEPT: Similarity Measurement
    The dot product measures the projection of one vector onto another.
    In ML applications:
    - Cosine similarity = (a·b)/(||a||·||b||) → for normalized vectors
    - Neural network layers: output = input·weights + bias
    - Self-attention scores in transformers
    
    🧮 Mathematical Definition: a·b = Σ(a_i * b_i) = ||a||·||b||·cos(θ)
    
    Args:
        a: First vector (1D array) - e.g., a word embedding vector
        b: Second vector (1D array) - e.g., a query vector in attention
        
    Returns:
        float: Dot product value (positive if vectors point in similar direction)
        
    Raises:
        ValueError: If inputs are not 1D or have mismatched shapes
        
    Example:
        >>> word_embedding = [0.2, 0.8, -0.1]
        >>> query_vector = [0.1, 0.7, 0.2]
        >>> similarity = vector_dot(word_embedding, query_vector)
    """
    # Dimension validation - crucial for debugging ML models
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Dot product requires 1D vectors. "
                         f"Got shapes: {a.shape} and {b.shape}")
    if a.shape != b.shape:
        raise ValueError(f"Vector dimension mismatch. "
                         f"Got {a.shape} and {b.shape}. "
                         "In ML, this often means embedding size mismatch.")
    
    # The @ operator is equivalent to np.dot() but more readable
    return float(a @ b)


def vector_outer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the outer product between two vectors.
    
    📚 ML CONCEPT: Feature Interaction Matrix
    Creates a matrix where M[i,j] = a[i] * b[j].
    ML applications:
    - Attention weights in transformers: query ⊗ key
    - Covariance matrices in statistics
    - Feature interaction terms (like in polynomial features)
    - Outer product layers in neural networks
    
    🧮 Mathematical Definition: (a⊗b)[i,j] = a[i] * b[j]
    
    Args:
        a: First vector (1D array) - typically a query or feature vector
        b: Second vector (1D array) - typically a key or another feature vector
        
    Returns:
        np.ndarray: Matrix of shape (len(a), len(b)) capturing all pairwise products
        
    Raises:
        ValueError: If inputs are not 1D vectors
        
    Example:
        >>> query = np.array([0.5, -0.2])    # Shape (2,)
        >>> key = np.array([0.1, 0.3, 0.4])  # Shape (3,)
        >>> attention_scores = vector_outer(query, key)  # Shape (2, 3)
    """
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Outer product requires 1D vectors")
    
    # np.outer computes all pairwise multiplications efficiently
    return np.outer(a, b)


def batch_dot(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Perform batched matrix multiplication.
    
    📚 ML CONCEPT: Parallel Processing in Neural Networks
    Processes multiple samples simultaneously for efficiency.
    Critical for:
    - Training on mini-batches (SGD optimization)
    - Transformer self-attention across sequence positions
    - Convolutional neural networks with batch dimension
    
    🧮 Mathematical Definition: For each batch i: C[i] = A[i] @ B[i]
    
    Args:
        A: Batch of matrices with shape (batch_size, m, n)
           e.g., (batch_size, sequence_length, embedding_dim)
        B: Batch of matrices with shape (batch_size, n, p)
           e.g., (batch_size, embedding_dim, hidden_dim)
           
    Returns:
        np.ndarray: Result of shape (batch_size, m, p)
                    e.g., (batch_size, sequence_length, hidden_dim)
                    
    Raises:
        ValueError: If batch sizes don't match or inner dimensions are incompatible
        
    Example:
        >>> # Mini-batch of 32 samples, each with 10 tokens of 256-dim embeddings
        >>> batch_embeddings = np.random.randn(32, 10, 256)
        >>> weight_matrix = np.random.randn(256, 128)
        >>> # Expand weight matrix to match batch dimension
        >>> batch_weights = np.tile(weight_matrix, (32, 1, 1)).reshape(32, 256, 128)
        >>> batch_output = batch_dot(batch_embeddings, batch_weights)  # Shape (32, 10, 128)
    """
    # Validation checks prevent silent errors in ML training
    if A.ndim != 3 or B.ndim != 3:
        raise ValueError("Batch operations require 3D arrays: "
                         f"(batch, rows, cols). Got shapes: {A.shape}, {B.shape}")
    
    if A.shape[0] != B.shape[0]:
        raise ValueError(f"Batch size mismatch: {A.shape[0]} vs {B.shape[0]}. "
                         "Each sample in batch must have corresponding weights.")
    
    if A.shape[2] != B.shape[1]:
        raise ValueError(f"Inner dimension mismatch: {A.shape[2]} vs {B.shape[1]}. "
                         "This is like trying to multiply embeddings of different sizes.")
    
    # The @ operator automatically handles batch dimension in NumPy
    return A @ B


def normalize_vectors(X: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize vectors to unit length (L2 normalization).
    
    📚 ML CONCEPT: Feature Scaling & Gradient Stability
    Why normalize in ML?
    1. Prevents feature dominance: Large magnitude features don't overwhelm others
    2. Improves convergence: Gradients have similar scale across dimensions
    3. Enables cosine similarity: Dot product = cosine similarity for unit vectors
    4. Used in: Batch normalization, layer normalization, word embeddings
    
    🧮 Mathematical Definition: v_normalized = v / ||v||₂
    
    Args:
        X: Input array of vectors - could be embeddings, features, or gradients
        axis: Axis along which to compute norms (-1 usually means last dimension)
        eps: Small constant to prevent division by zero (numerical stability)
        
    Returns:
        np.ndarray: Unit vectors with same shape as input
        
    Example:
        >>> # Word embeddings before normalization
        >>> embeddings = np.array([[3.0, 4.0],  # Length 5
        >>>                        [1.0, 1.0]]) # Length √2 ≈ 1.414
        >>> normalized = normalize_vectors(embeddings)
        >>> np.linalg.norm(normalized, axis=1)  # All lengths = 1.0
        array([1., 1.])
    """
    # Compute L2 norm along specified axis
    norm = np.linalg.norm(X, axis=axis, keepdims=True)
    
    # Handle zero vectors to prevent NaN gradients during training
    norm = np.where(norm < eps, 1.0, norm)  # If norm ≈ 0, treat as 1.0
    
    return X / norm


def matrix_rank_estimate(M: np.ndarray, tol: Optional[float] = None) -> int:
    """
    Estimate matrix rank using Singular Value Decomposition (SVD).
    
    📚 ML CONCEPT: Information Capacity & Model Complexity
    Rank measures linearly independent rows/columns.
    ML implications:
    - Low rank → Redundant features → Consider dimensionality reduction (PCA)
    - Full rank → All features contribute unique information
    - Used in: Recommender systems (low-rank matrix factorization),
               Autoencoders (bottleneck layer creates low-rank representation)
    
    🧮 Mathematical Definition: rank = # of non-zero singular values
    (Numerically: # of singular values > tolerance)
    
    Args:
        M: Input matrix (e.g., dataset features, weight matrix)
        tol: Threshold below which singular values are considered zero.
              If None, uses machine precision and matrix size.
              
    Returns:
        int: Estimated rank (number of "effective" dimensions)
        
    Example:
        >>> # Feature matrix with 100 samples, 50 features
        >>> X = np.random.randn(100, 50)
        >>> # But features 30-50 are just noise added to first 30 features
        >>> X[:, 30:] = X[:, :20] @ np.random.randn(20, 20)
        >>> rank = matrix_rank_estimate(X)  # Will be ≈ 30, not 50
    """
    if tol is None:
        # Standard numerical linear algebra tolerance formula
        # Accounts for floating-point precision and matrix scale
        tol = max(M.shape) * np.finfo(M.dtype).eps * np.linalg.norm(M, ord=2)
    
    # SVD reveals the intrinsic dimensionality of the matrix
    singular_values = np.linalg.svd(M, compute_uv=False)
    
    # Count singular values above threshold
    return np.sum(singular_values > tol)


def condition_number(M: np.ndarray) -> float:
    """
    Compute the condition number of a matrix.
    
    📚 ML CONCEPT: Numerical Stability & Training Difficulty
    Condition number = (max singular value) / (min singular value)
    
    Why it matters in ML:
    - High condition number (> 10^6) → Ill-conditioned matrix
    - Ill-conditioning causes: Gradient explosion/vanishing, slow convergence,
                                sensitivity to input noise
    - Well-conditioned matrices (~1-1000) → Stable optimization
    
    🧮 Mathematical Definition: κ(M) = σ_max(M) / σ_min(M)
    
    Args:
        M: Input matrix (e.g., Hessian matrix, feature covariance)
        
    Returns:
        float: Condition number (inf if singular/rank-deficient)
        
    Example:
        >>> # Weight matrix of a neural network layer
        >>> W = np.random.randn(256, 128)
        >>> cond = condition_number(W)
        >>> if cond > 1e6:
        >>>     print("Warning: Ill-conditioned weights may cause training issues")
        >>>     print("Consider: Weight normalization, better initialization")
    """
    singular_values = np.linalg.svd(M, compute_uv=False)
    
    # Check for singularity (zero singular values)
    if np.any(singular_values == 0):
        return float('inf')  # Perfectly singular matrix
    
    return float(singular_values[0] / singular_values[-1])


# ============================================================================
# VISUALIZATION FUNCTIONS WITH EDUCATIONAL ANNOTATIONS
# ============================================================================

def plot_dot_product(a: np.ndarray, b: np.ndarray) -> None:
    """
    Visualize dot product as vector projection with ML context.
    
    Educational focus:
    1. Shows geometric interpretation: a·b = |a||b|cos(θ)
    2. Demonstrates projection operation used in attention mechanisms
    3. Illustrates similarity measurement between embeddings
    
    Args:
        a: First 2D vector (for visualization)
        b: Second 2D vector (for visualization)
    """
    from matplotlib.lines import Line2D
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot original vectors
    ax.quiver(0, 0, a[0], a[1], angles='xy', scale_units='xy', 
              scale=1, color='r', linewidth=3, alpha=0.7, label=f'Vector a ({a})')
    ax.quiver(0, 0, b[0], b[1], angles='xy', scale_units='xy', 
              scale=1, color='b', linewidth=3, alpha=0.7, label=f'Vector b ({b})')
    
    # Compute projection of a onto b (used in attention mechanisms)
    proj_length = vector_dot(a, b) / vector_dot(b, b)  # Scalar projection
    projection = proj_length * b  # Vector projection
    
    # Plot projection (like attention output onto key vector)
    ax.quiver(0, 0, projection[0], projection[1], angles='xy', 
              scale_units='xy', scale=1, color='g', linewidth=2, 
              linestyle='--', alpha=0.8, label='Projection: a onto b')
    
    # Add projection line (shows the "shadow" cast by a on b)
    ax.plot([a[0], projection[0]], [a[1], projection[1]], 
            'k:', alpha=0.5, label='Orthogonal drop')
    
    # Formatting with educational annotations
    ax.set_xlim(-1, max(a[0], b[0], projection[0]) + 1)
    ax.set_ylim(-1, max(a[1], b[1], projection[1]) + 1)
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)
    ax.axvline(0, color='black', linewidth=0.5, alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add ML context annotations
    dot_result = vector_dot(a, b)
    cos_sim = dot_result / (np.linalg.norm(a) * np.linalg.norm(b))
    
    ax.text(0.02, 0.98, 
            f'Dot Product = {dot_result:.2f}\n'
            f'Cosine Similarity = {cos_sim:.2f}\n'
            f'Angle = {np.degrees(np.arccos(np.clip(cos_sim, -1, 1))):.1f}°',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_title('Dot Product: Foundation of Similarity in ML\n'
                '(Used in Attention, Recommendation, Embeddings)', 
                fontsize=14, pad=20)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


def plot_outer_product(a: np.ndarray, b: np.ndarray) -> None:
    """
    Visualize outer product as interaction heatmap.
    
    Educational focus:
    1. Shows all pairwise multiplications between vector elements
    2. Illustrates attention score matrices in transformers
    3. Demonstrates feature interaction matrices
    
    Args:
        a: First vector (will be rows in output)
        b: Second vector (will be columns in output)
    """
    outer = vector_outer(a, b)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Heatmap showing interaction strengths
    cax = ax.imshow(outer, cmap='RdBu_r', aspect='auto', 
                    interpolation='nearest')
    
    # Annotate each cell with value
    for i in range(outer.shape[0]):
        for j in range(outer.shape[1]):
            # Color text based on background brightness
            text_color = 'white' if abs(outer[i, j]) > np.max(np.abs(outer))/2 else 'black'
            ax.text(j, i, f'{outer[i, j]:.1f}', 
                    ha='center', va='center', color=text_color,
                    fontweight='bold')
    
    ax.set_title('Outer Product: Attention Score Matrix\n'
                f'Shape: {outer.shape} = (len(a), len(b))', 
                fontsize=14, pad=20)
    ax.set_xlabel('Vector b indices (e.g., Key positions in attention)')
    ax.set_ylabel('Vector a indices (e.g., Query positions in attention)')
    
    # Add colorbar with label
    cbar = plt.colorbar(cax, ax=ax)
    cbar.set_label('Interaction Strength', rotation=270, labelpad=15)
    
    # Add vector values as annotations
    ax.text(0.02, 0.98, f'a = {a}', transform=ax.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    ax.text(0.02, 0.02, f'b = {b}', transform=ax.transAxes,
            verticalalignment='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.show()


def plot_batch_dot_shapes(A: np.ndarray, B: np.ndarray, result: np.ndarray) -> None:
    """
    Visualize batch operation shapes for understanding parallel processing.
    
    Educational focus:
    1. Shows how batch dimension enables parallel computation
    2. Illustrates shape transformations in neural network layers
    3. Demonstrates mini-batch processing concept
    
    Args:
        A: Input batch A with shape (batch, m, n)
        B: Input batch B with shape (batch, n, p)
        result: Output from batch_dot with shape (batch, m, p)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # Create educational diagram
    text = f"""
    🚀 BATCH MATRIX MULTIPLICATION: PARALLEL PROCESSING IN ML
    
    This operation processes {A.shape[0]} samples simultaneously.
    Each sample gets its own matrix multiplication, but computed in parallel.
    
    ┌─────────────────────────────────────────────────────────────┐
    │                         BATCH INPUT A                        │
    │  Shape: {str(A.shape):<15} = (batch_size, m, n)           │
    │  Example: ({A.shape[0]}, sequence_length, embedding_dim) │
    └─────────────────────────────────────────────────────────────┘
                                    @
    ┌─────────────────────────────────────────────────────────────┐
    │                         BATCH INPUT B                        │
    │  Shape: {str(B.shape):<15} = (batch_size, n, p)           │
    │  Example: ({B.shape[0]}, embedding_dim, hidden_dim)      │
    └─────────────────────────────────────────────────────────────┘
                                    =
    ┌─────────────────────────────────────────────────────────────┐
    │                         OUTPUT                               │
    │  Shape: {str(result.shape):<15} = (batch_size, m, p)      │
    │  Example: ({result.shape[0]}, sequence_length, hidden_dim)│
    └─────────────────────────────────────────────────────────────┘
    
    🔍 MATHEMATICAL INTERPRETATION:
    For each batch element i (0 ≤ i < batch_size):
        result[i] = A[i] @ B[i]
    
    ⚡ PERFORMANCE BENEFIT:
    • GPUs process all batch elements in parallel
    • Amortizes data loading overhead
    • Provides better gradient estimates in SGD
    """
    
    ax.text(0.1, 0.5, text, fontsize=11, va='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax.set_title("Batch Processing: The Secret to Fast Neural Network Training", 
                fontsize=16, pad=30)
    plt.tight_layout()
    plt.show()


def plot_normalization(X: np.ndarray, normalized: np.ndarray) -> None:
    """
    Visualize vector normalization process.
    
    Educational focus:
    1. Shows L2 normalization geometrically
    2. Illustrates movement to unit sphere
    3. Demonstrates equal scaling of all vectors
    
    Args:
        X: Original vectors
        normalized: Normalized unit vectors
    """
    from matplotlib.lines import Line2D
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Plot 1: Original Vectors ---
    colors = ['r', 'b', 'g', 'orange']
    for i, vec in enumerate(X):
        ax1.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', 
                  scale=1, color=colors[i], width=0.02, alpha=0.8)
        
        # Add vector labels
        ax1.text(vec[0]/2, vec[1]/2, f'v{i+1}', fontsize=12, 
                bbox=dict(boxstyle='round', facecolor=colors[i], alpha=0.3))
        
        # Show magnitude
        magnitude = np.linalg.norm(vec)
        ax1.text(vec[0]*1.05, vec[1]*1.05, f'||v{i+1}|| = {magnitude:.2f}', 
                fontsize=9, color=colors[i])
    
    ax1.set_xlim(-0.5, np.max(np.abs(X[:,0])) * 1.3)
    ax1.set_ylim(-0.5, np.max(np.abs(X[:,1])) * 1.3)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.axhline(0, color='k', lw=0.5, alpha=0.5)
    ax1.axvline(0, color='k', lw=0.5, alpha=0.5)
    ax1.set_title('Original Vectors (Different Magnitudes)')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    
    # --- Plot 2: Normalized Vectors ---
    # Draw unit circle (all normalized vectors lie here)
    circle = plt.Circle((0, 0), 1, color='gray', fill=False, 
                       linestyle='--', linewidth=2, alpha=0.5)
    ax2.add_patch(circle)
    ax2.text(0, 1.1, 'Unit Circle', ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='gray', alpha=0.2))
    
    # Plot normalized vectors
    for i, vec in enumerate(normalized):
        ax2.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', 
                  scale=1, color=colors[i], width=0.03, alpha=0.8)
        
        # Verify and show magnitude = 1
        magnitude = np.linalg.norm(vec)
        ax2.text(vec[0]*1.1, vec[1]*1.1, f'||v{i+1}|| = {magnitude:.6f}', 
                fontsize=9, color=colors[i])
        
        # Draw arrow from original to normalized position
        if i < len(X):
            ax2.arrow(X[i,0], X[i,1], vec[0]-X[i,0], vec[1]-X[i,1],
                     head_width=0.05, head_length=0.1, fc='gray', ec='gray',
                     alpha=0.5, linestyle=':')
    
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.axhline(0, color='k', lw=0.5, alpha=0.5)
    ax2.axvline(0, color='k', lw=0.5, alpha=0.5)
    ax2.set_title('Normalized Vectors (All Magnitudes = 1.0)')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    
    # Add educational annotation
    fig.text(0.5, 0.02, 
             '📚 ML INSIGHT: Normalization ensures all vectors contribute equally to distance metrics.\n'
             'Cosine similarity between normalized vectors = their dot product.',
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.show()


def plot_matrix_analysis(M: np.ndarray, name: str) -> None:
    """
    Comprehensive matrix analysis visualization.
    
    Educational focus:
    1. Shows matrix structure via heatmap
    2. Visualizes singular value spectrum
    3. Annotates with rank and condition number insights
    
    Args:
        M: Matrix to analyze
        name: Descriptive name for the matrix type
    """
    # Compute analysis metrics
    singular_values = np.linalg.svd(M, compute_uv=False)
    rank = matrix_rank_estimate(M)
    cond_num = condition_number(M)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Plot 1: Matrix Heatmap ---
    im = ax1.imshow(M, cmap='RdBu_r', aspect='auto')
    
    # Annotate cells with values
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            text_color = 'white' if abs(M[i, j]) > np.max(np.abs(M))/2 else 'black'
            ax1.text(j, i, f'{M[i, j]:.2f}', 
                    ha='center', va='center', color=text_color, fontsize=8)
    
    ax1.set_title(f'{name} Matrix\nShape: {M.shape}, Rank: {rank}', 
                 fontsize=14, pad=20)
    ax1.set_xlabel('Columns (Features/Dimensions)')
    ax1.set_ylabel('Rows (Samples/Measurements)')
    
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Matrix Value', rotation=270, labelpad=15)
    
    # --- Plot 2: Singular Value Spectrum ---
    indices = range(1, len(singular_values) + 1)
    bars = ax2.bar(indices, singular_values, color='steelblue', alpha=0.7)
    
    # Color bars based on whether they contribute to rank
    if rank < len(singular_values):
        for i in range(rank, len(singular_values)):
            bars[i].set_color('lightgray')
            bars[i].set_alpha(0.4)
    
    ax2.set_title('Singular Value Spectrum\n"Information Content" of Matrix', 
                 fontsize=14, pad=20)
    ax2.set_xlabel('Singular Value Index')
    ax2.set_ylabel('Singular Value Magnitude')
    ax2.grid(True, alpha=0.3)
    
    # Add rank threshold line
    if rank < len(singular_values):
        ax2.axvline(x=rank + 0.5, color='red', linestyle='--', alpha=0.7, 
                   label=f'Rank = {rank}')
        ax2.text(rank + 0.6, np.max(singular_values)*0.9, 
                f'Rank threshold\n({rank} significant dimensions)',
                color='red', fontsize=9)
    
    # Add condition number annotation
    cond_status = "ILL-CONDITIONED ⚠️" if cond_num > 1e6 else "Well-conditioned ✓"
    cond_color = "red" if cond_num > 1e6 else "green"
    
    ax2.text(0.05, 0.95, 
            f'Condition Number: {cond_num:.2e}\n{cond_status}\n'
            f'κ = σ_max/σ_min = {singular_values[0]:.2e}/{singular_values[-1]:.2e}',
            transform=ax2.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            color=cond_color, fontweight='bold')
    
    ax2.legend()
    
    # Add educational footer
    fig.text(0.5, 0.01, 
             '📚 ML INSIGHT: Low rank suggests redundant features. '
             'High condition number indicates numerical instability.',
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# COMPREHENSIVE DEMONSTRATION FUNCTION
# ============================================================================

def main() -> None:
    """
    Interactive tutorial: Linear Algebra for Machine Learning.
    
    Walks through each concept with:
    1. Mathematical definition
    2. ML application context
    3. Visual demonstration
    4. Practical examples
    
    Follow this tutorial to understand the linear algebra behind:
    - Neural networks
    - Transformers & attention
    - Recommendation systems
    - Dimensionality reduction
    """
    
    print("=" * 70)
    print("LINEAR ALGEBRA FUNDAMENTALS FOR MACHINE LEARNING")
    print("=" * 70)
    print("\nWelcome! This interactive tutorial covers the math behind modern ML.")
    print("Each section includes: Concept → Math → ML Application → Visualization\n")
    
    # ------------------------------------------------------------------------
    # 1. DOT PRODUCT: Similarity Measurement
    # ------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("1. 🔢 VECTOR DOT PRODUCT: Measuring Similarity")
    print("=" * 60)
    print("\nApplication: Attention scores, recommendation systems, embeddings")
    
    # Create sample vectors (2D for easy visualization)
    word_vector = np.array([3.0, 4.0])     # e.g., "king" embedding
    query_vector = np.array([4.0, -1.0])   # e.g., "royalty" query
    
    print(f"\nExample: Word similarity measurement")
    print(f"Word embedding: {word_vector} (e.g., 'king')")
    print(f"Query vector:   {query_vector} (e.g., 'royalty')")
    
    similarity = vector_dot(word_vector, query_vector)
    print(f"\nDot product (raw similarity): {similarity:.2f}")
    
    # Normalize for cosine similarity
    norm_word = normalize_vectors(word_vector.reshape(1, -1)).flatten()
    norm_query = normalize_vectors(query_vector.reshape(1, -1)).flatten()
    cosine_sim = vector_dot(norm_word, norm_query)
    print(f"Cosine similarity: {cosine_sim:.3f} (range: -1 to 1)")
    
    plot_dot_product(word_vector, query_vector)
    
    # ------------------------------------------------------------------------
    # 2. OUTER PRODUCT: Feature Interactions
    # ------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2. ✖️ OUTER PRODUCT: Attention & Feature Interactions")
    print("=" * 60)
    print("\nApplication: Transformer attention, covariance matrices")
    
    query = np.array([0.5, 1.0, 1.5])    # 3-dimensional query
    key = np.array([2.0, -0.5])          # 2-dimensional key
    
    print(f"\nExample: Attention score matrix calculation")
    print(f"Query vector (length {len(query)}): {query}")
    print(f"Key vector (length {len(key)}): {key}")
    
    attention_scores = vector_outer(query, key)
    print(f"\nAttention score matrix shape: {attention_scores.shape}")
    print(f"Matrix[i,j] = query[i] * key[j] (all pairwise interactions)")
    
    plot_outer_product(query, key)
    
    # ------------------------------------------------------------------------
    # 3. BATCH OPERATIONS: Parallel Processing
    # ------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("3. 📦 BATCH OPERATIONS: Mini-batch Training")
    print("=" * 60)
    print("\nApplication: Neural network training, transformer inference")
    
    # Simulate a mini-batch: 4 samples, each with 3 tokens of 5-dim embeddings
    batch_size = 4
    seq_length = 3
    embed_dim = 5
    hidden_dim = 8
    
    batch_embeddings = np.random.randn(batch_size, seq_length, embed_dim)
    weight_matrix = np.random.randn(embed_dim, hidden_dim)
    
    # Expand weights for batch operation
    batch_weights = np.tile(weight_matrix, (batch_size, 1, 1))
    batch_weights = batch_weights.reshape(batch_size, embed_dim, hidden_dim)
    
    print(f"\nExample: Transformer layer forward pass")
    print(f"Input batch shape:   {batch_embeddings.shape}")
    print(f"  = (batch_size={batch_size}, sequence_length={seq_length}, ")
    print(f"     embedding_dim={embed_dim})")
    print(f"\nWeight matrix shape: {weight_matrix.shape}")
    print(f"  = (embedding_dim={embed_dim}, hidden_dim={hidden_dim})")
    
    batch_output = batch_dot(batch_embeddings, batch_weights)
    print(f"\nOutput batch shape:  {batch_output.shape}")
    print(f"  = (batch_size={batch_size}, sequence_length={seq_length}, ")
    print(f"     hidden_dim={hidden_dim})")
    
    plot_batch_dot_shapes(batch_embeddings, batch_weights, batch_output)
    
    # ------------------------------------------------------------------------
    # 4. NORMALIZATION: Stable Training
    # ------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("4. 📏 VECTOR NORMALIZATION: Stable Gradients")
    print("=" * 60)
    print("\nApplication: BatchNorm, LayerNorm, embedding normalization")
    
    # Create vectors with different magnitudes
    raw_vectors = np.array([
        [3.0, 4.0],     # Length 5.0
        [1.0, 1.0],     # Length 1.414
        [-2.0, 1.0],    # Length 2.236
        [0.5, -1.5]     # Length 1.581
    ])
    
    print("\nOriginal vectors (different magnitudes cause problems):")
    for i, vec in enumerate(raw_vectors):
        mag = np.linalg.norm(vec)
        print(f"  Vector {i+1}: {vec} → Magnitude: {mag:.3f}")
    
    normalized = normalize_vectors(raw_vectors)
    
    print("\nNormalized vectors (all on unit sphere):")
    for i, vec in enumerate(normalized):
        mag = np.linalg.norm(vec)
        print(f"  Vector {i+1}: {vec} → Magnitude: {mag:.6f}")
    
    plot_normalization(raw_vectors[:2], normalized[:2])  # Plot first 2 for clarity
    
    # ------------------------------------------------------------------------
    # 5. MATRIX ANALYSIS: Model Diagnostics
    # ------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("5. 📊 MATRIX ANALYSIS: Understanding Your Model")
    print("=" * 60)
    print("\nApplication: Debugging training, feature analysis, stability checks")
    
    # Create different types of matrices
    matrices = [
        ("Rank-deficient", np.array([[1.0, 2.0], [2.0, 4.0]])),  # 2nd row = 2×1st
        ("Well-conditioned", np.eye(3)),                         # Identity
        ("Ill-conditioned", np.array([[1.0, 1.0], [1.0, 1.0000001]])),  # Nearly singular
        ("Random features", np.random.randn(10, 5)),             # Typical feature matrix
    ]
    
    for name, M in matrices:
        print(f"\n{name} Matrix Analysis:")
        print(f"  Shape: {M.shape}")
        
        rank = matrix_rank_estimate(M)
        print(f"  Rank: {rank} / {min(M.shape)}")
        
        cond_num = condition_number(M)
        if cond_num == float('inf'):
            print("  Condition number: ∞ (Singular matrix)")
        elif cond_num > 1e6:
            print(f"  Condition number: {cond_num:.2e} ⚠️ (Ill-conditioned)")
            print("    ML Impact: May cause gradient explosion/vanishing")
        else:
            print(f"  Condition number: {cond_num:.2e} ✓ (Well-conditioned)")
        
        # Only plot the first three for brevity
        if name in ["Rank-deficient", "Well-conditioned", "Ill-conditioned"]:
            plot_matrix_analysis(M, name)
    
    # ------------------------------------------------------------------------
    # SUMMARY & NEXT STEPS
    # ------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("🎓 SUMMARY: Linear Algebra in Machine Learning")
    print("=" * 70)
    
    summary = """
    KEY TAKEAWAYS:
    
    1. Dot Products → Measure similarity between embeddings
       • Used in: Attention mechanisms, recommendation systems
       • Normalize first for cosine similarity
    
    2. Outer Products → Create interaction matrices
       • Used in: Transformer attention, feature crossing
       • Captures all pairwise relationships
    
    3. Batch Operations → Parallel processing
       • Used in: Mini-batch training, GPU acceleration
       • Processes multiple samples simultaneously
    
    4. Normalization → Stable training
       • Used in: BatchNorm, LayerNorm, embedding layers
       • Prevents feature dominance, improves convergence
    
    5. Matrix Analysis → Model diagnostics
       • Rank: Information content, feature redundancy
       • Condition number: Numerical stability
    
    NEXT STEPS:
    • Apply dot products to build a simple recommendation system
    • Use batch operations to implement a neural network layer
    • Analyze your model's weight matrices for stability issues
    • Experiment with normalization in training loops
    """
    
    print(summary)
    
    print("\n🧠 Remember: Every modern ML algorithm is built on these foundations!")
    print("Happy learning! 🚀")


if __name__ == "__main__":
    # Run the interactive tutorial
    main()