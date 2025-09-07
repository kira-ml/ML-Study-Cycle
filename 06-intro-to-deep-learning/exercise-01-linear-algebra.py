import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional

# --- Original functions (unchanged) ---

def vector_dot(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Inputs must be 1D vectors")
    if a.shape != b.shape:
        raise ValueError(f"Vectors must have same length. Got {a.shape} and {b.shape}")
    return float(a @ b)

def vector_outer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Inputs must be 1D vectors")
    return np.outer(a, b)

def batch_dot(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.ndim != 3 or B.ndim != 3:
        raise ValueError("Inputs must be 3D arrays")
    if A.shape[0] != B.shape[0]:
        raise ValueError(f"Batch sizes must match. Got {A.shape[0]} and {B.shape[0]}")
    if A.shape[2] != B.shape[1]:
        raise ValueError(f"Inner dimensions must match. Got {A.shape[2]} and {B.shape[1]}")
    return A @ B

def normalize_vectors(X: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(X, axis=axis, keepdims=True)
    norm = np.where(norm < eps, 1.0, norm)
    return X / norm

def matrix_rank_estimate(M: np.ndarray, tol: Optional[float] = None) -> int:
    if tol is None:
        tol = max(M.shape) * np.finfo(M.dtype).eps * np.linalg.norm(M, ord=2)
    s = np.linalg.svd(M, compute_uv=False)
    return np.sum(s > tol)

def condition_number(M: np.ndarray) -> float:
    s = np.linalg.svd(M, compute_uv=False)
    if np.any(s == 0):
        return float('inf')
    return float(s[0] / s[-1])

# --- Visualization Functions ---

def plot_dot_product(a, b):
    """Visualize dot product as projection."""
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.quiver(0, 0, a[0], a[1], angles='xy', scale_units='xy', scale=1, color='r')
    ax.quiver(0, 0, b[0], b[1], angles='xy', scale_units='xy', scale=1, color='b')
    # Projection of a onto b
    proj = (vector_dot(a, b) / vector_dot(b, b)) * b
    ax.quiver(0, 0, proj[0], proj[1], angles='xy', scale_units='xy', scale=1, color='g', linestyle='--')
    ax.set_xlim(-1, max(a[0], b[0], proj[0]) + 1)
    ax.set_ylim(-1, max(a[1], b[1], proj[1]) + 1)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True)
    ax.set_aspect('equal')
    legend_handles = [
        Line2D([0], [0], color='r', lw=2, label=f'a = {a}'),
        Line2D([0], [0], color='b', lw=2, label=f'b = {b}'),
        Line2D([0], [0], color='g', lw=2, linestyle='--', label='proj_b(a)')
    ]
    ax.legend(handles=legend_handles)
    ax.set_title(f'Dot Product = {vector_dot(a, b):.2f}\n(Geometric: |a||b|cosθ)')
    plt.show()

def plot_outer_product(a, b):
    """Visualize outer product as heatmap."""
    outer = vector_outer(a, b)
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(outer, cmap='viridis', aspect='auto')
    ax.set_title('Outer Product Heatmap')
    ax.set_xlabel('Vector b index')
    ax.set_ylabel('Vector a index')
    plt.colorbar(cax, ax=ax)
    for i in range(outer.shape[0]):
        for j in range(outer.shape[1]):
            ax.text(j, i, f'{outer[i, j]:.1f}', ha='center', va='center', color='white')
    plt.show()

def plot_batch_dot_shapes(A, B, result):
    """Schematic of batch matrix multiplication shapes."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    text = f"""
    Batch Matrix Multiplication:
    A: {A.shape}   →   [batch={A.shape[0]}, m={A.shape[1]}, n={A.shape[2]}]
    B: {B.shape}   →   [batch={B.shape[0]}, n={B.shape[1]}, p={B.shape[2]}]
    Result: {result.shape} → [batch={result.shape[0]}, m={result.shape[1]}, p={result.shape[2]}]

    Example: A[0] @ B[0] = Result[0]
    """
    ax.text(0.1, 0.5, text, fontsize=12, va='center')
    ax.set_title("Batch Dot Product Shape Schematic", fontsize=14)
    plt.show()

def plot_normalization(X, normalized):
    """Plot original and normalized vectors."""
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    # Original
    for i, vec in enumerate(X):
        ax[0].quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', scale=1, color=['r', 'b'][i])
    ax[0].set_xlim(-0.5, np.max(X[:,0]) + 0.5)
    ax[0].set_ylim(-0.5, np.max(X[:,1]) + 0.5)
    ax[0].grid(True)
    ax[0].set_aspect('equal')
    legend_handles = [
        Line2D([0], [0], color='r', lw=2, label=f'v1={X[0]}'),
        Line2D([0], [0], color='b', lw=2, label=f'v2={X[1]}')
    ]
    ax[0].legend(handles=legend_handles)
    ax[0].set_title('Original Vectors')
    ax[0].axhline(0, color='k', lw=0.5)
    ax[0].axvline(0, color='k', lw=0.5)
    # Normalized
    circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
    ax[1].add_patch(circle)
    for i, vec in enumerate(normalized):
        ax[1].quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', scale=1, color=['g', 'm'][i])
    ax[1].set_xlim(-1.2, 1.2)
    ax[1].set_ylim(-1.2, 1.2)
    ax[1].grid(True)
    ax[1].set_aspect('equal')
    legend_handles = [
        Line2D([0], [0], color='g', lw=2, label=f'norm_v1={normalized[0]}'),
        Line2D([0], [0], color='m', lw=2, label=f'norm_v2={normalized[1]}'),
        Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Unit Circle')
    ]
    ax[1].legend(handles=legend_handles)
    ax[1].set_title('Normalized Vectors (on Unit Circle)')
    ax[1].axhline(0, color='k', lw=0.5)
    ax[1].axvline(0, color='k', lw=0.5)
    plt.tight_layout()
    plt.show()

def plot_matrix_analysis(M, name):
    """Plot matrix heatmap and singular values."""
    s = np.linalg.svd(M, compute_uv=False)
    rank = matrix_rank_estimate(M)
    cond_num = condition_number(M)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Matrix heatmap
    im = ax1.imshow(M, cmap='coolwarm', aspect='auto')
    ax1.set_title(f'{name}\nRank={rank}, Cond={cond_num:.2e}')
    plt.colorbar(im, ax=ax1)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax1.text(j, i, f'{M[i,j]:.2f}', ha='center', va='center', color='white' if abs(M[i,j]) > np.max(np.abs(M))/2 else 'black')
    
    # Singular values
    ax2.plot(range(1, len(s)+1), s, 'bo-', linewidth=2, markersize=8)
    ax2.set_title('Singular Values')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Value')
    ax2.grid(True)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.text(0.05, 0.95, f'Rank = {rank}\nCond = {cond_num:.2e}', transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()

# --- Enhanced Main Function ---

def main():
    """Test all functions with comprehensive examples and visualizations."""
    
    print("🔢 1. Vector Dot Product")
    v1 = np.array([3.0, 4.0])  # Use 2D for easy plotting
    v2 = np.array([4.0, -1.0])
    print("Vector dot product:", vector_dot(v1, v2))
    plot_dot_product(v1, v2)
    
    print("\n✖️ 2. Vector Outer Product")
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([4.0, 5.0])
    print("Vector outer product:\n", vector_outer(v1, v2))
    plot_outer_product(v1, v2)
    
    print("\n📦 3. Batch Dot Product")
    A_batch = np.random.randn(2, 3, 4)
    B_batch = np.random.randn(2, 4, 5)
    batch_result = batch_dot(A_batch, B_batch)
    print("Batch dot product shape:", batch_result.shape)
    plot_batch_dot_shapes(A_batch, B_batch, batch_result)
    
    print("\n📏 4. Vector Normalization")
    X = np.array([[3.0, 4.0], [1.0, 2.0]])
    normalized = normalize_vectors(X)
    print("Normalized vectors:\n", normalized)
    print("Norms of normalized vectors:", np.linalg.norm(normalized, axis=1))
    plot_normalization(X, normalized)
    
    print("\n📊 5. Matrix Rank & Condition Number")
    M_rank1 = np.array([[1.0, 2.0], [2.0, 4.0]])
    M_full_rank = np.eye(3)
    M_ill_cond = np.array([[1.0, 1.0], [1.0, 1.0000001]])
    
    matrices = [M_rank1, M_full_rank, M_ill_cond]
    names = ["Rank-deficient", "Well-conditioned", "Ill-conditioned"]
    
    for name, M in zip(names, matrices):
        rank = matrix_rank_estimate(M)
        cond_num = condition_number(M)
        print(f"{name} matrix:")
        print(f"  Rank: {rank}")
        print(f"  Condition number: {cond_num:.6e}")
        plot_matrix_analysis(M, name)

if __name__ == "__main__":
    main()