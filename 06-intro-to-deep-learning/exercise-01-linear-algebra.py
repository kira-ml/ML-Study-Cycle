import numpy as np
from typing import Union, Optional

def vector_dot(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the dot product of two 1D vectors.
    
    Parameters:
    -----------
    a : np.ndarray
        First 1D vector
    b : np.ndarray
        Second 1D vector
        
    Returns:
    --------
    float
        Dot product of a and b
        
    Raises:
    -------
    ValueError
        If inputs are not 1D vectors or have incompatible shapes
    """
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Inputs must be 1D vectors")
    
    if a.shape != b.shape:
        raise ValueError(f"Vectors must have same length. Got {a.shape} and {b.shape}")
    
    return float(a @ b)

def vector_outer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the outer product of two 1D vectors.
    
    Parameters:
    -----------
    a : np.ndarray
        First 1D vector
    b : np.ndarray
        Second 1D vector
        
    Returns:
    --------
    np.ndarray
        Outer product matrix of shape (len(a), len(b))
        
    Raises:
    -------
    ValueError
        If inputs are not 1D vectors
    """
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Inputs must be 1D vectors")
    
    return np.outer(a, b)

def batch_dot(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute batch matrix multiplication for 3D arrays.
    
    Parameters:
    -----------
    A : np.ndarray
        3D array of shape (batch, m, n)
    B : np.ndarray
        3D array of shape (batch, n, p)
        
    Returns:
    --------
    np.ndarray
        Batch matrix product of shape (batch, m, p)
        
    Raises:
    -------
    ValueError
        If inputs are not 3D arrays or have incompatible shapes
    """
    if A.ndim != 3 or B.ndim != 3:
        raise ValueError("Inputs must be 3D arrays")
    
    if A.shape[0] != B.shape[0]:
        raise ValueError(f"Batch sizes must match. Got {A.shape[0]} and {B.shape[0]}")
    
    if A.shape[2] != B.shape[1]:
        raise ValueError(f"Inner dimensions must match. Got {A.shape[2]} and {B.shape[1]}")
    
    return A @ B

def normalize_vectors(X: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize vectors along specified axis.
    
    Parameters:
    -----------
    X : np.ndarray
        Input array containing vectors to normalize
    axis : int
        Axis along which to compute norms (default: -1)
    eps : float
        Small constant to avoid division by zero
        
    Returns:
    --------
    np.ndarray
        Normalized array with same shape as X
    """
    norm = np.linalg.norm(X, axis=axis, keepdims=True)
    # Handle zero-norm vectors
    norm = np.where(norm < eps, 1.0, norm)
    return X / norm

def matrix_rank_estimate(M: np.ndarray, tol: Optional[float] = None) -> int:
    """
    Estimate matrix rank using singular value decomposition.
    
    Parameters:
    -----------
    M : np.ndarray
        Input matrix
    tol : float, optional
        Tolerance for considering singular values as non-zero.
        If None, uses machine precision-based tolerance.
        
    Returns:
    --------
    int
        Estimated rank of matrix
    """
    if tol is None:
        # Use machine precision-based tolerance
        tol = max(M.shape) * np.finfo(M.dtype).eps * np.linalg.norm(M, ord=2)
    
    s = np.linalg.svd(M, compute_uv=False)
    return np.sum(s > tol)

def condition_number(M: np.ndarray) -> float:
    """
    Compute the condition number of a matrix (ratio of largest to smallest singular value).
    
    Parameters:
    -----------
    M : np.ndarray
        Input matrix
        
    Returns:
    --------
    float
        Condition number of the matrix
        
    Raises:
    -------
    LinAlgError
        If matrix is singular or SVD fails
    """
    s = np.linalg.svd(M, compute_uv=False)
    if np.any(s == 0):
        return float('inf')
    return float(s[0] / s[-1])

def main():
    """Test all functions with comprehensive examples."""
    
    # Test vector operations
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([4.0, 5.0, 6.0])
    
    print("Vector dot product:", vector_dot(v1, v2))
    print("Vector outer product:\n", vector_outer(v1, v2))
    
    # Test batch operations
    A_batch = np.random.randn(2, 3, 4)
    B_batch = np.random.randn(2, 4, 5)
    
    batch_result = batch_dot(A_batch, B_batch)
    print("Batch dot product shape:", batch_result.shape)
    
    # Test normalization
    X = np.array([[3.0, 4.0], [1.0, 2.0]])  # Fixed array creation
    normalized = normalize_vectors(X)
    print("Normalized vectors:\n", normalized)
    print("Norms of normalized vectors:", np.linalg.norm(normalized, axis=1))
    
    # Test matrix rank and condition number
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

if __name__ == "__main__":
    main()