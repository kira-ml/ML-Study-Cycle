"""
ex08_solve_linear_systems.py
===========================
Educational implementation of Gaussian Elimination with Back Substitution.

This module demonstrates fundamental linear algebra operations essential for
machine learning algorithms including linear regression, PCA, and optimization.

Core Concepts:
- Direct methods for solving linear systems
- Numerical stability considerations
- Algorithmic complexity analysis

Applications in ML:
- Solving normal equations in least squares
- Covariance matrix operations
- Feature space transformations
"""

import numpy as np
from typing import Tuple, Optional


def forward_elimination(A: np.ndarray, b: np.ndarray, 
                       verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Gaussian elimination to transform [A|b] to upper triangular form.
    
    Args:
        A: Coefficient matrix (n x n)
        b: Right-hand side vector (n,)
        verbose: Whether to display elimination steps
        
    Returns:
        U: Upper triangular matrix
        b_elim: Modified right-hand side
        
    Raises:
        ValueError: If pivot element is numerically zero
    """
    n = len(b)
    U = A.astype(np.float64).copy()
    b_elim = b.astype(np.float64).copy()
    
    if verbose:
        print("\n--- Forward Elimination Phase ---")
        print(f"Initial system:\n{np.column_stack((U, b_elim.reshape(-1, 1)))}")
    
    for col in range(n - 1):  # Pivot column
        pivot = U[col, col]
        
        # Numerical stability check
        if np.abs(pivot) < 1e-10:
            raise ValueError(f"Zero pivot at position ({col},{col}). "
                           "Partial pivoting required for stability.")
        
        # Eliminate below current pivot
        for row in range(col + 1, n):
            factor = U[row, col] / pivot
            
            # Vectorized row operation: row ← row - factor × pivot_row
            U[row, col:] -= factor * U[col, col:]
            b_elim[row] -= factor * b_elim[col]
            
            if verbose:
                print(f"\nEliminating row {row} using pivot row {col}:")
                print(f"Factor: {factor:.4f}")
                print(f"Updated augmented matrix:\n"
                      f"{np.column_stack((U, b_elim.reshape(-1, 1)))}")
    
    return U, b_elim


def back_substitution(U: np.ndarray, b: np.ndarray, 
                     verbose: bool = False) -> np.ndarray:
    """
    Solve upper triangular system Ux = b using backward substitution.
    
    Args:
        U: Upper triangular matrix (n x n)
        b: Modified right-hand side vector (n,)
        verbose: Whether to display substitution steps
        
    Returns:
        x: Solution vector
    """
    n = len(b)
    x = np.zeros(n, dtype=np.float64)
    
    if verbose:
        print("\n--- Back Substitution Phase ---")
    
    # Solve from last equation to first
    for i in range(n - 1, -1, -1):
        # Known terms from already solved variables
        known_terms = U[i, i+1:] @ x[i+1:] if i < n - 1 else 0
        
        # Solve: U[i,i] * x[i] = b[i] - Σ(U[i,j] * x[j]) for j > i
        x[i] = (b[i] - known_terms) / U[i, i]
        
        if verbose:
            print(f"Equation {i}: {U[i, i]:.4f}·x[{i}] = "
                  f"{b[i]:.4f} - {known_terms:.4f}")
            print(f"  → x[{i}] = {x[i]:.6f}")
    
    return x


def solve_linear_system(A: np.ndarray, b: np.ndarray, 
                       verbose: bool = False, 
                       enable_pivoting: bool = False) -> np.ndarray:
    """
    Solve linear system Ax = b using Gaussian elimination.
    
    Args:
        A: Square coefficient matrix (n x n)
        b: Right-hand side vector (n,)
        verbose: Display algorithmic steps
        enable_pivoting: Experimental partial pivoting
        
    Returns:
        x: Solution vector satisfying Ax = b
        
    Example:
        >>> A = np.array([[2, 1], [1, 2]], dtype=float)
        >>> b = np.array([5, 4], dtype=float)
        >>> x = solve_linear_system(A, b)
        >>> np.allclose(A @ x, b)
        True
    """
    # Input validation
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Matrix must be square, got shape {A.shape}")
    if A.shape[0] != len(b):
        raise ValueError(f"Dimension mismatch: A={A.shape}, b={len(b)}")
    
    n = A.shape[0]
    
    # Experimental partial pivoting (optional)
    if enable_pivoting:
        A_work, b_work = _partial_pivot(A.copy(), b.copy())
    else:
        A_work, b_work = A.copy(), b.copy()
    
    if verbose:
        print("="*60)
        print("GAUSSIAN ELIMINATION ALGORITHM")
        print(f"System size: {n} equations, {n} unknowns")
        print(f"Total operations: ~{2*n**3/3:.0f} flops (O(n³))")
        print("="*60)
    
    # Phase 1: Forward elimination to upper triangular form
    U, b_mod = forward_elimination(A_work, b_work, verbose)
    
    # Phase 2: Back substitution to obtain solution
    x = back_substitution(U, b_mod, verbose)
    
    return x


def _partial_pivot(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Experimental implementation of partial pivoting.
    
    For each column, swap current row with row having maximum pivot value
    to improve numerical stability.
    """
    n = len(b)
    
    for col in range(n - 1):
        # Find row with maximum absolute value in current column
        max_row = np.argmax(np.abs(A[col:, col])) + col
        
        if max_row != col:
            # Swap rows in A and b
            A[[col, max_row]] = A[[max_row, col]]
            b[[col, max_row]] = b[[max_row, col]]
    
    return A, b


def verify_solution(A: np.ndarray, b: np.ndarray, x: np.ndarray, 
                   method: str = "custom") -> dict:
    """
    Compute verification metrics for linear system solution.
    
    Returns:
        Dictionary containing:
        - residual: Ax - b
        - relative_error: ||Ax - b|| / ||b||
        - condition_number: κ(A)
    """
    residual = A @ x - b
    residual_norm = np.linalg.norm(residual)
    b_norm = np.linalg.norm(b)
    cond_A = np.linalg.cond(A)
    
    return {
        "method": method,
        "solution": x,
        "residual_norm": residual_norm,
        "relative_error": residual_norm / b_norm if b_norm > 0 else np.inf,
        "condition_number": cond_A,
        "is_accurate": residual_norm < 1e-8 * max(1.0, b_norm)
    }


def compare_solvers(A: np.ndarray, b: np.ndarray, verbose: bool = False):
    """
    Educational comparison between custom and NumPy implementations.
    """
    print("="*60)
    print("LINEAR SYSTEM SOLVER COMPARISON")
    print("="*60)
    
    # Display problem
    print(f"\nSystem: {A.shape[0]} equations with {A.shape[1]} unknowns")
    print(f"\nCoefficient matrix A (κ={np.linalg.cond(A):.2e}):")
    print(A)
    print(f"\nRight-hand side b:")
    print(b)
    
    # Custom implementation
    print("\n" + "-"*40)
    print("CUSTOM IMPLEMENTATION")
    print("-"*40)
    try:
        x_custom = solve_linear_system(A, b, verbose=verbose)
        result_custom = verify_solution(A, b, x_custom, "custom")
        
        print(f"\nSolution: {x_custom}")
        print(f"Residual ||Ax - b||₂ = {result_custom['residual_norm']:.2e}")
        print(f"Relative error = {result_custom['relative_error']:.2e}")
        if result_custom['is_accurate']:
            print("✓ Solution meets accuracy criteria")
    except ValueError as e:
        print(f"✗ Custom solver failed: {e}")
        result_custom = None
    
    # NumPy reference implementation
    print("\n" + "-"*40)
    print("NUMPY REFERENCE (np.linalg.solve)")
    print("-"*40)
    try:
        x_numpy = np.linalg.solve(A, b)
        result_numpy = verify_solution(A, b, x_numpy, "numpy")
        
        print(f"\nSolution: {x_numpy}")
        print(f"Residual ||Ax - b||₂ = {result_numpy['residual_norm']:.2e}")
        print(f"Relative error = {result_numpy['relative_error']:.2e}")
        
        if result_custom is not None:
            diff_norm = np.linalg.norm(x_custom - x_numpy)
            print(f"\nDifference between solvers: {diff_norm:.2e}")
    except np.linalg.LinAlgError as e:
        print(f"✗ NumPy solver failed: {e}")
    
    # Educational insights
    print("\n" + "="*60)
    print("COMPUTATIONAL INSIGHTS")
    print("="*60)
    
    n = A.shape[0]
    print(f"\nAlgorithmic Complexity:")
    print(f"• Gaussian elimination: ~{2*n**3/3:.0f} floating-point operations")
    print(f"• Memory: O({n**2}) for matrix storage")
    print(f"• Condition number: κ(A) = {np.linalg.cond(A):.2e}")
    
    print(f"\nPractical Recommendations:")
    print("• Use np.linalg.solve() for production (implements LAPACK)")
    print("• For large sparse systems: scipy.sparse.linalg.spsolve")
    print("• For ill-conditioned systems: regularization or SVD-based methods")


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

if __name__ == "__main__":
    # Well-conditioned test system from original example
    A_demo = np.array([
        [2, 1, -1],
        [-3, -1, 2],
        [-2, 1, 2]
    ], dtype=float)
    
    b_demo = np.array([8, -11, -3], dtype=float)
    
    # Run educational comparison
    compare_solvers(A_demo, b_demo, verbose=True)
    
    # Additional educational examples
    print("\n\n" + "="*60)
    print("ADDITIONAL EDUCATIONAL EXAMPLES")
    print("="*60)
    
    # Example 2: 2x2 system
    print("\nExample 2: 2x2 System")
    A2 = np.array([[3, 1], [1, 2]], dtype=float)
    b2 = np.array([9, 8], dtype=float)
    x2 = solve_linear_system(A2, b2, verbose=False)
    print(f"Solution: {x2}")
    print(f"Verification: A@x = {A2 @ x2}, Expected: {b2}")
    
    # Example 3: Diagonal system (trivial case)
    print("\nExample 3: Diagonal System")
    A3 = np.diag([1, 2, 3, 4])
    b3 = np.array([1, 2, 3, 4], dtype=float)
    x3 = solve_linear_system(A3, b3, verbose=False)
    print(f"Solution: {x3}")
    
    # Performance note
    print("\n" + "-"*60)
    print("PERFORMANCE NOTE:")
    print("This educational implementation prioritizes clarity over speed.")
    print("For n=1000, np.linalg.solve is ~1000x faster due to:")
    print("• Optimized BLAS/LAPACK libraries")
    print("• Cache-efficient memory access patterns")
    print("• Multi-threaded execution")
    print("-"*60)