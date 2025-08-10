"""
ex08_solve_linear_systems.py
---------------------------
In this implementation, I demonstrate how to solve linear systems of equations Ax = b using
Gaussian Elimination with Back Substitution. This fundamental approach underpins many
computational methods in machine learning, including least squares regression, matrix
factorizations, and optimization algorithms.

Linear system solving is ubiquitous in ML applications:
- Solving normal equations in linear regression
- Computing matrix inverses for covariance calculations
- Eigenvalue problems in PCA and spectral methods
- Optimization algorithms requiring linear solver subroutines

This educational implementation provides insight into the computational mechanics that
power more sophisticated numerical libraries, while highlighting numerical considerations
crucial for production ML systems.

Notes
-----
While this implementation prioritizes clarity over performance, production ML pipelines
typically use optimized LAPACK-based solvers (via numpy.linalg.solve) that include
sophisticated pivoting strategies and numerical stability enhancements.
"""

import numpy as np

def gaussian_elimination_with_backsubstitution(A, b, verbose=True):
    """
    Solve the linear system Ax = b using Gaussian elimination and back substitution.
    
    Parameters
    ----------
    A : np.ndarray
        Coefficient matrix of shape (n, n).
    b : np.ndarray
        Right-hand side vector of shape (n,).
    verbose : bool, optional
        If True, print intermediate steps for educational purposes.
        
    Returns
    -------
    np.ndarray
        Solution vector x of shape (n,).
        
    Notes
    -----
    This implementation demonstrates the fundamental algorithm without pivoting.
    For numerical stability in production code, partial or complete pivoting
    should be implemented to handle near-zero pivot elements.
    
    The algorithm complexity is O(n³) for the elimination phase and O(n²) for
    back substitution, making it suitable for small to medium-sized systems.
    """
    # Create augmented matrix [A|b] to apply row operations simultaneously
    # Using float64 ensures adequate numerical precision for the computations
    A_work = A.astype(np.float64)
    b_work = b.astype(np.float64)
    
    # Construct augmented matrix for Gaussian elimination
    Ab = np.hstack((A_work, b_work.reshape(-1, 1)))
    n = Ab.shape[0]
    
    if verbose:
        print("Initial augmented matrix [A|b]:\n", Ab)
        print()
    
    # Forward elimination: transform to upper triangular form
    for i in range(n):
        # Pivot selection: check for numerical stability
        if abs(Ab[i, i]) < 1e-12:
            raise ValueError(
                f"Near-zero pivot {Ab[i, i]:.2e} at position ({i}, {i}). "
                "Consider implementing pivoting for numerical stability."
            )
        
        # Eliminate entries below the current pivot
        for j in range(i + 1, n):
            # Compute elimination multiplier: how much to subtract from row j
            multiplier = Ab[j, i] / Ab[i, i]
            
            # Apply row operation: row_j ← row_j - multiplier × row_i
            Ab[j] -= multiplier * Ab[i]
            
            if verbose:
                print(f"Row {j} ← Row {j} - {multiplier:.4f} × Row {i}")
                print(f"Updated augmented matrix:\n{Ab}\n")
    
    # Back substitution: solve for unknowns from bottom to top
    x = np.zeros(n)
    
    for i in range(n - 1, -1, -1):
        # Solve for x[i]: isolate variable from the linear equation
        # Ab[i, -1] represents the RHS after elimination
        # Ab[i, i+1:n] contains coefficients of already-solved variables
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i + 1:n], x[i + 1:n])) / Ab[i, i]
        
        if verbose:
            print(f"x[{i}] = ({Ab[i, -1]:.4f} - Σ(coeffs × known_vars)) / {Ab[i, i]:.4f} = {x[i]:.6f}")
    
    return x


# Define a well-conditioned test system for demonstration
# This system is chosen to have a unique solution and reasonable numerical properties
A = np.array([
    [2, 1, -1],   # First equation:  2x₁ + x₂ - x₃ = 8
    [-3, -1, 2],  # Second equation: -3x₁ - x₂ + 2x₃ = -11
    [-2, 1, 2]    # Third equation:  -2x₁ + x₂ + 2x₃ = -3
], dtype=float)

b = np.array([8, -11, -3], dtype=float)

print("=== Linear System Ax = b ===")
print("Coefficient matrix A:")
print(A)
print("\nRight-hand side vector b:")
print(b)
print("\n" + "="*50)

# Solve using our educational implementation
print("\n=== Solution via Custom Gaussian Elimination ===")
x_custom = gaussian_elimination_with_backsubstitution(A, b, verbose=True)

print("\n=== Final Results ===")
print(f"Solution vector x: {x_custom}")

# Verification: compute residual to validate solution accuracy
residual = A @ x_custom - b
residual_norm = np.linalg.norm(residual)
print(f"\nSolution verification:")
print(f"Residual ||Ax - b||₂ = {residual_norm:.2e}")

if residual_norm < 1e-10:
    print("✓ Solution is numerically accurate")
else:
    print("⚠ Solution may have numerical errors")

# Comparison with NumPy's optimized solver
print("\n=== Comparison with NumPy Implementation ===")
x_numpy = np.linalg.solve(A, b)
print(f"NumPy solution:        {x_numpy}")
print(f"Custom solution:       {x_custom}")
print(f"Difference norm:       {np.linalg.norm(x_numpy - x_custom):.2e}")

# Educational note on computational complexity and practical considerations
print(f"\n=== Computational Insights ===")
print(f"Matrix size: {A.shape[0]}×{A.shape[1]}")
print(f"Theoretical operations: ~{(A.shape[0]**3)/3:.0f} (O(n³/3) for elimination)")
print(f"Memory usage: O(n²) for matrix storage")
print("\nIn production ML pipelines:")
print("• Use np.linalg.solve() for better numerical stability")
print("• Consider iterative methods for large sparse systems")
print("• Leverage GPU acceleration for massive linear systems")
