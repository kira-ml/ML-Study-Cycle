"""
ex06_determinant_calculation.py
------------------------------
In this implementation, I demonstrate how to compute matrix determinants using both NumPy's 
optimized linear algebra routines and manual cofactor expansion. This educational approach 
illustrates the mathematical foundations underlying determinant computation, which is crucial 
for understanding matrix invertibility, eigenvalue calculations, and geometric transformations 
in machine learning applications.

The determinant is a scalar value that encodes important properties of a matrix:
- Non-zero determinant indicates matrix invertibility (essential for solving linear systems)
- Magnitude represents scaling factor of linear transformations
- Sign indicates orientation preservation/reversal

This implementation serves both pedagogical and practical purposes, providing insights into
numerical linear algebra that are valuable for ML engineers working with matrix operations.
"""

import numpy as np

# Define sample matrices for determinant computation
# These matrices are chosen to demonstrate different computational approaches
A = np.array([[1, 2],
              [3, 4]])  # 2x2 matrix for direct formula application

B = np.array([
    [1, 2, 3],
    [0, 1, 4],  # Strategic zero placement simplifies manual expansion
    [5, 6, 0]
])  # 3x3 matrix for cofactor expansion demonstration


def det_2x2(mat):
    """
    Compute the determinant of a 2x2 matrix using the closed-form formula.
    
    Parameters
    ----------
    mat : np.ndarray
        A 2x2 NumPy array representing the matrix.
        
    Returns
    -------
    float
        The determinant value computed as ad - bc for matrix [[a,b],[c,d]].
        
    Notes
    -----
    This is the base case for recursive determinant computation and demonstrates
    the fundamental relationship between matrix elements and their determinant.
    In ML contexts, 2x2 determinants frequently appear in 2D transformations
    and covariance matrix analysis.
    """
    return mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]


# Example 1: 2x2 Determinant Computation
# Demonstrate both manual and NumPy approaches for comparison
A_det_numpy = np.linalg.det(A)
A_det_manual = det_2x2(A)

print("Matrix A:\n", A)
print(f"Determinant of A (NumPy): {A_det_numpy:.6f}")
print(f"Determinant of A (manual): {A_det_manual:.6f}")
print(f"Computation difference: {abs(A_det_numpy - A_det_manual):.2e}\n")


# Example 2: 3x3 Determinant via Cofactor Expansion
# This manual implementation illustrates the recursive structure of determinant computation
# and provides insight into the computational complexity of larger matrices.

print("=" * 50)
print("3x3 Determinant Computation via Cofactor Expansion")
print("=" * 50)

det_accumulator = 0  # Accumulator for determinant computation

# Perform Laplace expansion along the first row
# This choice is arbitrary; any row or column could be used
for col_idx in range(B.shape[1]):
    # Compute cofactor sign: (-1)^(row + col) for position (0, col_idx)
    cofactor_sign = (-1) ** col_idx
    
    # Extract the (n-1)x(n-1) minor by removing row 0 and column col_idx
    # This step reduces the problem size, demonstrating the recursive nature
    minor_matrix = np.delete(np.delete(B, 0, axis=0), col_idx, axis=1)
    
    # Compute determinant of the 2x2 minor
    minor_determinant = det_2x2(minor_matrix)
    
    # Current cofactor contribution to the overall determinant
    cofactor_contribution = cofactor_sign * B[0, col_idx] * minor_determinant
    
    # Educational output: show the step-by-step computation
    print(f"Column {col_idx}: Element = {B[0, col_idx]}")
    print(f"Minor matrix:\n{minor_matrix}")
    print(f"Minor determinant: {minor_determinant}")
    print(f"Cofactor contribution: {cofactor_sign} × {B[0, col_idx]} × {minor_determinant} = {cofactor_contribution}")
    
    det_accumulator += cofactor_contribution
    print(f"Running total: {det_accumulator}\n")

print("Matrix B:\n", B)
print(f"Determinant of B (manual cofactor expansion): {det_accumulator}")

# Validation against NumPy's optimized implementation
# NumPy uses LU decomposition for efficient determinant computation
B_det_numpy = np.linalg.det(B)
print(f"Determinant of B (NumPy): {B_det_numpy:.6f}")
print(f"Numerical difference: {abs(det_accumulator - B_det_numpy):.2e}")

# Note: In production ML pipelines, NumPy's implementation should be preferred
# due to its numerical stability and computational efficiency
