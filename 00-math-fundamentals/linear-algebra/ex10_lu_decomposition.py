import numpy as np
from typing import Tuple

def lu_decomposition_no_pivoting(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    LU Decomposition using Gaussian Elimination without Pivoting.
    
    This function factors a square matrix A into the product of:
        A = L * U
    where:
        L is a lower triangular matrix with 1's on the diagonal
        U is an upper triangular matrix
    
    Mathematical Basis:
    ---------------
    For a square matrix A of size n×n, we seek matrices L and U such that:
    
        [a₁₁ a₁₂ ... a₁ₙ]   [1    0  ... 0]   [u₁₁ u₁₂ ... u₁ₙ]
        [a₂₁ a₂₂ ... a₂ₙ] = [l₂₁  1  ... 0] × [0   u₂₂ ... u₂ₙ]
        [    ...        ]   [ ...     ... ]   [        ...     ]
        [aₙ₁ aₙ₂ ... aₙₙ]   [lₙ₁ lₙ₂ ... 1]   [0   0   ... uₙₙ]
    
    The decomposition is computed using Gaussian elimination where:
        lⱼᵢ = multiplier used to eliminate element aⱼᵢ
        uᵢⱼ = elements after elimination
    
    Parameters:
    -----------
    A : np.ndarray
        Square matrix to decompose (n×n)
    
    Returns:
    --------
    L : np.ndarray
        Lower triangular matrix with 1's on diagonal
    U : np.ndarray
        Upper triangular matrix
    
    Raises:
    -------
    ValueError:
        - If matrix A is not square
        - If A is singular (zero pivot encountered)
    """
    
    # Validate input matrix
    if not A.shape[0] == A.shape[1]:
        raise ValueError(f"Matrix must be square. Got shape {A.shape}")
    
    n = A.shape[0]
    
    # Initialize L as identity matrix and U as copy of A
    # L will store multipliers, U will be transformed to upper triangular
    L = np.eye(n, dtype=float)
    U = A.astype(float).copy()
    
    print("=" * 60)
    print("LU DECOMPOSITION (NO PIVOTING)")
    print("=" * 60)
    print(f"Input matrix A ({n}×{n}):\n{A}\n")
    print("Initial state:")
    print(f"L (identity matrix):\n{L}")
    print(f"U (copy of A):\n{U}")
    print("-" * 60)
    
    # Perform forward elimination (Gaussian elimination)
    print("\nFORWARD ELIMINATION PROCESS:")
    print("-" * 40)
    
    for i in range(n - 1):  # i is the pivot row index
        pivot = U[i, i]
        
        # Check for zero pivot (matrix is singular)
        if np.isclose(pivot, 0):
            raise ValueError(f"Zero pivot encountered at position ({i},{i}). "
                           "Matrix is singular or requires pivoting.")
        
        print(f"\nStep {i+1}: Pivot element U[{i},{i}] = {pivot:.4f}")
        print(f"Eliminating elements below pivot in column {i}")
        
        for j in range(i + 1, n):  # j are rows below pivot
            # Compute the multiplier for row elimination
            # This is the factor needed to zero out element (j,i)
            multiplier = U[j, i] / pivot
            
            # Store multiplier in L matrix
            L[j, i] = multiplier
            
            # Perform row operation: Rⱼ ← Rⱼ - multiplier × Rᵢ
            U[j, i:] = U[j, i:] - multiplier * U[i, i:]
            
            print(f"\n  Eliminating U[{j},{i}]:")
            print(f"    Multiplier l[{j},{i}] = {U[j, i] + multiplier * pivot:.4f} / {pivot:.4f} = {multiplier:.4f}")
            print(f"    Updated row {j}: R{j+1} ← R{j+1} - {multiplier:.4f} × R{i+1}")
            print(f"    Intermediate U:\n{U}")
    
    print("\n" + "=" * 60)
    print("DECOMPOSITION COMPLETE")
    print("=" * 60)
    
    return L, U

def verify_decomposition(A: np.ndarray, L: np.ndarray, U: np.ndarray) -> None:
    """
    Verify the correctness of LU decomposition.
    
    This function:
    1. Checks if A = L × U
    2. Verifies L is lower triangular with 1's on diagonal
    3. Verifies U is upper triangular
    4. Computes reconstruction error
    
    Parameters:
    -----------
    A : np.ndarray
        Original matrix
    L : np.ndarray
        Lower triangular matrix from decomposition
    U : np.ndarray
        Upper triangular matrix from decomposition
    """
    
    print("\nVERIFICATION")
    print("-" * 40)
    
    # 1. Check matrix multiplication
    A_reconstructed = L @ U
    reconstruction_error = np.max(np.abs(A - A_reconstructed))
    
    print(f"1. Matrix reconstruction:")
    print(f"   L × U =\n{A_reconstructed}")
    print(f"   Maximum reconstruction error: {reconstruction_error:.2e}")
    
    # 2. Check properties of L
    print(f"\n2. Properties of L:")
    print(f"   Is lower triangular: {np.allclose(L, np.tril(L))}")
    print(f"   Has 1's on diagonal: {np.allclose(np.diag(L), 1)}")
    print(f"   L =\n{L}")
    
    # 3. Check properties of U
    print(f"\n3. Properties of U:")
    print(f"   Is upper triangular: {np.allclose(U, np.triu(U))}")
    print(f"   U =\n{U}")
    
    # 4. Compare with numpy's built-in LU
    try:
        P, L_np, U_np = scipy.linalg.lu(A)
        print(f"\n4. Comparison with SciPy's LU (with pivoting):")
        print(f"   SciPy uses permutation matrix P for numerical stability")
    except ImportError:
        print("\nNote: Install SciPy for comparison with professional implementation")

def solve_linear_system(L: np.ndarray, U: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve linear system A·x = b using LU decomposition.
    
    Since A = L·U, the system becomes:
        L·U·x = b
    
    We solve this in two steps:
    1. Forward substitution: Solve L·y = b for y
    2. Backward substitution: Solve U·x = y for x
    
    Parameters:
    -----------
    L : np.ndarray
        Lower triangular matrix from LU decomposition
    U : np.ndarray
        Upper triangular matrix from LU decomposition
    b : np.ndarray
        Right-hand side vector
    
    Returns:
    --------
    x : np.ndarray
        Solution vector
    """
    
    n = len(b)
    y = np.zeros(n)
    x = np.zeros(n)
    
    print("\nSOLVING LINEAR SYSTEM L·U·x = b")
    print("-" * 40)
    print(f"Right-hand side b = {b}")
    
    # Forward substitution: Solve L·y = b
    print("\n1. Forward substitution (solve L·y = b):")
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
        print(f"   y[{i}] = b[{i}] - Σ L[{i},:{i}]·y[:{i}] = {y[i]:.4f}")
    
    # Backward substitution: Solve U·x = y
    print("\n2. Backward substitution (solve U·x = y):")
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
        print(f"   x[{i}] = (y[{i}] - Σ U[{i},{i+1}:]·x[{i+1}:]) / U[{i},{i}] = {x[i]:.4f}")
    
    return x

def main():
    """
    Main function demonstrating LU decomposition with examples.
    """
    
    # Example 1: 2×2 matrix (from original code)
    print("EXAMPLE 1: 2×2 Matrix")
    A1 = np.array([[2, 3], [4, 7]], dtype=float)
    
    try:
        L1, U1 = lu_decomposition_no_pivoting(A1)
        verify_decomposition(A1, L1, U1)
        
        # Demonstrate solving linear system
        b1 = np.array([8, 18], dtype=float)  # Solution should be x = [1, 2]
        x1 = solve_linear_system(L1, U1, b1)
        print(f"\n3. Solution verification:")
        print(f"   Computed x = {x1}")
        print(f"   A·x = {A1 @ x1} (should equal b = {b1})")
        print(f"   Error: {np.max(np.abs(A1 @ x1 - b1)):.2e}")
        
    except ValueError as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 60)
    
    # Example 2: 3×3 matrix
    print("\n\nEXAMPLE 2: 3×3 Matrix")
    A2 = np.array([[1, 2, 3], [2, 5, 8], [3, 8, 14]], dtype=float)
    
    try:
        L2, U2 = lu_decomposition_no_pivoting(A2)
        verify_decomposition(A2, L2, U2)
    except ValueError as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 60)
    
    # Example 3: Matrix requiring pivoting (will fail)
    print("\n\nEXAMPLE 3: Matrix Requiring Pivoting")
    A3 = np.array([[0, 1], [1, 0]], dtype=float)
    
    try:
        L3, U3 = lu_decomposition_no_pivoting(A3)
        verify_decomposition(A3, L3, U3)
    except ValueError as e:
        print(f"Expected error: {e}")
        print("\nThis demonstrates why pivoting is necessary for numerical stability!")
        print("Pivoting rearranges rows to avoid zero or small pivots.")

if __name__ == "__main__":
    # Try to import SciPy for comparison (optional)
    try:
        import scipy.linalg
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False
        print("Note: SciPy not installed. Some features disabled.")
    
    main()