import numpy as np
from typing import Tuple

"""
QR Decomposition: The Linear Algebra Glow-Up You Didn't Know You Needed

Think of QR decomposition as giving your matrix a total makeover. 
We're breaking down any matrix A into Q (the cool, orthonormal one) 
and R (the structured, triangular one). This isn't just math - it's 
giving your numerical algorithms main character energy.

Real ones use this for:
• Solving least squares problems (no cap)
• Computing eigenvalues (the matrix's personality traits)
• Making algorithms numerically stable (vibes check passed)
"""

def classical_gram_schmidt(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implements Classical Gram-Schmidt orthogonalization.
    
    Low-key warning: This algorithm can be numerically unstable 
    for poorly conditioned matrices. For serious applications, 
    modified Gram-Schmidt or Householder reflections are the move.
    
    Parameters
    ----------
    A : np.ndarray
        Input matrix to decompose (m × n, with m ≥ n)
    
    Returns
    -------
    Q : np.ndarray
        Orthonormal matrix (columns are orthonormal vectors)
    R : np.ndarray
        Upper triangular matrix
    
    Raises
    ------
    ValueError
        If matrix columns are linearly dependent (not full rank)
    """
    m, n = A.shape
    
    # Initialize Q (will hold our orthonormal glow-up vectors)
    Q = np.zeros((m, n), dtype=A.dtype)
    
    # R starts as all zeros - will fill upper triangle only
    R = np.zeros((n, n), dtype=A.dtype)
    
    # The main slay: orthogonalize each column
    for i in range(n):
        # Start with the original column (no edits yet)
        v = A[:, i].copy()
        
        # Project out components along previous orthonormal directions
        # This is where we ensure "mind your own business" orthogonality
        for j in range(i):
            # Compute how much of v is in Q[:, j] direction
            R[j, i] = Q[:, j] @ v  # Dot product, but make it Pythonic
            # Subtract that projection - keeping it independent
            v -= R[j, i] * Q[:, j]
        
        # Normalize to get that unit vector aesthetic
        norm_v = np.linalg.norm(v)
        
        if norm_v < 1e-10:
            raise ValueError(
                f"Column {i} is linearly dependent on previous columns. "
                "Matrix needs to have full rank for this decomposition."
            )
        
        R[i, i] = norm_v
        Q[:, i] = v / norm_v  # Normalization = vector glow-up
    
    return Q, R


def validate_decomposition(A: np.ndarray, Q: np.ndarray, R: np.ndarray) -> None:
    """
    Validates QR decomposition with absolute drama.
    
    Performs three critical checks:
    1. Reconstruction accuracy (did we lose the original vibe?)
    2. Orthonormality (are Q's columns actually serving?)
    3. Triangular structure (is R keeping it structured?)
    """
    print("=" * 60)
    print("QR DECOMPOSITION VALIDATION - THE RECKONING")
    print("=" * 60)
    
    # Check 1: Can we reconstruct the original matrix?
    A_reconstructed = Q @ R
    recon_error = np.linalg.norm(A - A_reconstructed, 'fro')
    rel_error = recon_error / np.linalg.norm(A, 'fro')
    
    print(f"\n✅ Reconstruction Check:")
    print(f"   Original A shape: {A.shape}")
    print(f"   Absolute reconstruction error: {recon_error:.2e}")
    print(f"   Relative reconstruction error: {recon_error:.2e}")
    
    # Check 2: Is Q actually orthonormal? (QᵀQ should be identity)
    Q_squared = Q.T @ Q
    identity_target = np.eye(Q.shape[1])
    ortho_error = np.linalg.norm(Q_squared - identity_target, 'fro')
    
    print(f"\n✅ Orthonormality Check:")
    print(f"   QᵀQ (should be identity):")
    print(f"   {Q_squared}")
    print(f"   Orthogonality error: {ortho_error:.2e}")
    
    # Check 3: Is R actually upper triangular?
    lower_tri_entries = np.sum(np.abs(np.tril(R, -1)))
    
    print(f"\n✅ Triangular Structure Check:")
    print(f"   R shape: {R.shape}")
    print(f"   Sum of lower triangle (should be ~0): {lower_tri_entries:.2e}")
    
    if rel_error < 1e-10 and ortho_error < 1e-10 and lower_tri_entries < 1e-10:
        print("\n🎯 ALL CHECKS PASSED - DECOMPOSITION IS VALID")
    else:
        print("\n⚠️  WARNING: Some checks failed - numerical instability possible")
    
    print("=" * 60)


def main():
    """
    Main demonstration - because seeing is believing.
    """
    # Our test matrix - chosen to be interesting but not trivial
    A = np.array([
        [2, 7, 8],
        [4, 5, 9],
        [3, 8, 2]
    ], dtype=float)
    
    print("Original Matrix A (before the glow-up):")
    print(A)
    print(f"\nMatrix shape: {A.shape}")
    print(f"Condition number: {np.linalg.cond(A):.2f}")
    
    try:
        # Perform the decomposition
        Q, R = classical_gram_schmidt(A)
        
        print("\n" + "=" * 60)
        print("DECOMPOSITION RESULTS")
        print("=" * 60)
        
        print("\nQ Matrix (orthonormal columns - the main character):")
        print(Q)
        
        print("\nR Matrix (upper triangular - the supporting role):")
        print(R)
        
        # Run the validation suite
        validate_decomposition(A, Q, R)
        
    except ValueError as e:
        print(f"\n❌ Decomposition failed: {e}")
        print("Try a matrix with linearly independent columns!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    # Let's run this thing
    main()