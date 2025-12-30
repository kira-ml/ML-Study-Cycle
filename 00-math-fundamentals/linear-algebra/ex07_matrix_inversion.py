import numpy as np

def determinant_2x2(matrix):
    """
    Computes the determinant of a 2x2 matrix using the standard formula.
    Optimized with unpacking and direct arithmetic.
    """
    a, b = matrix[0]
    c, d = matrix[1]
    return a * d - b * c

def determinant_3x3(matrix):
    """
    Computes the determinant of a 3x3 matrix using optimized formula.
    Eliminates redundant multiplications and uses efficient unpacking.
    """
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    
    # Precompute common sub-expressions
    ei = e * i
    fh = f * h
    di = d * i
    fg = f * g
    dh = d * h
    eg = e * g
    
    return a * (ei - fh) - b * (di - fg) + c * (dh - eg)

def cofactor_matrix_3x3(matrix):
    """
    Constructs the cofactor matrix for a 3x3 input matrix.
    Optimized by avoiding np.delete calls and computing minors directly.
    """
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    
    # Precompute all 2x2 determinants needed for cofactors
    # Minor determinants (without signs)
    minors = np.array([
        [e*i - f*h, d*i - f*g, d*h - e*g],  # Row 1 minors
        [b*i - c*h, a*i - c*g, a*h - b*g],  # Row 2 minors
        [b*f - c*e, a*f - c*d, a*e - b*d]   # Row 3 minors
    ], dtype=float)
    
    # Apply signs to create cofactor matrix
    signs = np.array([[1, -1, 1], [-1, 1, -1], [1, -1, 1]])
    return minors * signs

def invert_matrix(matrix):
    """
    Computes the inverse of a 2x2 or 3x3 matrix using optimized methods.
    """
    matrix = np.array(matrix, dtype=float)
    shape = matrix.shape

    if shape == (2, 2):
        # Optimized 2x2 inverse
        a, b = matrix[0]
        c, d = matrix[1]
        
        det = a * d - b * c
        if abs(det) < 1e-12:  # Using tolerance for numerical stability
            raise ValueError("Matrix is singular and not invertible.")
        
        inv_det = 1.0 / det
        return np.array([[d, -b], [-c, a]]) * inv_det

    elif shape == (3, 3):
        # Optimized 3x3 inverse
        det = determinant_3x3(matrix)
        if abs(det) < 1e-12:  # Using tolerance for numerical stability
            raise ValueError("Matrix is singular and not invertible.")
        
        cofactor = cofactor_matrix_3x3(matrix)
        return cofactor.T / det

    else:
        raise ValueError("Only 2x2 or 3x3 matrices are supported.")

# ==========================
# Example Usage
# ==========================

# Define test matrices
A_2x2 = np.array([[4, 7], [2, 6]])
A_3x3 = np.array([[3, 0, 2], [2, 0, -2], [0, 1, 1]])

# Compute inverses
inv_2x2 = invert_matrix(A_2x2)
inv_3x3 = invert_matrix(A_3x3)

# Verify results with numpy for comparison
print("Inverse of 2x2 matrix:")
print("Our result:\n", inv_2x2)
print("NumPy result:\n", np.linalg.inv(A_2x2))

print("\nInverse of 3x3 matrix:")
print("Our result:\n", inv_3x3)
print("NumPy result:\n", np.linalg.inv(A_3x3))

# Test with nearly singular matrix (optional)
test_matrix = np.array([[1, 2], [2, 4]])
try:
    inv = invert_matrix(test_matrix)
except ValueError as e:
    print(f"\nCorrectly caught singular matrix: {e}")