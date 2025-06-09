import numpy as np

# In this implementation, I demonstrate how to compute the inverse of 2x2 and 3x3 matrices
# manually using the adjugate method. This includes computing determinants, cofactors,
# and validating invertibility prior to inversion. NumPy is used to facilitate array operations.

def determinant_2x2(matrix):
    """
    Computes the determinant of a 2x2 matrix using the standard formula:
    det = ad - bc
    """
    return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]

def determinant_3x3(matrix):
    """
    Computes the determinant of a 3x3 matrix using cofactor expansion along the first row.
    The formula expands as:
    det = a(ei − fh) − b(di − fg) + c(dh − eg)
    where a, b, c are the elements of the first row.
    """
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

def cofactor_matrix_3x3(matrix):
    """
    Constructs the cofactor matrix for a 3x3 input matrix.
    For each element, it computes the corresponding 2x2 minor and applies the correct sign
    using the cofactor formula: C_ij = (-1)^(i+j) * det(minor_ij)
    """
    cofactor = np.zeros((3, 3))
    for row in range(3):
        for col in range(3):
            # Remove the current row and column to form the 2x2 minor matrix
            minor = np.delete(np.delete(matrix, row, axis=0), col, axis=1)
            sign = (-1) ** (row + col)
            cofactor[row, col] = sign * determinant_2x2(minor)
    return cofactor

def invert_matrix(matrix):
    """
    Computes the inverse of a 2x2 or 3x3 matrix using the adjugate (classical) method.
    The function first checks the matrix shape, validates that it is invertible
    (i.e., has a non-zero determinant), and then constructs the inverse accordingly.

    Parameters:
    - matrix: 2x2 or 3x3 array-like input.

    Returns:
    - The inverse matrix as a NumPy array.

    Raises:
    - ValueError: if the matrix is not square of size 2x2 or 3x3, or is singular.
    """
    matrix = np.array(matrix, dtype=float)
    shape = matrix.shape

    if shape == (2, 2):
        det = determinant_2x2(matrix)
        if det == 0:
            raise ValueError("Matrix is singular and not invertible.")

        # Construct the adjugate matrix by swapping and negating elements appropriately
        adjugate = np.array([
            [ matrix[1, 1], -matrix[0, 1]],
            [-matrix[1, 0],  matrix[0, 0]]
        ])
        return (1 / det) * adjugate

    elif shape == (3, 3):
        det = determinant_3x3(matrix)
        if det == 0:
            raise ValueError("Matrix is singular and not invertible.")

        # Compute cofactor matrix, then transpose to get the adjugate
        cofactor = cofactor_matrix_3x3(matrix)
        adjugate = cofactor.T
        return (1 / det) * adjugate

    else:
        raise ValueError("Only 2x2 or 3x3 matrices are supported.")

# ==========================
# Example Usage
# ==========================

# Define a 2x2 matrix with a non-zero determinant
A_2x2 = np.array([
    [4, 7],
    [2, 6]
])

# Define a 3x3 matrix with a non-zero determinant
A_3x3 = np.array([
    [3, 0, 2],
    [2, 0, -2],
    [0, 1, 1]
])

# Compute the inverse of the 2x2 matrix
inv_2x2 = invert_matrix(A_2x2)

# Compute the inverse of the 3x3 matrix
inv_3x3 = invert_matrix(A_3x3)

# Display the results
print("Inverse of 2x2 matrix:\n", inv_2x2)
print("\nInverse of 3x3 matrix:\n", inv_3x3)
