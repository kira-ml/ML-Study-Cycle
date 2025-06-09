import numpy as np

# In this implementation, I demonstrate how to compute the determinant of matrices using both
# NumPy's built-in linear algebra routines and a manual cofactor expansion method.
# This comparison helps build intuition for how determinants are computed under the hood.

# Define a 2x2 matrix A.
A = np.array([[1, 2],
              [3, 4]])

# Define a 3x3 matrix B.
B = np.array([
    [1, 2, 3],
    [0, 1, 4],
    [5, 6, 0]
])


def det_2x2(mat):
    """
    Computes the determinant of a 2x2 matrix using the closed-form formula:
    |a b|
    |c d| = ad - bc

    Parameters:
        mat (np.ndarray): A 2x2 NumPy array.

    Returns:
        float: Determinant of the 2x2 matrix.
    """
    return mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]


# Compute the determinant of matrix A using NumPy's built-in function.
# This is a reliable and efficient approach for general use cases.
A_det = np.linalg.det(A)

print("Matrix A:\n", A)
print("Determinant of A:", A_det)


# Now, I manually compute the determinant of the 3x3 matrix B using cofactor (Laplace) expansion along the first row.
# This approach recursively reduces the matrix to smaller submatrices (minors) until reaching a 2x2 base case.

det = 0  # Accumulator for the determinant result

# Iterate across the first row (row 0) to expand the determinant
for j in range(B.shape[1]):
    # Calculate the cofactor sign based on position (0 + j)
    sign = (-1) ** j

    # Extract the minor by removing row 0 and column j
    minor = np.delete(np.delete(B, 0, axis=0), j, axis=1)

    # Compute the determinant of the 2x2 minor
    minor_det = det_2x2(minor)

    # Print the intermediate values to illustrate the recursive logic of Laplace expansion
    print(f"Minor matrix for column {j}:\n{minor}")
    print(f"Element: {B[0, j]}, Sign: {sign}, Determinant of minor: {minor_det}")

    # Accumulate the current cofactor contribution
    det += sign * B[0, j] * minor_det

    print(f"Current determinant total: {det}\n")

# Display the final result of the manual determinant computation
print("Matrix B:\n", B)
print("Determinant of B (manual):", det)


# For validation, I compute the determinant of matrix B using NumPy's built-in routine.
# This confirms the correctness of the manual Laplace expansion.
B_det_np = np.linalg.det(B)
print("Determinant of B (NumPy):", B_det_np)
