import numpy as np

# In this implementation, I demonstrate foundational matrix operations using NumPy,
# including matrix addition, subtraction, scalar multiplication, and transposition.
# These operations form the backbone of many algorithms in machine learning, 
# data science, and linear algebra.

# Define Matrix A: a 2x3 matrix.
# This matrix will serve as the base for all subsequent operations.
matrix_a = np.array([[1, 2, 3],
                     [4, 5, 6]])

# Define Matrix B: another 2x3 matrix, shaped identically to Matrix A.
# Performing element-wise operations requires the matrices to have the same shape.
matrix_b = np.array([[6, 5, 4],
                     [3, 2, 1]])

# Perform element-wise matrix addition.
# Each element in matrix_a is added to its corresponding element in matrix_b.
matrix_add = matrix_a + matrix_b

# Perform element-wise matrix subtraction.
# Here, each element in matrix_b is subtracted from the corresponding element in matrix_a.
matrix_sub = matrix_a - matrix_b

# Define a scalar value to multiply with matrix_a.
# Scalar multiplication scales each element in the matrix by the given value.
scalar = 2
matrix_scalar_mul = scalar * matrix_a

# Compute the transpose of matrix_a.
# Transposing swaps rows and columns, which is often required in operations
# like dot products, matrix multiplication, or aligning data representations.
matrix_transpose = matrix_a.T

# Output the results of all matrix operations.
# Printing each step provides a clear view of how the matrices are transformed.
print("Matrix A:\n", matrix_a)
print("Matrix B:\n", matrix_b)
print("\nMatrix Addition (A + B):\n", matrix_add)
print("\nMatrix Subtraction (A - B):\n", matrix_sub)
print(f"\nScalar Multiplication ({scalar} * A):\n", matrix_scalar_mul)
print("\nTranspose of A (Aᵀ):\n", matrix_transpose)
