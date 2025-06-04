import numpy as np


matrix_a = np.array([[1, 2, 3],
                     [4, 5, 6]])

matrix_b = np.array([[6, 5, 4],
                     [3, 2, 1]])


matrix_add = matrix_a + matrix_b


matrix_sub = matrix_a - matrix_b


scalar = 2
matrix_scalar_mul = scalar * matrix_a


matrix_transpose = matrix_a.T


print("Matrix A:\n", matrix_a)
print("Matrix B:\n", matrix_b)
print("\nMatrix Addition (A + B):\n", matrix_add)
print("\nMatrix Subtraction (A - B):\n", matrix_sub)
print(f"\nScalar Multiplication ({scalar} * A):\n", matrix_scalar_mul)
print("\nTranspose of A (Aᵀ):\n", matrix_transpose)
