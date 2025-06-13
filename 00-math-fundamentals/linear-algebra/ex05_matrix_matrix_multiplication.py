import numpy as np

# In this implementation, I demonstrate how to perform matrix multiplication using NumPy.
# We start by defining two matrices A and B. These are 2D NumPy arrays with compatible dimensions.

A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

# Before proceeding with matrix multiplication, it's critical to validate that the operation is mathematically valid.
# Specifically, the number of columns in A must match the number of rows in B.
if A.shape[1] != B.shape[0]:
    raise ValueError("Incompatible dimensions for matrix multiplication: A.columns must equal B.rows.")

# I initialize an output matrix `result` with zeros, having the appropriate shape for the result of A @ B.
# The result will have the same number of rows as A and the same number of columns as B.
# This step is typically required in manual matrix multiplication algorithms.
result = np.zeros((A.shape[0], B.shape[1]))

# Now, I use nested loops to explicitly compute each element of the result matrix.
# For each element at position (i, j), I calculate the dot product between:
# - the i-th row of A
# - the j-th column of B
# This manual computation illustrates the fundamental logic behind matrix multiplication.
for i in range(A.shape[0]):
    for j in range(B.shape[1]):
        result[i, j] = np.dot(A[i, :], B[:, j])  # Dot product between row vector from A and column vector from B

# While the nested loop above is instructive, NumPy provides a highly optimized built-in method to perform the same operation:
result = np.dot(A, B)

# Alternatively, I can use the @ operator (available in Python 3.5+), which is functionally identical to np.dot for 2D arrays.
# This syntax is cleaner and increasingly preferred for clarity and readability.
result = A @ B

# Finally, I print the original matrices and the resulting product to the console.
print("Matrix A:\n", A)
print("Matrix B:\n", B)
print("\nResult of A @ B (matrix-matrix multiplication):\n", result)
