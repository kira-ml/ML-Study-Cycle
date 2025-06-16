import numpy as np

# In this implementation, I demonstrate LU decomposition using Gaussian elimination without pivoting.
# This method factors a square matrix A into the product of a lower triangular matrix L and an upper triangular matrix U.
# Here, I walk through the steps explicitly to clarify how L and U are computed iteratively.

# Define the original matrix A
A = np.array([[2, 3], [4, 7]], dtype=float)
print("Original matrix A:\n", A)

# Get the size of the matrix (assuming square matrix for LU decomposition)
n = A.shape[0]

# Initialize L as an identity matrix of size n x n
# This will be populated during elimination to represent the multipliers used in row operations
L = np.eye(n)

# Initialize U as a copy of A; U will be transformed into an upper triangular matrix through row operations
U = A.copy()

print("Initial Lower matrix L:\n", L)
print("Initial Upper matrix U:\n", U)

# Perform forward elimination to transform U into upper triangular form
# and populate L with the corresponding Gaussian elimination multipliers
for i in range(n):
    for j in range(i + 1, n):
        # Compute the factor by which we multiply the pivot row before subtracting it from row j
        # This ensures that element (j, i) in U becomes zero
        factor = U[j, i] / U[i, i]
        
        # Store the factor in the L matrix at the corresponding position
        L[j, i] = factor
        
        # Perform the row operation on U: row_j = row_j - factor * row_i
        U[j] = U[j] - factor * U[i]

        # Print intermediate results for educational purposes
        print(f"\nEliminating row {j}, column {i}")
        print("Factor:", factor)
        print("Updated U:\n", U)
        print("Updated L:\n", L)

# Reconstruct the original matrix A by multiplying L and U to verify correctness of decomposition
A_reconstructed = L @ U
print("\nReconstructed A from L and U:\n", A_reconstructed)