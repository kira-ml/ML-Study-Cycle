import numpy as np

# In this implementation, I demonstrate how to perform QR decomposition from scratch
# using the classical Gram-Schmidt process. The QR decomposition factorizes a matrix A 
# into two components: 
# - Q: an orthonormal matrix (columns are orthogonal unit vectors)
# - R: an upper triangular matrix
#
# This factorization is widely used in numerical linear algebra, such as solving
# linear least squares problems and eigenvalue algorithms.

# Step 1: Define a real-valued 3x3 matrix A.
# You can generalize this method to any m x n matrix where m ≥ n.
A = np.array([
    [2, 7, 8],
    [4, 5, 9],
    [3, 8, 2],
], dtype=float)

# Retrieve matrix dimensions
m, n = A.shape  # m: number of rows, n: number of columns

# Initialize matrices Q and R
# Q will store the orthonormal basis vectors (same shape as A)
# R will store the upper triangular matrix (n x n since A has n columns)
Q = np.zeros((m, n))
R = np.zeros((n, n))

# Step 2: Classical Gram-Schmidt Orthogonalization
# For each column of A, we subtract projections onto the previous orthonormal vectors
# to ensure orthogonality, and then normalize the result to get a unit vector.

for i in range(n):
    # Start with the original column vector v from A
    v = A[:, i].copy()

    # Subtract projections onto previously computed Q vectors
    for j in range(i):
        # Compute the projection scalar
        R[j, i] = np.dot(Q[:, j], A[:, i])
        # Remove the component in the direction of Q[:, j]
        v -= R[j, i] * Q[:, j]

    # Compute the norm of the orthogonalized vector
    R[i, i] = np.linalg.norm(v)

    # Normalize to get the i-th orthonormal vector
    Q[:, i] = v / R[i, i]

# Step 3: Validation

# Multiply Q and R to reconstruct A
A_reconstructed = Q @ R

print("Reconstructed A (Q @ R):\n", A_reconstructed)
print("\nOriginal A:\n", A)
print("\nDifference (A - Q @ R):\n", A - A_reconstructed)

# Verify that Q is truly orthonormal: Q.T @ Q should be the identity matrix
print("\nQ.T @ Q (should be identity):\n", Q.T @ Q)
