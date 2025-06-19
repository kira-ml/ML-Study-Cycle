import numpy as np

# In this implementation, I demonstrate the core steps of computing the Singular Value Decomposition (SVD)
# of a symmetric matrix using NumPy. This involves calculating the left and right singular vectors,
# the singular values, and then verifying the reconstruction of the original matrix using the SVD components.

# Define a symmetric 2x2 matrix A
A = np.array([
    [3, 1],
    [1, 3]
], dtype=float)

print("Matrix A:\n", A)

# Transpose of A, denoted A^T
A_T = A.T

# Compute A^T A (used to obtain the right singular vectors V and singular values)
AtA = A_T @ A

# Compute A A^T (used to obtain the left singular vectors U and singular values)
AAt = A @ A_T

print("A^T A:\n", AtA)
print("A A^T:\n", AAt)

# Step 1: Compute the eigenvalues and eigenvectors of A^T A
# These eigenvectors form the matrix V (right singular vectors of A)
eigenvalues_AtA, V = np.linalg.eig(AtA)

# Sort the eigenvalues (and corresponding eigenvectors) in descending order for numerical stability
idx = np.argsort(eigenvalues_AtA)[::-1]
eigenvalues_AtA = eigenvalues_AtA[idx]
V = V[:, idx]

# Singular values are the square roots of the eigenvalues of A^T A (or A A^T, since A is symmetric)
singular_values = np.sqrt(eigenvalues_AtA)

print("Eigenvalues of A^T A:", eigenvalues_AtA)
print("Singular values (√eigenvalues):", singular_values)
print("Right singular vectors (columns of V):\n", V)

# Step 2: Compute the eigenvalues and eigenvectors of A A^T
# These eigenvectors form the matrix U (left singular vectors of A)
eigenvalues_AAt, U = np.linalg.eig(AAt)

# Again, sort in descending order for alignment with the singular values and right singular vectors
idx = np.argsort(eigenvalues_AAt)[::-1]
eigenvalues_AAt = eigenvalues_AAt[idx]
U = U[:, idx]

print("Eigenvalues of A A^T:", eigenvalues_AAt)
print("Left singular vectors (columns of U):\n", U)

# Step 3: Construct the Sigma matrix
# Sigma is a diagonal matrix with the singular values placed along the diagonal
Sigma = np.zeros_like(A, dtype=float)
np.fill_diagonal(Sigma, singular_values)

print("Sigma matrix (diagonal with singular values):\n", Sigma)

# Step 4: Verify the decomposition by reconstructing A
# According to the SVD formula: A ≈ U Σ V^T
A_reconstructed = U @ Sigma @ V.T

print("Reconstructed A (from U Σ V^T):\n", A_reconstructed)
print("Original A:\n", A)
