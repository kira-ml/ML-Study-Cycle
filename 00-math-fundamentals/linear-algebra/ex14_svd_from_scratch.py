import numpy as np

# In this implementation, I demonstrate the core steps of computing the Singular Value Decomposition (SVD)
# of a symmetric matrix using NumPy. This involves calculating the left and right singular vectors,
# the singular values, and then verifying the reconstruction of the original matrix using the SVD components.

# Define a symmetric 2x2 matrix A
A = np.array([
    [3, 1],
    [1, 3]
], dtype=float)

symmetry_threshold = 1e-10
assert np.allclose(A, A.T, atol=symmetry_threshold, rtol=1e-8), \
    f"Matrix A is not symmetric (threshold: {symmetry_threshold})"




print("Matrix A:\n", A)

AtA = A.T @ A


print("A^T A:\n", AtA)


# Step 1: Compute the eigenvalues and eigenvectors of A^T A
# These eigenvectors form the matrix V (right singular vectors of A)
eigenvalues_AtA, V = np.linalg.eig(AtA)
eigenvalues_Ata = np.real_if_close(eigenvalues_AtA)
assert np.all(np.isreal(eigenvalues_AtA)), "Complex eigenvalues detected"

# Sort the eigenvalues (and corresponding eigenvectors) in descending order for numerical stability
idx = np.argsort(eigenvalues_AtA)[::-1]
eigenvalues_AtA = eigenvalues_AtA[idx]
V = V[:, idx]

# Singular values are the square roots of the eigenvalues of A^T A (or A A^T, since A is symmetric)
singular_values = np.sqrt(np.abs(eigenvalues_AtA))

print("Eigenvalues of A^T A:", eigenvalues_AtA)
print("Singular values (√eigenvalues):", singular_values)
print("Right singular vectors (columns of V):\n", V)

# Step 2: Compute the eigenvalues and eigenvectors of A A^T
# These eigenvectors form the matrix U (left singular vectors of A)


# Again, sort in descending order for alignment with the singular values and right singular vectors
AAt = A @ A.T
eigenvalues_AAt, U = np.linalg.eig(AAt)
U = U[:, np.argsort(eigenvalues_AAt)[::-1]]


print("Eigenvalues of A A^T:", eigenvalues_AAt)
print("Left singular vectors (columns of U):\n", U)

# Step 3: Construct the Sigma matrix
# Sigma is a diagonal matrix with the singular values placed along the diagonal
Sigma = np.diag(singular_values)

print("Sigma matrix (diagonal with singular values):\n", Sigma)

# Step 4: Verify the decomposition by reconstructing A
# According to the SVD formula: A ≈ U Σ V^T
reconstruction_error = np.linalg.norm(A - U @ Sigma @ V.T, 'fro')

print(f"Reconstruction Frobenius error: {reconstruction_error:.2e}")
assert reconstruction_error < 1e-8, "SVD verification failed"
