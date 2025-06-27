import numpy as np

# In this implementation, I demonstrate the core steps of computing the Singular Value Decomposition (SVD)
# of a symmetric matrix using NumPy. The procedure includes calculating the left and right singular vectors,
# extracting the singular values, and verifying that the decomposition reconstructs the original matrix accurately.

# I begin by defining a symmetric 2x2 matrix A.
A = np.array([
    [3, 1],
    [1, 3]
], dtype=float)

# I enforce symmetry explicitly to ensure that later steps relying on A = A^T are numerically sound.
symmetry_threshold = 1e-10
assert np.allclose(A, A.T, atol=symmetry_threshold, rtol=1e-8), \
    f"Matrix A is not symmetric (threshold: {symmetry_threshold})"

print("Matrix A:\n", A)

# For a symmetric matrix, A^T A and A A^T share non-zero eigenvalues.
# Here, I compute A^T A, which will be used to derive the right singular vectors.
AtA = A.T @ A
print("A^T A:\n", AtA)

# Step 1: Compute the eigenvalues and eigenvectors of A^T A.
# The eigenvectors of A^T A correspond to the right singular vectors (columns of V).
eigenvalues_AtA, V = np.linalg.eig(AtA)

# I ensure the eigenvalues are treated as real numbers (since A^T A is symmetric and positive semidefinite).
eigenvalues_AtA = np.real_if_close(eigenvalues_AtA)
assert np.all(np.isreal(eigenvalues_AtA)), "Complex eigenvalues detected"

# To improve numerical stability and consistent ordering, I sort eigenvalues and their corresponding eigenvectors
# in descending order.
idx = np.argsort(eigenvalues_AtA)[::-1]
eigenvalues_AtA = eigenvalues_AtA[idx]
V = V[:, idx]

# Singular values are defined as the square roots of the eigenvalues of A^T A.
# Because eigenvalues could be very small negative numbers due to floating-point errors, I take their absolute values.
singular_values = np.sqrt(np.abs(eigenvalues_AtA))

print("Eigenvalues of A^T A:", eigenvalues_AtA)
print("Singular values (√eigenvalues):", singular_values)
print("Right singular vectors (columns of V):\n", V)

# Step 2: Compute the eigenvalues and eigenvectors of A A^T.
# These eigenvectors correspond to the left singular vectors (columns of U).
AAt = A @ A.T
eigenvalues_AAt, U = np.linalg.eig(AAt)

# I sort the eigenvectors of A A^T to align them with the ordering of singular values.
U = U[:, np.argsort(eigenvalues_AAt)[::-1]]

print("Eigenvalues of A A^T:", eigenvalues_AAt)
print("Left singular vectors (columns of U):\n", U)

# Step 3: Construct the Sigma matrix.
# Sigma is the diagonal matrix that holds the singular values along its main diagonal.
Sigma = np.diag(singular_values)
print("Sigma matrix (diagonal with singular values):\n", Sigma)

# Step 4: Verify that the decomposition reconstructs the original matrix.
# According to the SVD formulation, A should equal U @ Sigma @ V^T.
# I compute the Frobenius norm of the reconstruction error to assess numerical accuracy.
reconstruction_error = np.linalg.norm(A - U @ Sigma @ V.T, 'fro')
print(f"Reconstruction Frobenius error: {reconstruction_error:.2e}")

# I assert that the reconstruction error is below a tight tolerance,
# ensuring that the decomposition is correct within numerical precision.
assert reconstruction_error < 1e-8, "SVD verification failed"
