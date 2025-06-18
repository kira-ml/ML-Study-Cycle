import numpy as np

# In this implementation, I demonstrate how to compute the eigenvalues and eigenvectors
# of a 2x2 matrix using the characteristic polynomial and verify diagonalization.
# This serves as a foundational technique in linear algebra and is applicable to many
# machine learning algorithms, including PCA and spectral methods.

# Define a 2x2 real-valued matrix A
A = np.array([
    [4, 1],
    [2, 3]
], dtype=float)

print("Matrix A:\n", A)

# --- Step 1: Compute the trace and determinant of A ---
# These are the coefficients of the characteristic polynomial:
# λ² - (trace)λ + (determinant) = 0
trace_A = np.trace(A)
det_A = np.linalg.det(A)

print("Trace of A:", trace_A)
print("Determinant of A:", det_A)

# --- Step 2: Solve the characteristic polynomial to get eigenvalues ---
# The characteristic equation is: λ² - trace_A*λ + det_A = 0
coeffs = [1, -trace_A, det_A]
eigenvalues = np.roots(coeffs)

print("Eigenvalues (roots of characteristic equation):", eigenvalues)

# --- Step 3: Compute an eigenvector for the first eigenvalue ---
lambda_1 = eigenvalues[0]  # Select the first eigenvalue

# Create identity matrix I and compute (A - λI)
I = np.eye(2)
A_minus_lambdaI = A - lambda_1 * I
print(f"A - λ₁I:\n{A_minus_lambdaI}")

# Solve (A - λI)v = 0 to find the eigenvector.
# For a 2x2 matrix, we can manually construct an eigenvector.
# Here, we solve for v = [1, x] by rearranging the first row of A - λI
# Avoid division by zero by ensuring A[0,1] ≠ 0.
v1 = np.array([1, (lambda_1 - A[0, 0]) / A[0, 1]])

# Normalize the eigenvector to unit length
v1_normalized = v1 / np.linalg.norm(v1)

print("Eigenvector for λ₁ (normalized):", v1_normalized)

# --- Step 4: Repeat for the second eigenvalue ---
lambda_2 = eigenvalues[1]
A_minus_lambdaI_2 = A - lambda_2 * I
print(f"A - λ₂I:\n{A_minus_lambdaI_2}")

# Solve similarly for the second eigenvector
v2 = np.array([1, (lambda_2 - A[0, 0]) / A[0, 1]])
v2_normalized = v2 / np.linalg.norm(v2)

print("Eigenvector for λ₂ (normalized):", v2_normalized)

# --- Step 5: Diagonalize the matrix ---
# Construct diagonal matrix D with eigenvalues and matrix P with eigenvectors
D = np.diag(eigenvalues)
P = np.column_stack((v1_normalized, v2_normalized))

print("Diagonal matrix D:\n", D)
print("Eigenvector matrix P (columns are eigenvectors):\n", P)

# --- Step 6: Reconstruct A using its eigendecomposition ---
# If A = P D P⁻¹, then we can reconstruct A as a consistency check
A_reconstructed = P @ D @ np.linalg.inv(P)

print("Reconstructed A (via P D P⁻¹):\n", A_reconstructed)
