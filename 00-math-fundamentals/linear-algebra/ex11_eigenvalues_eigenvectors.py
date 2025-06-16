import numpy as np

A = np.array([
    [4, 1],
    [2, 3]],
    dtype=float)
print("Matrix A:", A)


trace_A = np.trace(A)
det_A = np.linalg.det(A)

print("Trace of A:", trace_A)
print("Determinant of A:", det_A)


coeffs = [1, -trace_A, det_A]
eigenvalues = np.roots(coeffs)
print("Eigenvalues (roots of characteristic equation):", eigenvalues)



lambda_1 = eigenvalues[0]
I = np.eye(2)
A_minus_lambdaI = A - lambda_1 * I
print(f"A - λ₁I:\n{A_minus_lambdaI}")


v1 = np.array([1, (lambda_1 - A[0, 0]) / A[0, 1]])

v1_normalized = v1 / np.linalg.norm(v1)

print("Eigenvector for λ₁:", v1_normalized)

D = np.diag(eigenvalues)
P = np.column_stack((v1_normalized, v2_normalized))
print("Diagonal matrix D:\n", D)
print("Eigenvector matrix P:\n", P)

A_reconstructed = P @ D @ np.linalg.inv(P)
print("Reconstructed A:\n", A_reconstructed)