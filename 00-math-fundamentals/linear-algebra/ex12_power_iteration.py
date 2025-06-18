import numpy as np

# In this implementation, I demonstrate how to approximate the dominant eigenvalue 
# and its corresponding eigenvector of a square matrix using the **Power Iteration** method.
# This is a fundamental technique in numerical linear algebra, with applications in 
# Principal Component Analysis (PCA), Google's PageRank, and many other spectral methods.

# Step 1: Initialize the input matrix
# For demonstration, I create a random 5x5 matrix `A`. In practice, this could be
# a covariance matrix, graph adjacency matrix, or any symmetric real matrix.
A = np.random.rand(5, 5)

# Step 2: Initialize a random vector `b` with the same number of elements as columns in A.
# This serves as the starting point for the power iteration.
b = np.random.rand(A.shape[1])

# Step 3: Set the number of iterations and an optional convergence tolerance
# Power iteration is an iterative algorithm and usually converges within a reasonable number
# of steps if the dominant eigenvalue is well separated from the others.
num_iterations = 100
tolerance = 1e-10  # (not used here explicitly, but useful for convergence checking)

# Step 4: Perform the Power Iteration loop
# In each iteration, we multiply the current vector by the matrix A and normalize the result.
# This gradually aligns `b` with the eigenvector corresponding to the dominant eigenvalue.
for i in range(num_iterations):
    # Matrix-vector multiplication: apply A to the current vector
    b = np.dot(A, b)

    # Normalize the resulting vector to prevent numerical overflow or underflow
    b_norm = np.linalg.norm(b)
    b = b / b_norm

# Step 5: Estimate the dominant eigenvalue using the Rayleigh quotient
# This gives a scalar approximation of the eigenvalue associated with the current vector `b`.
eigenvalue = np.dot(b.T, np.dot(A, b)) / np.dot(b.T, b)

# Step 6: Output the results
# `b` approximates the dominant eigenvector of A, and `eigenvalue` is the corresponding eigenvalue.
print("Final estimated eigenvector (normalized):")
print(b)

print("\nEstimated dominant eigenvalue:")
print(eigenvalue)
