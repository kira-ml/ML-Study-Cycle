import numpy as np

# In this implementation, I demonstrate how to solve a linear system of equations Ax = b
# using Gaussian Elimination followed by Back Substitution. This method is a foundational
# approach in numerical linear algebra, useful for educational purposes and for understanding
# the inner workings of solvers used in scientific computing.

# Define a 3x3 coefficient matrix A. I ensure it is of type float to allow for decimal division.
A = np.array([
    [2, 1, -1],
    [-3, -1, 2],
    [-2, 1, 2]
], dtype=float)

# Define the constants vector b, which corresponds to the right-hand side of the linear system.
b = np.array([8, -11, -3], dtype=float)

# Display the initial inputs for verification and transparency.
print("Matrix A:\n", A)
print("Vector b:\n", b)

# To perform Gaussian Elimination, I first construct the augmented matrix [A | b].
# This matrix allows us to apply row operations directly to both the coefficients and constants.
Ab = np.hstack((A, b.reshape(-1, 1)))
print("Initial augmented matrix [A|b]:\n", Ab)

# Get the number of equations (which is also the number of rows).
n = Ab.shape[0]

# --- Forward Elimination Phase ---
# The goal here is to transform the matrix into an upper triangular form.
# This simplifies solving the system through backward substitution.

for i in range(n):
    # Numerical stability check: if the pivot element is too close to zero,
    # the division would become unstable. In a production-grade solver, we'd perform partial pivoting.
    if abs(Ab[i, i]) < 1e-12:
        raise ValueError(f"Zero pivot encountered at row {i}. Consider using pivoting to avoid instability.")

    # Eliminate all entries below the pivot (i.e., make them zero).
    for j in range(i + 1, n):
        # Compute the multiplier needed to zero out Ab[j, i].
        multiplier = Ab[j, i] / Ab[i, i]
        print(f"Eliminating row {j}, using row {i} with multiplier {multiplier:.4f}")

        # Apply the row operation: row_j = row_j - multiplier * row_i
        Ab[j] -= multiplier * Ab[i]
        print(f"Augmented matrix after eliminating row {j}:\n{Ab}\n")

# --- Back Substitution Phase ---
# At this point, the matrix is upper triangular.
# I solve for each variable starting from the last row and working upward.

# Initialize the solution vector with zeros.
x = np.zeros(n)

for i in range(n - 1, -1, -1):
    # For each row, solve for x[i] by isolating it in the equation:
    # Ab[i, -1] = Ab[i, i] * x[i] + sum of known x[j] * Ab[i, j] for j > i
    x[i] = (Ab[i, -1] - np.dot(Ab[i, i + 1:n], x[i + 1:n])) / Ab[i, i]
    print(f"Back substitution at row {i}: x[{i}] = {x[i]:.4f}")

# Output the final solution vector.
print("\nSolution vector x:")
print(x)
