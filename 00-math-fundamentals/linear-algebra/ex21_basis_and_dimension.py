"""
ex21_basis_and_dimension.py
--------------------------
In this implementation, I demonstrate how to determine whether a set of vectors forms a basis for a vector space, compute the rank (dimension) of a matrix, and extract a basis for the null space using NumPy and SciPy. This script is structured to illustrate foundational concepts in linear algebra that are directly relevant to both academic research and practical machine learning pipelines.

The code is written for clarity and pedagogical value, with detailed documentation and comments to guide motivated learners through each step.
"""

import numpy as np
from scipy.linalg import null_space

# --- Basis Verification in R^2 ---

vector_1 = np.array([1, 2])
vector_2 = np.array([3, 4])

# Stack vectors as columns to form a 2x2 matrix.
# This is a standard approach for checking linear independence and basis properties.
matrix = np.column_stack([vector_1, vector_2])

# The rank of the matrix equals the number of linearly independent columns.
rank = np.linalg.matrix_rank(matrix)
dimension = 2  # The dimension of R^2

if rank == dimension:
    print("The vectors form a basis for ℝ².")
else:
    print("The vector do NOT form a basis for ℝ²")

# --- Rank and Null Space of a 3x3 Matrix ---

A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]])

# Singular Value Decomposition (SVD) is used here to robustly compute the rank,
# which is numerically stable and widely used in ML/data science pipelines.
u, s, vh = np.linalg.svd(A)

# The number of non-negligible singular values gives the rank (dimension of the column space).
column_rank = np.sum(s > 1e-10)
print(f"Rank (column space dimension): {column_rank}")

# The null_space function from SciPy returns an orthonormal basis for the null space of A.
# This is essential in many ML and optimization contexts (e.g., understanding solution sets).
null_basis = null_space(A)

print(f"Basis for null space:")
print(null_basis)