import numpy as np

# In this implementation, I demonstrate how matrix-vector multiplication
# can be interpreted both computationally (via NumPy's `dot` function) 
# and conceptually (as a linear combination of a matrix's columns).
# This dual perspective is fundamental in linear algebra and widely used 
# in machine learning models, especially in understanding affine transformations.

# Define matrix A as a 3x2 NumPy array.
# Each row of A represents a 2D data point or a feature mapping,
# and each column will participate in the linear combination.
A = np.array([
    [2, 3],
    [4, 5],
    [6, 7],
])

# Define vector v as a 2-element array.
# This vector serves as a set of weights in the linear combination of A's columns.
v = np.array([1, 2])

# Perform matrix-vector multiplication using NumPy's dot product.
# The result is a 3-element vector, where each element is the dot product
# of a row in A with the vector v.
result = np.dot(A, v)

# Manually compute the linear combination of A's columns using the components of v.
# This reinforces the interpretation of matrix-vector multiplication as a weighted
# sum of a matrix’s columns.
col1 = A[:, 0] * v[0]  # Scale the first column of A by v[0]
col2 = A[:, 1] * v[1]  # Scale the second column of A by v[1]
linear_comb = col1 + col2  # Add the scaled columns to form the final result

# Print the original matrix, vector, and both computation approaches.
# Comparing the results illustrates that matrix-vector multiplication is
# equivalent to a linear combination of the matrix's columns.
print("Matrix A:\n", A)
print("Vector v:\n", v)
print("\nResult of A · v (matrix-vector multiplication):\n", result)
print("\nLinear combination (col1 + col2):\n", linear_comb)
