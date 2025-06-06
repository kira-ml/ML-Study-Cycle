import numpy as np

# In this implementation, I demonstrate basic element-wise operations 
# between two NumPy arrays, which represent mathematical vectors.
# NumPy is a powerful library for numerical computing in Python, and 
# its array operations are optimized for performance and simplicity.

# Define the first vector using a NumPy array.
# This array serves as Vector A for subsequent operations.
vector_a = np.array([1, 2, 3])

# Define the second vector (Vector B) in a similar fashion.
vector_b = np.array([4, 5, 6])

# Perform element-wise addition.
# Each corresponding element from vector_a and vector_b is summed.
vector_add = vector_a + vector_b

# Perform element-wise subtraction.
# Each element in vector_b is subtracted from the corresponding element in vector_a.
vector_sub = vector_a - vector_b

# Perform element-wise multiplication.
# This operation multiplies corresponding elements of the two vectors.
vector_mul = vector_a * vector_b

# Perform element-wise division.
# Each element in vector_a is divided by the corresponding element in vector_b.
# This demonstrates how NumPy handles floating-point division between integer arrays.
vector_div = vector_a / vector_b

# Output each vector and the result of each operation to the console.
# This allows us to verify that the operations produced the expected results.
print("Vector A:", vector_a)
print("Vector B:", vector_b)
print("Addition (A + B):", vector_add)
print("Subtraction (A - B):", vector_sub)
print("Element-wise Multiplication (A * B):", vector_mul)
print("Element-wise Division (A / B):", vector_div)
