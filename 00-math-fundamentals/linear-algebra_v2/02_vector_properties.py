import numpy as np


v = np.array([1, 2, 3, 4, 5])


print("Our Vector:")
print(f"v = {v}")
print(f"Type of v: {type(v)}")
print()  # Empty line for readability


vector_size = v.size


print("Part 2: Vector Size (.size attribute)")
print(f"v.size = {vector_size}")
print(f"Interpretation: There are {vector_size} elements in vector v")
print(f"Check: v = {v} has {vector_size} numbers: {list(v)}\n")


vector_shape = v.shape


print("Part 3: Vector Shape (.shape attribute)")
print(f"v.shape = {vector_shape}")
print(f"Interpretation: This is a {len(vector_shape)}-dimensional array")
print(f"The comma in {vector_shape} indicates it's a tuple with one element")

print("\nShape notation explained:")
print(f"v.shape = {vector_shape} means:")
print("  - The array has 1 dimension (it's a vector)")
print(f"  - That dimension has {vector_shape[0]} elements")
print("  - The comma (,) indicates this is a tuple, not just a number\n")



print("Part 4: Relationship between .size and .shape")
print(f"v = {v}")
print(f"v.size = {v.size}  (total elements)")
print(f"v.shape = {v.shape}  (dimensions/structure)")



print(f"\nFor 1D vectors: v.size = v.shape[0]")
print(f"Check: {v.size} = {v.shape[0]}")
print(f"Is this true? {v.size == v.shape[0]}\n")
print("Part 5: Comparing NumPy attributes with Python's len()")
print(f"v.size = {v.size}  (NumPy attribute - total elements)")
print(f"len(v) = {len(v)}  (Python function - length of first dimension)")

print("\nFor 1D vectors, len(v) and v.shape[0] give the same result:")
print(f"len(v) = {len(v)}, v.shape[0] = {v.shape[0]}")
print(f"Are they equal? {len(v) == v.shape[0]}")


print("\nBUT for 2D arrays, they're different:")

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
    ])

print(f"\nmatrix = \n{matrix}")
print(f"matrix.shape = {matrix.shape}  (2 rows, 3 columns)")
print(f"matrix.size = {matrix.size}  (2 × 3 = 6 total elements)")
print(f"len(matrix) = {len(matrix)}  (only gives 2 - the number of rows!)")

print("\nConclusion: Use .size and .shape for consistent behavior across all dimensions!\n")


print("Part 6: Practice Exercises")
print("=" * 50)

# Exercise 1: Create your own vector
print("\nExercise 1: Create a vector with your favorite 4 numbers")



print(f"\nExercise 2: Compare different vectors")


v1 = np.array([10])

v2 = np.array([])
v3 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Longer vector




print(f"v1 = {v1}")
print(f"  v1.size = {v1.size}")
print(f"  v1.shape = {v1.shape}")
print()

print(f"v2 = {v2} (empty vector)")
print(f"  v2.size = {v2.size}")
print(f"  v2.shape = {v2.shape}")
print()

print(f"v3 = {v3}")
print(f"  v3.size = {v3.size}")
print(f"  v3.shape = {v3.shape}")



