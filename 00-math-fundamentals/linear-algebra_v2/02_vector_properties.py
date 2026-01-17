import numpy as np

"""
NumPy Array Attributes: Understanding .size, .shape, and len()
This tutorial demonstrates key NumPy array attributes and how they differ
from standard Python operations. The examples progress from simple 1D vectors
to more complex 2D arrays.
"""

# Create a basic 1D array (vector)
v = np.array([1, 2, 3, 4, 5])

print("Array Setup:")
print(f"v = {v}")
print(f"Type of v: {type(v)} (numpy.ndarray)")
print()

# PART 1: .size - Total Element Count
# The .size attribute returns the total number of elements in the array,
# regardless of its dimensionality. This is equivalent to the product
# of all dimensions in the shape tuple.
vector_size = v.size

print("Part 1: Vector Size (.size attribute)")
print(f"v.size = {vector_size}")
print(f"Interpretation: There are {vector_size} elements in vector v")
print(f"Verification: v = {v} contains {vector_size} numbers: {list(v)}")
print()

# PART 2: .shape - Dimensional Structure
# The .shape attribute returns a tuple representing the array's dimensions.
# For a 1D array: (n,) where n is the number of elements
# For a 2D array: (rows, columns)
# For a 3D array: (depth, rows, columns), etc.
vector_shape = v.shape

print("Part 2: Vector Shape (.shape attribute)")
print(f"v.shape = {vector_shape}")
print(f"This is a {len(vector_shape)}-dimensional array")
print(f"The comma in {vector_shape} indicates it's a tuple, not an integer")

print("\nShape notation explanation:")
print(f"v.shape = {vector_shape} means:")
print(f"  - Dimensionality: 1D array")
print(f"  - Dimension size: {vector_shape[0]} elements")
print(f"  - Tuple nature: The comma indicates a single-element tuple")
print("  (Tuples are immutable sequences, unlike lists)")
print()

# PART 3: Relationship Between .size and .shape
# For any array: .size = product of all dimensions in .shape
# For 1D arrays: .size = .shape[0]
# For 2D arrays: .size = .shape[0] × .shape[1]
print("Part 3: .size vs .shape - Mathematical Relationship")
print(f"v = {v}")
print(f"v.size = {v.size}  (Total element count)")
print(f"v.shape = {v.shape}  (Dimensional structure)")

print(f"\nFor 1D vectors: v.size = v.shape[0]")
print(f"Verification: {v.size} = {v.shape[0]}")
print(f"Equality holds: {v.size == v.shape[0]}")
print()

# PART 4: NumPy Attributes vs Python's len()
# Python's len() function returns the length of the first dimension only,
# whereas .size returns the total number of elements across all dimensions.
print("Part 4: NumPy Attributes vs Python's len() Comparison")
print(f"v.size = {v.size}  (NumPy's total count across all dimensions)")
print(f"len(v) = {len(v)}  (Python's first-dimension length only)")

print("\nFor 1D vectors, both approaches yield the same result:")
print(f"len(v) = {len(v)}, v.shape[0] = {v.shape[0]}")
print(f"Equality: {len(v) == v.shape[0]}")
print("Note: For higher-dimensional arrays, len() and .size diverge")
print()

# PART 5: 2D Arrays - Critical Distinctions
# With multidimensional arrays, len() only examines the first dimension,
# making .size and .shape essential for complete understanding.
print("Part 5: 2D Array Considerations")
print("Important: len() only returns the length of the first dimension")

# Create a 2D array (matrix)
matrix = np.array([
    [1, 2, 3],   # Row 1
    [4, 5, 6]    # Row 2
])

print(f"\nmatrix = \n{matrix}")
print(f"matrix.shape = {matrix.shape}  (2 rows, 3 columns)")
print(f"matrix.size = {matrix.size}  (2 × 3 = 6 total elements)")
print(f"len(matrix) = {len(matrix)}  (Only returns 2, the row count)")

print("\nRecommendation: Use .size for total element count,")
print(".shape for dimensional structure, and be cautious with len()")
print()

# PART 6: Practical Examples and Edge Cases
print("Part 6: Practical Examples and Edge Cases")
print("=" * 60)

print("\nExample 1: Single-element vector")
v1 = np.array([10])
print(f"v1 = {v1}")
print(f"  v1.size = {v1.size} (Single element)")
print(f"  v1.shape = {v1.shape} (Note the tuple notation)")

print("\nExample 2: Empty vector")
v2 = np.array([])
print(f"v2 = {v2}")
print(f"  v2.size = {v2.size} (Zero elements)")
print(f"  v2.shape = {v2.shape} (1D array with zero length)")
print("  Caution: Operations on empty arrays may produce errors")

print("\nExample 3: Larger vector")
v3 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"v3 = {v3}")
print(f"  v3.size = {v3.size} ({len(v3)} elements total)")
print(f"  v3.shape = {v3.shape} (1D with {len(v3)} elements)")

print("\n" + "=" * 60)
print("Summary Reference:")
print("• .size: Total number of elements across all dimensions")
print("• .shape: Tuple describing array dimensions (rows, columns, ...)")
print("• len(): Length of first dimension only (standard Python behavior)")
print("• For 1D arrays: size = shape[0] = len()")
print("• For nD arrays: size = product of all shape dimensions")
print("\nBest practice: Use .size and .shape for comprehensive array analysis")