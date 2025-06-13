import numpy as np

# In this implementation, I demonstrate how to compute the row echelon form of a matrix
# using elementary row operations. This technique is foundational for tasks like rank calculation,
# solving systems of linear equations, and understanding matrix invertibility.

# Define the input matrix. I use a float dtype to ensure division operations behave correctly.
A = np.array([
    [2, 4, -2],
    [4, 9, -3],
    [-2, -3, 7]
], dtype=float)

print("Original Matrix:")
print(A)

# Determine the number of rows and columns in the matrix.
rows, cols = A.shape

# `row` represents the current row we're processing as a pivot row.
row = 0

# Iterate through each column to perform forward elimination.
for col in range(cols):
    print(f"\nProcessing column {col}")

    # Identify the pivot row: the first row at or below `row` with a non-zero entry in this column.
    pivot_row = None
    for r in range(row, rows):
        if A[r, col] != 0:
            pivot_row = r
            break

    # If no pivot is found, the column is linearly dependent; skip to the next column.
    if pivot_row is None:
        print("No pivot in this column.")
        continue

    # If the current row is not already the pivot, swap it with the pivot row.
    if pivot_row != row:
        A[[row, pivot_row]] = A[[pivot_row, row]]
        print(f"Swapped row {row} with pivot row {pivot_row}")

    print(f"Matrix after potential row swap for column {col}:")
    print(A)

    # Normalize the pivot row so that the leading entry becomes 1.
    pivot_value = A[row, col]
    A[row] = A[row] / pivot_value
    print(f"Normalized row {row} (pivot = {pivot_value})")
    print(A)

    # Eliminate entries below the pivot in the current column to form an upper triangular structure.
    for r in range(row + 1, rows):
        factor = A[r, col]
        A[r] -= factor * A[row]
        print(f"Eliminated row {r} using row {row} (factor = {factor})")
        print(f"Matrix after elimination below pivot in column {col}:")
        print(A)

    # Move to the next row for the following pivot operation.
    row += 1
