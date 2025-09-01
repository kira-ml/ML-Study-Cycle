import numpy as np
import matplotlib.pyplot as plt

# Define matrices
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

# Validate dimensions
if A.shape[1] != B.shape[0]:
    raise ValueError("Incompatible dimensions for matrix multiplication: A.columns must equal B.rows.")

# Matrix multiplication
result = A @ B

# Print results
print("Matrix A:\n", A)
print("Matrix B:\n", B)
print("\nResult of A @ B:\n", result)

# --- Visualization ---
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

matrices = {"Matrix A": A, "Matrix B": B, "A @ B": result}

for ax, (title, matrix) in zip(axes, matrices.items()):
    im = ax.imshow(matrix, cmap="Blues", alpha=0.8)

    # Annotate cells with values
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")

    ax.set_title(title)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_xticklabels([f"Col {j}" for j in range(matrix.shape[1])])
    ax.set_yticklabels([f"Row {i}" for i in range(matrix.shape[0])])

fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.04)
plt.suptitle("Matrix Multiplication Visualization")
plt.tight_layout()
plt.show()
