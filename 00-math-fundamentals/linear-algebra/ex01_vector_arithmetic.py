import numpy as np
import matplotlib.pyplot as plt

# Define vectors (2D for visualization)
vector_a = np.array([1, 2])
vector_b = np.array([4, 5])

# Perform element-wise operations using a dictionary for scalability
operations = {
    "A + B": vector_a + vector_b,
    "A - B": vector_a - vector_b,
    "A * B": vector_a * vector_b,
    "A / B": vector_a / vector_b,
}

# Print results
print("Vector A:", vector_a)
print("Vector B:", vector_b)
for name, result in operations.items():
    print(f"{name}: {result}")

# --- Visualization (2D quiver plot) ---
plt.figure(figsize=(8, 8))
origin = np.zeros(2)

# Colors mapped to each vector
vectors = {
    "Vector A": (vector_a, "r"),
    "Vector B": (vector_b, "b"),
    "A + B": (operations["A + B"], "g"),
    "A - B": (operations["A - B"], "m"),
}

# Plot all vectors
for label, (vec, color) in vectors.items():
    plt.quiver(*origin, *vec, angles='xy', scale_units='xy', scale=1, color=color, label=label)

# Set limits dynamically (extract only vectors, not colors)
all_points = np.array([vec for vec, _ in vectors.values()])
max_val = np.max(np.abs(all_points)) + 2
plt.xlim(-max_val, max_val)
plt.ylim(-max_val, max_val)

plt.grid(True)
plt.legend()
plt.title('Vector Arithmetic Visualization')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')  # keep aspect ratio
plt.show()
