import numpy as np
import matplotlib.pyplot as plt

"""
VECTOR ARITHMETIC: A COMPUTATIONAL VISUALIZATION
This module demonstrates fundamental vector operations with visual representation.
Understanding vector arithmetic is essential for fields including:
- Machine Learning (feature transformations)
- Computer Graphics (object positioning)
- Physics Simulations (force calculations)
"""

# Initialize vectors in two-dimensional space
vector_a = np.array([1, 2])
vector_b = np.array([4, 5])

# Execute element-wise operations systematically
# Each operation produces a new vector through distinct mathematical rules
arithmetic_operations = {
    "Addition (A + B)": vector_a + vector_b,        # Component-wise sum
    "Subtraction (A - B)": vector_a - vector_b,     # Component-wise difference
    "Element-wise Multiplication (A ⊙ B)": vector_a * vector_b,  # Hadamard product
    "Element-wise Division (A ÷ B)": vector_a / vector_b,       # Component-wise quotient
}

# Display numerical results with clear formatting
print("=" * 50)
print("VECTOR ARITHMETIC: NUMERICAL RESULTS")
print("=" * 50)
print(f"Vector A: {vector_a}")
print(f"Vector B: {vector_b}")
print("-" * 30)

for operation_name, resultant_vector in arithmetic_operations.items():
    print(f"{operation_name:<35} → {resultant_vector}")

# Create visualization for spatial understanding
plt.figure(figsize=(9, 9))
plot_origin = np.zeros(2)  # All vectors originate from (0,0)

# Define visualization parameters for each vector
vector_visualizations = {
    "Vector A": (vector_a, "#FF6B6B", 1.0),      # Coral red
    "Vector B": (vector_b, "#4ECDC4", 1.0),      # Turquoise
    "Sum (A + B)": (arithmetic_operations["Addition (A + B)"], "#2ECC71", 1.0),  # Emerald
    "Difference (A - B)": (arithmetic_operations["Subtraction (A - B)"], "#9B59B6", 1.0),  # Amethyst
}

# Generate vector plot with consistent scaling
for vector_label, (vector_data, color_code, alpha_value) in vector_visualizations.items():
    plt.quiver(
        *plot_origin, *vector_data,
        angles='xy', scale_units='xy', scale=1,
        color=color_code, alpha=alpha_value,
        width=0.015, label=vector_label
    )

# Calculate dynamic axis limits for optimal visualization
all_vector_data = np.array([data for data, _, _ in vector_visualizations.values()])
maximum_magnitude = np.max(np.abs(all_vector_data)) * 1.2

plt.xlim(-maximum_magnitude, maximum_magnitude)
plt.ylim(-maximum_magnitude, maximum_magnitude)

# Configure plot aesthetics
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='upper right', framealpha=0.95)
plt.title('Vector Operations: Geometric Representation', fontsize=14, pad=20)
plt.xlabel('X Component', fontsize=11)
plt.ylabel('Y Component', fontsize=11)
plt.axis('equal')  # Maintain proportional scaling

# Add explanatory annotation
plt.annotate(
    'Note: Vectors represent magnitude and direction\n'
    'in mathematical space. Each operation creates\na new vector with distinct properties.',
    xy=(0.02, 0.02), xycoords='axes fraction',
    fontsize=9, style='italic', alpha=0.8,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.2)
)

plt.tight_layout()
plt.show()

"""
KEY EDUCATIONAL TAKEAWAYS:

1. Vector Addition: Geometrically represents the diagonal of the parallelogram
   formed by the two vectors (parallelogram law).

2. Vector Subtraction: Yields the vector connecting the tip of B to the tip of A,
   equivalent to A + (-B).

3. Element-wise Operations: Unlike dot products, these operations maintain
   the same dimensionality and operate on corresponding components.

4. Visualization Importance: Spatial representation enhances comprehension
   of abstract mathematical concepts through visual learning pathways.
"""