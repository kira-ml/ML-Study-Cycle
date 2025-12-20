"""
ex21_basis_and_dimension.py
-------------------------------------------------------------------------------
Fundamental Concepts in Linear Algebra: Basis, Dimension, and Rank

Author: kira-ml
Objective: Demonstrate core linear algebra concepts essential for understanding
           machine learning, neural networks, and data science.

Key Concepts Covered:
1. Basis vectors and their role in vector spaces
2. Dimension as the "degrees of freedom" in a space
3. Rank as the true dimension of a matrix's column space
4. Null space as solutions to homogeneous equations

These concepts are foundational for:
- Understanding neural network weight spaces
- Principal Component Analysis (PCA)
- Solving linear systems in optimization
- Dimensionality reduction techniques
"""

import numpy as np
from scipy.linalg import null_space

# ==============================================================================
# PART 1: BASIS VECTORS IN ℝ² - BUILDING BLOCKS OF A VECTOR SPACE
# ==============================================================================

print("=" * 70)
print("PART 1: BASIS VECTORS IN ℝ²")
print("=" * 70)

print("\nCONCEPT: A basis is a set of linearly independent vectors that")
print("         span an entire vector space. Think of them as coordinate axes.\n")

# Define two candidate vectors in ℝ²
vector_1 = np.array([1, 2])
vector_2 = np.array([3, 4])

print(f"Vector 1: {vector_1}")
print(f"Vector 2: {vector_2}")
print(f"\nGeometric interpretation:")
print(f"  • Vector 1 points to coordinates (1, 2)")
print(f"  • Vector 2 points to coordinates (3, 4)")
print(f"  • Together, they can potentially reach any point in ℝ²")

# Create matrix by stacking vectors as columns
# Why columns? Each column represents a basis vector's direction
matrix_A = np.column_stack([vector_1, vector_2])
print(f"\nMatrix A (columns = basis vectors):")
print(matrix_A)

# Calculate rank: number of linearly independent columns
rank_A = np.linalg.matrix_rank(matrix_A)
dimension_R2 = 2  # ℝ² has dimension 2 (two coordinates needed)

print(f"\nAnalysis:")
print(f"  • Rank of matrix A: {rank_A}")
print(f"  • Dimension of ℝ²: {dimension_R2}")
print(f"  • Linear independence check: Are columns independent? {rank_A == 2}")

# Basis verification
if rank_A == dimension_R2:
    print(f"\n✅ RESULT: The vectors form a basis for ℝ²!")
    print(f"   Why? Because they are linearly independent (rank = 2)")
    print(f"   and they span the entire ℝ² space.")
else:
    print(f"\n❌ RESULT: These vectors do NOT form a basis for ℝ²")
    print(f"   Why? Either they are linearly dependent or don't span ℝ²")

# ==============================================================================
# PART 2: VISUALIZING LINEAR COMBINATIONS
# ==============================================================================

print("\n" + "=" * 70)
print("PART 2: VISUALIZING SPAN AND LINEAR COMBINATIONS")
print("=" * 70)

print("\nCONCEPT: Any vector in ℝ² can be expressed as a linear combination")
print("         of basis vectors: v = c₁·v₁ + c₂·v₂\n")

# Example: Express vector [5, 6] using our basis
target_vector = np.array([5, 6])
print(f"Target vector to express: {target_vector}")

# Solve for coefficients c₁, c₂ in: c₁·v₁ + c₂·v₂ = target
# This is solving A·c = target, where A = [v₁ v₂]
coefficients = np.linalg.solve(matrix_A, target_vector)

print(f"\nSolution: {target_vector} = {coefficients[0]:.2f}·{vector_1} + {coefficients[1]:.2f}·{vector_2}")
print(f"Verification: {coefficients[0]:.2f}×{vector_1} + {coefficients[1]:.2f}×{vector_2} = {coefficients[0]*vector_1 + coefficients[1]*vector_2}")

# ==============================================================================
# PART 3: RANK AND DIMENSION OF A MATRIX
# ==============================================================================

print("\n" + "=" * 70)
print("PART 3: RANK AND DIMENSION OF A 3×3 MATRIX")
print("=" * 70)

print("\nCONCEPT: Rank = dimension of the column space (image of the matrix)")
print("         It tells us the true 'information content' of the matrix.\n")

# Create a 3×3 matrix (often represents linear transformation ℝ³ → ℝ³)
matrix_B = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print("Matrix B:")
print(matrix_B)

print("\nObservation: Notice that Row 3 = Row 1 + 2 × (Row 2 - Row 1)")
print("            This suggests linear dependence!")

# Method 1: Traditional rank calculation
rank_traditional = np.linalg.matrix_rank(matrix_B)
print(f"\n1. Traditional rank calculation: {rank_traditional}")

# Method 2: Singular Value Decomposition (SVD) - more insightful
print("\n2. Singular Value Decomposition (SVD) analysis:")
print("   SVD decomposes B into: B = U · Σ · Vᵀ")
print("   where Σ contains singular values (strength of each dimension)")

U, singular_values, Vt = np.linalg.svd(matrix_B)
print(f"\n   Singular values: {singular_values}")
print(f"   Interpretation: {singular_values[0]:.2f}, {singular_values[1]:.2f}, {singular_values[2]:.2f}")

# Determine rank from significant singular values
tolerance = 1e-10
significant_singular_values = singular_values > tolerance
rank_from_svd = np.sum(significant_singular_values)

print(f"\n   Number of significant singular values (> {tolerance}): {rank_from_svd}")
print(f"   SVD-based rank: {rank_from_svd}")

print(f"\n📊 DIMENSION ANALYSIS:")
print(f"   • Dimension of ℝ³ (ambient space): 3")
print(f"   • Rank of matrix B (column space dimension): {rank_from_svd}")
print(f"   • Dimension of null space: {matrix_B.shape[1] - rank_from_svd}")

# ==============================================================================
# PART 4: NULL SPACE - SOLUTIONS TO HOMOGENEOUS EQUATIONS
# ==============================================================================

print("\n" + "=" * 70)
print("PART 4: NULL SPACE (KERNEL) OF A MATRIX")
print("=" * 70)

print("\nCONCEPT: Null space = {x | B·x = 0}")
print("         These are directions that get mapped to zero by the transformation.\n")

# Compute null space basis vectors
null_basis = null_space(matrix_B)

print(f"Null space dimension: {null_basis.shape[1]} (expected: {matrix_B.shape[1] - rank_from_svd})")

if null_basis.shape[1] > 0:
    print(f"\nBasis for null space (columns are basis vectors):")
    print(null_basis)
    
    print(f"\nVerification that B × (null vector) = 0:")
    for i in range(null_basis.shape[1]):
        null_vector = null_basis[:, i].reshape(-1, 1)
        result = matrix_B @ null_vector
        print(f"  B × v{i+1} = {result.flatten()} ≈ 0? {np.allclose(result, 0, atol=1e-10)}")
else:
    print("\nNull space is trivial (only zero vector) - matrix has full column rank")

# ==============================================================================
# PART 5: RANK-NULLITY THEOREM DEMONSTRATION
# ==============================================================================

print("\n" + "=" * 70)
print("PART 5: RANK-NULLITY THEOREM")
print("=" * 70)

print("\nTHEOREM: For an m×n matrix A:")
print("         rank(A) + nullity(A) = n")
print("         where nullity = dimension of null space\n")

n = matrix_B.shape[1]  # number of columns
rank = rank_from_svd
nullity = null_basis.shape[1]

print(f"For our matrix B:")
print(f"  • n (number of columns): {n}")
print(f"  • rank(B): {rank}")
print(f"  • nullity(B): {nullity}")
print(f"  • rank + nullity = {rank} + {nullity} = {rank + nullity}")
print(f"  • Theorem holds: {rank + nullity == n}")

# ==============================================================================
# PART 6: PRACTICAL APPLICATION - DETECTING REDUNDANT FEATURES
# ==============================================================================

print("\n" + "=" * 70)
print("PART 6: MACHINE LEARNING APPLICATION")
print("=" * 70)

print("\nAPPLICATION: Feature matrix analysis for dimensionality reduction\n")

# Simulate a dataset with 4 features, but one is redundant
feature_matrix = np.array([
    [1, 2, 3, 4],      # Feature 4 = Feature 1 + Feature 2
    [2, 3, 4, 5],      # (linear dependence)
    [3, 4, 5, 6],
    [4, 5, 6, 7]
])

print("Feature matrix (rows = samples, columns = features):")
print(feature_matrix)

feature_rank = np.linalg.matrix_rank(feature_matrix)
print(f"\nRank of feature matrix: {feature_rank}")
print(f"Number of features: {feature_matrix.shape[1]}")
print(f"Redundant features: {feature_matrix.shape[1] - feature_rank}")

if feature_rank < feature_matrix.shape[1]:
    print("\n🔍 INSIGHT: Feature matrix has linear dependencies")
    print("            This suggests we can reduce dimensionality")
    print("            without losing information (e.g., using PCA)")

# ==============================================================================
# SUMMARY AND KEY TAKEAWAYS
# ==============================================================================

print("\n" + "=" * 70)
print("SUMMARY: KEY LINEAR ALGEBRA CONCEPTS FOR ML")
print("=" * 70)

summary_points = [
    "1. BASIS: Minimum set of vectors needed to describe a space",
    "2. DIMENSION: Number of vectors in a basis (degrees of freedom)",
    "3. RANK: True dimension of a matrix's column space",
    "4. NULL SPACE: Solutions to Ax = 0 (what gets 'lost' in transformation)",
    "5. RANK-NULLITY: Fundamental relationship between rank and nullity",
    "",
    "ML APPLICATIONS:",
    "• Feature selection: Remove linearly dependent features",
    "• PCA: Find principal components (new basis for data)",
    "• Neural networks: Weight matrices transform between vector spaces",
    "• Regularization: Adding constraints to solution space"
]

for point in summary_points:
    print(point)

print("\n" + "=" * 70)
print("END OF DEMONSTRATION - Created by kira-ml")
print("=" * 70)

# Optional: Interactive exploration
if __name__ == "__main__":
    print("\n💡 Try experimenting with different matrices!")
    print("   Suggested exercises:")
    print("   1. Change vector_2 to [2, 4] and see what happens")
    print("   2. Create a full-rank 3×3 matrix (e.g., identity matrix)")
    print("   3. Analyze a 4×2 feature matrix from a real dataset")