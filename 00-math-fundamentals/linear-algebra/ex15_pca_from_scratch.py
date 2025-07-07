import numpy as np

"""
Principal Component Analysis (PCA) Implementation

This implementation demonstrates PCA from first principles, showing:
1. Data generation and preprocessing
2. Covariance matrix computation
3. Eigen decomposition
4. Dimensionality reduction

I use NumPy for all linear algebra operations to ensure efficiency.
"""

# I set a random seed for reproducible results
np.random.seed(42)

"""
Generate synthetic correlated 2D data:
- Mean centered at origin [0, 0]
- Custom covariance structure showing feature correlation
- 100 samples for clear visualization
"""
mu = [0, 0]
cov = [[4, 3], [3, 5]]  # Positive covariance indicates correlated features
X = np.random.multivariate_normal(mu, cov, 100)

print("Shape of X:", X.shape)  # Verify (100 samples × 2 features)
print("First 5 samples:\n", X[:5])  # Inspect raw data distribution

"""
Data Centering:
Critical for PCA as it makes the mean the origin point.
I subtract the mean along each feature dimension (axis=0).
This doesn't affect covariance but ensures first PC captures max variance.
"""
X_centered = X - np.mean(X, axis=0)

print("\nAfter centering:")
print("Mean of first feature:", np.mean(X_centered[:, 0]))  # Should be ~0
print("Mean of second feature:", np.mean(X_centered[:, 1]))  # Should be ~0

"""
Covariance Matrix Computation:
- rowvar=False indicates columns represent features
- bias=1 uses population covariance (N denominator)
This matrix captures how features vary together.
"""
cov_matrix = np.cov(X_centered, rowvar=False, bias=1)

print("\nCovariance matrix:")
print("Shape:", cov_matrix.shape)  # Should be (2×2) for our 2 features
print("Values:\n", cov_matrix)  # Should approximate our input cov matrix

"""
Eigen Decomposition:
- Eigenvalues represent variance explained by each component
- Eigenvectors define the directions of maximum variance
I use NumPy's linalg.eig which returns normalized eigenvectors.
"""
eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

print("\nEigenanalysis results:")
print("Eigenvalues:", eigen_values)  # Component variances
print("Eigenvectors:\n", eigen_vectors)  # Principal directions

"""
Sort Components by Explained Variance:
- argsort gives ascending order, so I reverse with [::-1]
- This ensures first PC (index 0) has largest eigenvalue
"""
sorted_indices = np.argsort(eigen_values)[::-1]
eigen_vectors_sorted = eigen_vectors[:, sorted_indices]

"""
Projection Matrix Construction:
I keep all components (k=2) for demonstration, though in practice
we might reduce dimensionality by choosing k < original features.
"""
k = 2
projection_matrix = eigen_vectors_sorted[:, :k]

print("\nProjection matrix:")
print("Shape:", projection_matrix.shape)  # (2×2) for full projection
print("Matrix:\n", projection_matrix)  # Columns are principal components

"""
Data Transformation:
Project centered data onto principal components.
This rotates the data to align with directions of max variance.
"""
X_pca = X_centered @ projection_matrix

print("\nTransformed data:")
print("Shape:", X_pca.shape)  # Same sample count, new feature space
print("First 5 samples:\n", X_pca[:5])  # Data in PC coordinates