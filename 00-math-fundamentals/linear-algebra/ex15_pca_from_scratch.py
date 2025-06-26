import numpy as np

np.random.seed(42)
mu = [0, 0]
cov = [[4, 3], [3, 5]]
X = np.random.multivariate_normal(mu, cov, 100)

print("Shape of X:", X)
print("First 5 samples:", X[:5])


X_centered = X - np.mean(X, axis=0)

print("Shape after centering:", X_centered)
print("Mean of first feature:", np.mean(X_centered[:, 0]))


cov_matrix = np.cov(X_centered, rowvar=False, bias=1)

print("Shape of covariance matrix:", cov_matrix.shape)
print("Covariance matrix:\n", cov_matrix)


eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

print("Eigenvalues:", eigen_values)
print("Eigenvectors shape:", eigen_vectors.shape)
print("First Eigenvectors:", eigen_vectors[:, 0])


sorted_indices = np.argsort(eigen_values)[::-1]
eigen_vectors_sorted = eigen_vectors[:, sorted_indices]


k = 2
projection_matrix = eigen_vectors_sorted[:, :k]

print("Projection matrix shape:", projection_matrix.shape)
print("Top 2 eigenvectors:\n", projection_matrix)

X_pca = X_centered @ projection_matrix


print("Transformed data shape:", X_pca.shape)
print("First 5 transformed samples:\n", X_pca[:5])