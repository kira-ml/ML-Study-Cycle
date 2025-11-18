import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, make_classification
from sklearn.preprocessing import StandardScaler

def compute_covariance_matrix_manual(X):
    """
    Compute the covariance matrix manually without using np.cov.
    
    Parameters:
    X : numpy array of shape (n_samples, n_features)
    
    Returns:
    cov_matrix : covariance matrix of shape (n_features, n_features)
    """
    n_samples = X.shape[0]
    
    # Center the data (subtract mean from each feature)
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix: (X_centered.T @ X_centered) / (n_samples - 1)
    cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
    
    return cov_matrix

def compute_correlation_matrix_manual(X):
    """
    Compute the correlation matrix manually from covariance matrix.
    
    Parameters:
    X : numpy array of shape (n_samples, n_features)
    
    Returns:
    corr_matrix : correlation matrix of shape (n_features, n_features)
    """
    # Compute covariance matrix first
    cov_matrix = compute_covariance_matrix_manual(X)
    
    # Extract standard deviations (square root of diagonal elements)
    std_devs = np.sqrt(np.diag(cov_matrix))
    
    # Normalize covariance matrix to get correlation matrix
    # corr(i,j) = cov(i,j) / (std_i * std_j)
    corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
    
    return corr_matrix

def verify_covariance_properties(cov_matrix, tolerance=1e-10):
    """
    Verify properties of covariance matrix:
    1. Symmetric
    2. Positive Semi-Definite
    3. Diagonal elements are variances
    
    Parameters:
    cov_matrix : covariance matrix
    tolerance : numerical tolerance
    
    Returns:
    dict : verification results
    """
    results = {}
    
    # 1. Check symmetry
    results['symmetric'] = np.allclose(cov_matrix, cov_matrix.T, atol=tolerance)
    
    # 2. Check positive semi-definite
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    results['positive_semi_definite'] = np.all(eigenvalues >= -tolerance)
    results['eigenvalues'] = eigenvalues
    
    # 3. Diagonal elements are variances (non-negative)
    variances = np.diag(cov_matrix)
    results['variances_non_negative'] = np.all(variances >= -tolerance)
    results['variances'] = variances
    
    # 4. Additional properties
    results['trace'] = np.trace(cov_matrix)
    results['determinant'] = np.linalg.det(cov_matrix)
    results['rank'] = np.linalg.matrix_rank(cov_matrix)
    
    return results

def demonstrate_basic_example():
    """
    Demonstrate covariance and correlation computation with a simple example.
    """
    print("=" * 70)
    print("COVARIANCE AND CORRELATION MATRICES")
    print("=" + 70)
    
    # Simple example with clear relationships
    np.random.seed(42)
    n_samples = 100
    
    # Create correlated features
    x1 = np.random.randn(n_samples)
    x2 = 2 * x1 + np.random.randn(n_samples) * 0.5  # Strong positive correlation
    x3 = -1 * x1 + np.random.randn(n_samples) * 0.5  # Strong negative correlation  
    x4 = np.random.randn(n_samples)  # Uncorrelated
    
    X = np.column_stack([x1, x2, x3, x4])
    feature_names = ['x1', 'x2 (≈2*x1)', 'x3 (≈-x1)', 'x4 (noise)']
    
    print("Dataset shape:", X.shape)
    print("Feature descriptions:", feature_names)
    
    # Compute matrices manually
    cov_manual = compute_covariance_matrix_manual(X)
    corr_manual = compute_correlation_matrix_manual(X)
    
    # Compare with numpy built-in functions
    cov_numpy = np.cov(X, rowvar=False)
    corr_numpy = np.corrcoef(X, rowvar=False)
    
    print(f"\n1. COVARIANCE MATRIX (Manual Computation):")
    print(cov_manual)
    
    print(f"\n2. COVARIANCE MATRIX (NumPy np.cov):")
    print(cov_numpy)
    
    print(f"\n3. Matrices are equal: {np.allclose(cov_manual, cov_numpy)}")
    
    print(f"\n4. CORRELATION MATRIX (Manual Computation):")
    print(corr_manual)
    
    print(f"\n5. CORRELATION MATRIX (NumPy np.corrcoef):")
    print(corr_numpy)
    
    print(f"\n6. Matrices are equal: {np.allclose(corr_manual, corr_numpy)}")
    
    # Verify properties
    properties = verify_covariance_properties(cov_manual)
    print(f"\n7. COVARIANCE MATRIX PROPERTIES:")
    print(f"   Symmetric: {properties['symmetric']}")
    print(f"   Positive Semi-Definite: {properties['positive_semi_definite']}")
    print(f"   Variances non-negative: {properties['variances_non_negative']}")
    print(f"   Eigenvalues: {properties['eigenvalues']}")
    print(f"   Total variance (trace): {properties['trace']:.4f}")

def visualize_matrices():
    """
    Create heatmap visualizations of covariance and correlation matrices.
    """
    print("\n" + "=" + 70)
    print("MATRIX VISUALIZATION")
    print("=" + 70)
    
    # Use Iris dataset for demonstration
    iris = load_iris()
    X = iris.data
    feature_names = iris.feature_names
    
    # Compute matrices
    cov_matrix = compute_covariance_matrix_manual(X)
    corr_matrix = compute_correlation_matrix_manual(X)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot covariance matrix
    sns.heatmap(cov_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                xticklabels=feature_names, yticklabels=feature_names,
                center=0, ax=ax1)
    ax1.set_title('Covariance Matrix\n(Iris Dataset)')
    
    # Plot correlation matrix
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                xticklabels=feature_names, yticklabels=feature_names,
                vmin=-1, vmax=1, center=0, ax=ax2)
    ax2.set_title('Correlation Matrix\n(Iris Dataset)')
    
    plt.tight_layout()
    plt.show()
    
    print("Covariance Matrix:")
    print(cov_matrix)
    print(f"\nCorrelation Matrix:")
    print(corr_matrix)

def pca_connection_demo():
    """
    Demonstrate the connection to Principal Component Analysis (PCA).
    """
    print("\n" + "=" + 70)
    print("CONNECTION TO PRINCIPAL COMPONENT ANALYSIS (PCA)")
    print("=" + 70)
    
    # Create dataset with clear principal components
    np.random.seed(42)
    n_samples = 200
    
    # Create correlated 2D data
    theta = np.random.randn(n_samples)
    X1 = theta + np.random.randn(n_samples) * 0.1
    X2 = 2 * theta + np.random.randn(n_samples) * 0.1
    X = np.column_stack([X1, X2])
    
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov_matrix = compute_covariance_matrix_manual(X_centered)
    
    # Perform eigen decomposition (foundation of PCA)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort by descending eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print("Covariance Matrix:")
    print(cov_matrix)
    
    print(f"\nEigenvalues (variances of principal components):")
    print(eigenvalues)
    
    print(f"\nEigenvectors (principal components direction):")
    print(eigenvectors)
    
    print(f"\nTotal variance: {np.trace(cov_matrix):.4f}")
    print(f"Variance explained by PC1: {eigenvalues[0] / np.trace(cov_matrix):.2%}")
    print(f"Variance explained by PC2: {eigenvalues[1] / np.trace(cov_matrix):.2%}")
    
    # Visualization
    plt.figure(figsize=(10, 8))
    
    # Plot original data
    plt.scatter(X_centered[:, 0], X_centered[:, 1], alpha=0.6, label='Data points')
    
    # Plot principal components
    scale = 2
    for i, (eigenvalue, eigenvector) in enumerate(zip(eigenvalues, eigenvectors.T)):
        plt.arrow(0, 0, eigenvector[0] * scale * eigenvalue, 
                 eigenvector[1] * scale * eigenvalue, 
                 head_width=0.1, head_length=0.1, fc=f'C{i+1}', ec=f'C{i+1}', 
                 linewidth=2, label=f'PC{i+1} (λ={eigenvalue:.2f})')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Principal Components Analysis (PCA)\nEigenvectors of Covariance Matrix')
    plt.axis('equal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def feature_selection_insights():
    """
    Demonstrate how covariance/correlation matrices inform feature selection.
    """
    print("\n" + "=" + 70)
    print("FEATURE SELECTION INSIGHTS")
    print("=" + 70)
    
    # Create dataset with different correlation patterns
    np.random.seed(42)
    n_samples = 100
    
    # Features with different relationships
    x1 = np.random.randn(n_samples)
    x2 = 0.9 * x1 + np.random.randn(n_samples) * 0.1  # Highly correlated
    x3 = np.random.randn(n_samples)  # Independent
    x4 = -0.8 * x1 + np.random.randn(n_samples) * 0.2  # Strong negative correlation
    x5 = 0.3 * x1 + np.random.randn(n_samples) * 0.9  # Weak correlation
    
    X = np.column_stack([x1, x2, x3, x4, x5])
    feature_names = ['x1', 'x2 (high +corr)', 'x3 (independent)', 
                     'x4 (high -corr)', 'x5 (weak corr)']
    
    corr_matrix = compute_correlation_matrix_manual(X)
    
    print("Correlation Matrix:")
    print(np.round(corr_matrix, 3))
    
    print(f"\nFEATURE SELECTION ANALYSIS:")
    
    # Identify highly correlated features (potential redundancy)
    high_corr_threshold = 0.8
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            corr_val = abs(corr_matrix[i, j])
            if corr_val > high_corr_threshold:
                print(f"🚨 High correlation: {feature_names[i]} vs {feature_names[j]}: {corr_val:.3f}")
    
    # Feature importance based on variance
    variances = np.var(X, axis=0)
    print(f"\nFEATURE VARIANCES (potential importance indicator):")
    for name, var in zip(feature_names, variances):
        print(f"  {name}: {var:.4f}")

def multivariate_statistics_connection():
    """
    Show connection to multivariate statistics.
    """
    print("\n" + "=" + 70)
    print("MULTIVARIATE STATISTICS CONNECTION")
    print("=" + 70)
    
    # Generate two classes with different covariance structures
    np.random.seed(42)
    
    # Class 1: Spherical covariance
    class1 = np.random.multivariate_normal(
        mean=[0, 0], 
        cov=[[1, 0], [0, 1]], 
        size=100
    )
    
    # Class 2: Elliptical covariance (correlated features)
    class2 = np.random.multivariate_normal(
        mean=[3, 3], 
        cov=[[2, 1.5], [1.5, 2]], 
        size=100
    )
    
    X = np.vstack([class1, class2])
    y = np.hstack([np.zeros(100), np.ones(100)])
    
    # Compute within-class and between-class covariance
    cov_within_class1 = compute_covariance_matrix_manual(class1)
    cov_within_class2 = compute_covariance_matrix_manual(class2)
    
    # Pooled within-class covariance (for LDA)
    cov_pooled = (cov_within_class1 + cov_within_class2) / 2
    
    print("Class 1 Covariance Matrix:")
    print(cov_within_class1)
    print(f"\nClass 2 Covariance Matrix:")
    print(cov_within_class2)
    print(f"\nPooled Within-Class Covariance Matrix (LDA):")
    print(cov_pooled)
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(class1[:, 0], class1[:, 1], alpha=0.6, label='Class 1')
    plt.scatter(class2[:, 0], class2[:, 1], alpha=0.6, label='Class 2')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Two Classes with Different Covariance Structures')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Plot covariance ellipses
    from matplotlib.patches import Ellipse
    def plot_covariance_ellipse(mean, cov, color, alpha=0.3):
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width, height = 2 * np.sqrt(eigenvalues)
        ellipse = Ellipse(mean, width, height, angle, color=color, alpha=alpha)
        plt.gca().add_patch(ellipse)
    
    plot_covariance_ellipse(np.mean(class1, axis=0), cov_within_class1, 'blue')
    plot_covariance_ellipse(np.mean(class2, axis=0), cov_within_class2, 'red')
    plt.scatter(class1[:, 0], class1[:, 1], alpha=0.6, label='Class 1')
    plt.scatter(class2[:, 0], class2[:, 1], alpha=0.6, label='Class 2')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Covariance Ellipses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to run all demonstrations.
    """
    demonstrate_basic_example()
    visualize_matrices()
    pca_connection_demo()
    feature_selection_insights()
    multivariate_statistics_connection()
    
    print("\n" + "=" + 70)
    print("KEY INSIGHTS SUMMARY")
    print("=" + 70)
    print("✓ Covariance matrix captures feature relationships and variances")
    print("✓ Correlation matrix normalizes covariance to [-1, 1] range")  
    print("✓ Eigen decomposition of covariance matrix is foundation of PCA")
    print("✓ Correlation analysis helps in feature selection")
    print("✓ Covariance structure is crucial in multivariate statistics")
    print("✓ These concepts are fundamental for dimensionality reduction")
    print("✓ Understanding covariance helps in regularization and model design")
    print("=" + 70)

if __name__ == "__main__":
    main()