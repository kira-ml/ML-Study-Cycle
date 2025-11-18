import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel

def compute_gram_matrix_linear(X):
    """
    Compute the Gram matrix G = X X^T for linear kernel.
    
    Parameters:
    X : numpy array of shape (n_samples, n_features)
    
    Returns:
    G : Gram matrix of shape (n_samples, n_samples)
    """
    return X @ X.T

def verify_gram_properties(G, tolerance=1e-10):
    """
    Verify the properties of a Gram matrix:
    1. Symmetric: G = G^T
    2. Positive Semi-Definite: All eigenvalues >= 0
    
    Parameters:
    G : Gram matrix
    tolerance : numerical tolerance for comparisons
    
    Returns:
    dict : Dictionary containing verification results
    """
    results = {}
    
    # 1. Check symmetry
    is_symmetric = np.allclose(G, G.T, atol=tolerance)
    results['symmetric'] = is_symmetric
    
    # 2. Check positive semi-definite (all eigenvalues >= 0)
    eigenvalues = np.linalg.eigvalsh(G)  # Use eigh for symmetric matrices
    all_non_negative = np.all(eigenvalues >= -tolerance)
    results['positive_semi_definite'] = all_non_negative
    results['eigenvalues'] = eigenvalues
    
    # 3. Additional properties
    results['rank'] = np.linalg.matrix_rank(G)
    results['trace'] = np.trace(G)
    results['frobenius_norm'] = np.linalg.norm(G, 'fro')
    
    return results

def demonstrate_linear_gram():
    """
    Demonstrate Gram matrix computation and properties for linear kernel.
    """
    print("=" * 70)
    print("GRAM MATRIX AND KERNEL TRICK BASICS")
    print("=" * 70)
    
    # Example 1: Simple 2D dataset
    print("\n1. SIMPLE 2D DATASET")
    X_simple = np.array([[1, 2], 
                         [3, 4], 
                         [5, 6]])
    
    print(f"Input data X (3 samples, 2 features):")
    print(X_simple)
    
    # Compute Gram matrix manually and using sklearn
    G_manual = compute_gram_matrix_linear(X_simple)
    G_sklearn = linear_kernel(X_simple)
    
    print(f"\nGram matrix G = X X^T (manual computation):")
    print(G_manual)
    
    print(f"\nGram matrix G (sklearn linear_kernel):")
    print(G_sklearn)
    
    # Verify properties
    properties = verify_gram_properties(G_manual)
    print(f"\nGram Matrix Properties:")
    print(f"Symmetric: {properties['symmetric']}")
    print(f"Positive Semi-Definite: {properties['positive_semi_definite']}")
    print(f"Eigenvalues: {properties['eigenvalues']}")
    print(f"Rank: {properties['rank']}")
    print(f"Trace: {properties['trace']:.2f}")

def demonstrate_kernel_trick():
    """
    Demonstrate the kernel trick with different kernel functions.
    """
    print("\n" + "=" * 70)
    print("KERNEL TRICK DEMONSTRATION")
    print("=" + 70)
    
    # Create a non-linearly separable dataset
    np.random.seed(42)
    n_samples = 4
    X = np.random.randn(n_samples, 2)
    
    print(f"Input data X ({n_samples} samples, 2 features):")
    print(X)
    
    # Different kernel functions
    kernels = {
        'Linear': linear_kernel,
        'RBF (Gaussian)': lambda X: rbf_kernel(X, gamma=0.1),
        'Polynomial (degree=2)': lambda X: polynomial_kernel(X, degree=2)
    }
    
    for kernel_name, kernel_func in kernels.items():
        G = kernel_func(X)
        properties = verify_gram_properties(G)
        
        print(f"\n{kernel_name} Kernel Gram Matrix:")
        print(G)
        print(f"Properties - Symmetric: {properties['symmetric']}, "
              f"PSD: {properties['positive_semi_definite']}")
        print(f"Eigenvalues: {properties['eigenvalues']}")

def feature_space_visualization():
    """
    Visualize how kernels map data to feature spaces.
    """
    print("\n" + "=" * 70)
    print("FEATURE SPACE VISUALIZATION")
    print("=" + 70)
    
    # Create a simple 2D dataset that's not linearly separable
    theta = np.linspace(0, 2*np.pi, 8, endpoint=False)
    X_original = np.column_stack([np.cos(theta), np.sin(theta)])
    
    # Polynomial kernel of degree 2 feature mapping
    # For x = [x1, x2], polynomial kernel of degree 2 corresponds to:
    # φ(x) = [x1², x2², √2 x1 x2, √2 x1, √2 x2, 1]
    def polynomial_feature_map(X, degree=2):
        n_samples = X.shape[0]
        x1, x2 = X[:, 0], X[:, 1]
        
        if degree == 2:
            # Mapping for polynomial kernel of degree 2
            phi = np.column_stack([
                x1**2, x2**2, np.sqrt(2)*x1*x2, np.sqrt(2)*x1, np.sqrt(2)*x2, np.ones(n_samples)
            ])
        return phi
    
    # Compute feature mapping
    X_mapped = polynomial_feature_map(X_original)
    
    print("Original data (first 3 samples):")
    print(X_original[:3])
    print("\nMapped to feature space with polynomial kernel (degree=2, first 3 samples):")
    print(X_mapped[:3])
    
    # Verify kernel trick: φ(x_i)·φ(x_j) = K(x_i, x_j)
    G_direct = X_mapped @ X_mapped.T  # Direct computation in feature space
    G_kernel = polynomial_kernel(X_original, degree=2)  # Kernel computation
    
    print(f"\nKernel Trick Verification:")
    print(f"Direct computation in feature space (φ(x_i)·φ(x_j)):")
    print(G_direct)
    print(f"\nKernel computation K(x_i, x_j):")
    print(G_kernel)
    print(f"\nMatrices are equal: {np.allclose(G_direct, G_kernel)}")

def svm_connection_demo():
    """
    Demonstrate the connection between Gram matrices and SVMs.
    """
    print("\n" + "=" * 70)
    print("CONNECTION TO SUPPORT VECTOR MACHINES (SVMs)")
    print("=" + 70)
    
    # Create a simple binary classification dataset
    np.random.seed(42)
    n_per_class = 3
    X1 = np.random.randn(n_per_class, 2) + [2, 2]  # Class 1
    X2 = np.random.randn(n_per_class, 2) + [-2, -2]  # Class 2
    X = np.vstack([X1, X2])
    y = np.array([1, 1, 1, -1, -1, -1])  # Labels
    
    print("Binary classification dataset:")
    print(f"Features X:\n{X}")
    print(f"Labels y: {y}")
    
    # Compute Gram matrix with RBF kernel
    G_rbf = rbf_kernel(X, gamma=0.1)
    
    print(f"\nRBF Kernel Gram Matrix:")
    print(G_rbf)
    
    # Show how Gram matrix appears in SVM dual problem
    # The dual SVM problem involves the matrix y_i y_j K(x_i, x_j)
    yyT = np.outer(y, y)
    SVM_matrix = yyT * G_rbf
    
    print(f"\nSVM Dual Problem Matrix (y_i y_j K(x_i, x_j)):")
    print(SVM_matrix)
    
    properties = verify_gram_properties(SVM_matrix)
    print(f"\nSVM Matrix Properties:")
    print(f"Symmetric: {properties['symmetric']}")
    print(f"Positive Semi-Definite: {properties['positive_semi_definite']}")

def practical_insights():
    """
    Provide practical insights about Gram matrices in ML.
    """
    print("\n" + "=" * 70)
    print("PRACTICAL INSIGHTS FOR MACHINE LEARNING")
    print("=" + 70)
    
    insights = [
        "1. Gram matrices capture pairwise similarities between data points",
        "2. Kernel trick allows working in high-dimensional spaces without explicit computation",
        "3. Positive semi-definiteness ensures the kernel represents a valid feature space",
        "4. Symmetry reflects that similarity is a symmetric relationship",
        "5. In SVMs, the Gram matrix determines the optimization problem structure",
        "6. Eigenvalues of Gram matrix relate to the effective dimensionality of feature space",
        "7. Regularization is often needed for ill-conditioned Gram matrices"
    ]
    
    for insight in insights:
        print(insight)

def main():
    """
    Main function to run all demonstrations.
    """
    demonstrate_linear_gram()
    demonstrate_kernel_trick()
    feature_space_visualization()
    svm_connection_demo()
    practical_insights()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" + 70)
    print("✓ Gram matrices are symmetric and positive semi-definite")
    print("✓ They represent inner products in feature space")
    print("✓ Kernel trick enables efficient computation in high-dimensional spaces")
    print("✓ Fundamental concept for kernel methods like SVMs")
    print("=" + 70)

if __name__ == "__main__":
    main()