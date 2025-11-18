import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hilbert

def compute_condition_number_demo():
    """
    Demonstrate how condition number affects the sensitivity of linear system solutions.
    """
    print("=" * 60)
    print("Condition Number and Numerical Stability Demo")
    print("=" * 60)
    
    # Example 1: Well-conditioned matrix
    print("\n1. WELL-CONDITIONED MATRIX")
    A1 = np.array([[2, 1], [1, 2]])
    b1 = np.array([3, 3])
    
    cond_A1 = np.linalg.cond(A1)
    x1 = np.linalg.solve(A1, b1)
    
    print(f"Matrix A:\n{A1}")
    print(f"Condition number: {cond_A1:.6f}")
    print(f"True solution x: {x1}")
    
    # Add small perturbation to b
    b1_perturbed = b1 + np.array([0.01, -0.01])
    x1_perturbed = np.linalg.solve(A1, b1_perturbed)
    error1 = np.linalg.norm(x1_perturbed - x1) / np.linalg.norm(x1)
    
    print(f"Perturbed solution: {x1_perturbed}")
    print(f"Relative error: {error1:.6f}")
    
    # Example 2: Ill-conditioned matrix (Hilbert matrix)
    print("\n2. ILL-CONDITIONED MATRIX (Hilbert)")
    n = 5
    A2 = hilbert(n)  # Hilbert matrices are famously ill-conditioned
    x_true = np.ones(n)
    b2 = A2 @ x_true
    
    cond_A2 = np.linalg.cond(A2)
    x2 = np.linalg.solve(A2, b2)
    
    print(f"Condition number: {cond_A2:.2e}")
    print(f"True solution x: {x_true}")
    print(f"Computed solution: {x2}")
    print(f"Relative error: {np.linalg.norm(x2 - x_true) / np.linalg.norm(x_true):.6f}")
    
    # Example 3: Nearly singular matrix
    print("\n3. NEARLY SINGULAR MATRIX")
    A3 = np.array([[1, 1], [1, 1.0001]])
    b3 = np.array([2, 2.0001])
    
    cond_A3 = np.linalg.cond(A3)
    x3 = np.linalg.solve(A3, b3)
    
    print(f"Matrix A:\n{A3}")
    print(f"Condition number: {cond_A3:.2f}")
    print(f"Solution x: {x3}")
    
    # Small perturbation
    b3_perturbed = b3 + np.array([0.001, 0])
    x3_perturbed = np.linalg.solve(A3, b3_perturbed)
    error3 = np.linalg.norm(x3_perturbed - x3) / np.linalg.norm(x3)
    
    print(f"After perturbation:")
    print(f"New solution: {x3_perturbed}")
    print(f"Relative error: {error3:.6f}")

def condition_number_vs_error():
    """
    Plot how condition number relates to solution error.
    """
    print("\n" + "=" * 60)
    print("Condition Number vs Solution Error")
    print("=" * 60)
    
    sizes = range(3, 12)
    condition_numbers = []
    relative_errors = []
    
    for n in sizes:
        # Create Hilbert matrix of size n (ill-conditioned)
        A = hilbert(n)
        x_true = np.ones(n)
        b = A @ x_true
        
        # Compute condition number
        cond_num = np.linalg.cond(A)
        condition_numbers.append(cond_num)
        
        # Solve system and compute error
        x_computed = np.linalg.solve(A, b)
        error = np.linalg.norm(x_computed - x_true) / np.linalg.norm(x_true)
        relative_errors.append(error)
        
        print(f"n={n}: Condition number = {cond_num:.2e}, Relative error = {error:.2e}")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.semilogy(sizes, condition_numbers, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Matrix Size n')
    plt.ylabel('Condition Number (log scale)')
    plt.title('Condition Number of Hilbert Matrices')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.semilogy(sizes, relative_errors, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Matrix Size n')
    plt.ylabel('Relative Error (log scale)')
    plt.title('Solution Error for Hilbert Systems')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def stability_analysis():
    """
    Analyze numerical stability in practical ML scenarios.
    """
    print("\n" + "=" * 60)
    print("ML Relevance: Regularization and Stability")
    print("=" * 60)
    
    # Simulate an ML feature matrix
    np.random.seed(42)
    n_samples, n_features = 100, 10
    
    # Create correlated features (common in ML)
    X = np.random.randn(n_samples, n_features)
    # Make some features highly correlated
    X[:, 1] = X[:, 0] + 0.01 * np.random.randn(n_samples)
    X[:, 2] = X[:, 0] + 0.02 * np.random.randn(n_samples)
    
    # Normal equation: X^T X w = X^T y
    XTX = X.T @ X
    cond_original = np.linalg.cond(XTX)
    
    print(f"Condition number of X^T X: {cond_original:.2e}")
    
    # Regularization (Ridge regression)
    lambda_vals = [0, 1e-8, 1e-6, 1e-4, 1e-2]
    
    print("\nEffect of Regularization on Condition Number:")
    print("Lambda\t\tCondition Number\tImprovement")
    print("-" * 50)
    
    for lambda_val in lambda_vals:
        XTX_regularized = XTX + lambda_val * np.eye(n_features)
        cond_regularized = np.linalg.cond(XTX_regularized)
        improvement = cond_original / cond_regularized
        
        print(f"{lambda_val:.0e}\t\t{cond_regularized:.2e}\t\t{improvement:.2f}x")

def main():
    """
    Main function to run all demonstrations.
    """
    compute_condition_number_demo()
    condition_number_vs_error()
    stability_analysis()
    
    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("1. High condition numbers amplify errors in linear systems")
    print("2. Ill-conditioned matrices are sensitive to small perturbations")
    print("3. Regularization improves numerical stability in ML")
    print("4. Condition number > 1e10 often indicates serious numerical issues")
    print("=" * 60)

if __name__ == "__main__":
    main()