"""
Moore-Penrose Pseudoinverse: The Generalized Matrix Inverse

When regular matrix inverses fail, pseudoinverse provides solutions for 
over-determined and under-determined linear systems. Essential for linear 
regression and regularization with real-world data.

Learning Outcomes:
- Solve systems with more equations than unknowns (over-determined)
- Solve systems with fewer equations than unknowns (under-determined)
- Understand the connection to linear regression
- Compare SVD and Normal Equations methods

Created by: @kira-ml
Machine Learning Student | Numerical Linear Algebra Series
"""

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')


class PseudoinversePro:
    """
    Professional implementation of Moore-Penrose Pseudoinverse.
    
    This class demonstrates how pseudoinverse handles:
    1. Over-determined systems (more equations than unknowns)
    2. Under-determined systems (more unknowns than equations)
    3. Singular and non-square matrices
    
    Methods:
    - SVD-based pseudoinverse (numerically stable)
    - Normal equations method (traditional approach)
    - Linear regression demonstration
    - Method comparison and analysis
    """
    
    def __init__(self, method='svd'):
        """
        Initialize the pseudoinverse calculator.
        
        Parameters:
        -----------
        method : str, default='svd'
            'svd' - Singular Value Decomposition method (most stable)
            'normal' - Normal equations method (traditional)
        """
        self.method = method
        print("\n" + "=" * 60)
        print("PSEUDOINVERSE CALCULATOR INITIALIZED")
        print("=" * 60)
        print(f"Selected Method: {method.upper()}")
        print("Ready to handle non-square and singular matrices")
    
    def pseudoinverse(self, A):
        """
        Calculate Moore-Penrose Pseudoinverse A⁺.
        
        The pseudoinverse is the generalization of the matrix inverse
        for non-square matrices. It provides:
        - Least squares solution for over-determined systems
        - Minimum norm solution for under-determined systems
        
        Parameters:
        -----------
        A : numpy.ndarray
            Input matrix of shape (m, n)
            
        Returns:
        --------
        A_plus : numpy.ndarray
            Pseudoinverse of shape (n, m)
            
        Notes:
        ------
        For square invertible matrices, returns regular inverse.
        For other cases, uses SVD or Normal Equations method.
        """
        print(f"\n[Analysis] Matrix Shape: {A.shape[0]} × {A.shape[1]}")
        print(f"[Analysis] Matrix Rank: {np.linalg.matrix_rank(A)}")
        
        # Handle square matrices with regular inverse
        if A.shape[0] == A.shape[1]:
            print("[Info] Square matrix detected")
            if np.linalg.det(A) != 0:
                print("[Info] Matrix is invertible - using regular inverse")
                return np.linalg.inv(A)
            else:
                print("[Info] Square but singular - using pseudoinverse")
        
        # Dispatch to selected method
        if self.method == 'svd':
            return self._svd_pseudoinverse(A)
        else:
            return self._normal_pseudoinverse(A)
    
    def _svd_pseudoinverse(self, A):
        """
        Calculate pseudoinverse using Singular Value Decomposition.
        
        Mathematical Formulation:
        A = U Σ Vᵀ  (SVD decomposition)
        A⁺ = V Σ⁺ Uᵀ  (Pseudoinverse)
        
        where Σ⁺ is obtained by taking reciprocal of non-zero
        singular values and transposing.
        
        Advantages:
        - Handles rank-deficient matrices
        - Numerically stable
        - Reveals matrix structure
        """
        print("\n[Method] Using SVD-based pseudoinverse")
        
        # Perform SVD
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        
        print(f"  - U shape: {U.shape} (left singular vectors)")
        print(f"  - S shape: {S.shape} (singular values)")
        print(f"  - Vt shape: {Vt.shape} (right singular vectors transposed)")
        
        # Create pseudoinverse of diagonal matrix Σ
        # Use tolerance to handle numerical precision
        tol = max(A.shape) * np.finfo(S.dtype).eps * S[0]
        S_plus = np.zeros_like(S)
        mask = S > tol
        S_plus[mask] = 1 / S[mask]
        
        print(f"  - Tolerance threshold: {tol:.2e}")
        print(f"  - Non-zero singular values: {mask.sum()}/{len(S)}")
        
        # Reconstruct pseudoinverse: A⁺ = V Σ⁺ Uᵀ
        Σ_plus = np.diag(S_plus)
        A_plus = Vt.T @ Σ_plus @ U.T
        
        return A_plus
    
    def _normal_pseudoinverse(self, A):
        """
        Calculate pseudoinverse using Normal Equations.
        
        For tall matrices (m ≥ n): A⁺ = (AᵀA)⁻¹Aᵀ
        For wide matrices (m < n): A⁺ = Aᵀ(AAᵀ)⁻¹
        
        Limitations:
        - Requires full column or row rank
        - Can be numerically unstable for ill-conditioned matrices
        """
        print("\n[Method] Using Normal Equations method")
        
        m, n = A.shape
        
        if m >= n:  # Tall matrix (over-determined)
            print(f"  - Tall matrix: {m} ≥ {n}")
            print("  - Formula: A⁺ = (AᵀA)⁻¹Aᵀ")
            
            ATA = A.T @ A
            print(f"  - Condition number of AᵀA: {np.linalg.cond(ATA):.2e}")
            
            if np.linalg.matrix_rank(ATA) < n:
                print("  [Warning] AᵀA is singular - results may be unstable")
            
            return np.linalg.inv(ATA) @ A.T
        
        else:  # Wide matrix (under-determined)
            print(f"  - Wide matrix: {m} < {n}")
            print("  - Formula: A⁺ = Aᵀ(AAᵀ)⁻¹")
            
            AAT = A @ A.T
            print(f"  - Condition number of AAᵀ: {np.linalg.cond(AAT):.2e}")
            
            return A.T @ np.linalg.inv(AAT)
    
    def solve_system(self, A, b):
        """
        Solve linear system Ax = b using pseudoinverse.
        
        This method provides:
        - Least squares solution when over-determined
        - Minimum norm solution when under-determined
        - Exact solution when exactly determined
        
        Parameters:
        -----------
        A : numpy.ndarray
            Coefficient matrix
        b : numpy.ndarray
            Target vector
            
        Returns:
        --------
        x : numpy.ndarray
            Solution vector
        """
        print("\n" + "-" * 50)
        print("SOLVING LINEAR SYSTEM: Ax = b")
        print("-" * 50)
        
        m, n = A.shape
        print(f"\n[System] Dimensions: {m} equations, {n} unknowns")
        print(f"[System] Vector b length: {len(b)}")
        
        # Classify the system
        if m > n:
            print("[Type] Over-determined: More equations than unknowns")
            print("       → Computing least squares solution")
        elif m < n:
            print("[Type] Under-determined: More unknowns than equations")
            print("       → Computing minimum norm solution")
        else:
            print("[Type] Exactly determined")
            print("       → Computing exact solution (if possible)")
        
        # Calculate solution: x = A⁺ b
        A_plus = self.pseudoinverse(A)
        x = A_plus @ b
        
        print(f"\n[Solution] Vector x shape: {x.shape}")
        print(f"[Solution] First 3 values: {x[:3].round(4)}")
        
        # Calculate residual
        residual = np.linalg.norm(A @ x - b)
        print(f"[Accuracy] Residual ||Ax - b||: {residual:.6f}")
        
        if residual < 1e-10:
            print("[Status] Exact solution achieved")
        elif m > n:
            print("[Status] Least squares solution (best fit)")
        elif m < n:
            print("[Status] Minimum norm solution")
        
        return x
    
    def compare_methods(self, A):
        """
        Compare SVD and Normal Equations methods.
        
        This demonstrates numerical stability differences
        between the two approaches.
        """
        print("\n" + "=" * 50)
        print("METHOD COMPARISON: SVD vs NORMAL EQUATIONS")
        print("=" * 50)
        
        # Calculate using both methods
        print("\n[Calculation] Computing pseudoinverses...")
        
        self.method = 'svd'
        A_plus_svd = self.pseudoinverse(A)
        
        self.method = 'normal'
        A_plus_normal = self.pseudoinverse(A)
        
        # Compare results
        difference = np.linalg.norm(A_plus_svd - A_plus_normal)
        print(f"\n[Comparison] Norm difference: {difference:.2e}")
        
        if difference < 1e-12:
            print("[Result] Methods produce identical results")
        elif difference < 1e-6:
            print("[Result] Minor numerical differences")
        else:
            print("[Result] Significant differences detected")
        
        # Verify pseudoinverse properties
        print("\n[Verification] Checking pseudoinverse properties:")
        
        # Property 1: A A⁺ A should equal A
        prop1_result = A @ A_plus_svd @ A
        prop1_error = np.linalg.norm(prop1_result - A) / np.linalg.norm(A)
        print(f"  1. A A⁺ A = A: Error = {prop1_error:.2e}", 
              "✓" if prop1_error < 1e-10 else "✗")
        
        # Property 2: A⁺ A A⁺ should equal A⁺
        prop2_result = A_plus_svd @ A @ A_plus_svd
        prop2_error = np.linalg.norm(prop2_result - A_plus_svd) / np.linalg.norm(A_plus_svd)
        print(f"  2. A⁺ A A⁺ = A⁺: Error = {prop2_error:.2e}",
              "✓" if prop2_error < 1e-10 else "✗")
        
        # Property 3: (A A⁺) should be symmetric
        prop3_result = A @ A_plus_svd
        prop3_error = np.linalg.norm(prop3_result - prop3_result.T)
        print(f"  3. A A⁺ symmetric: Error = {prop3_error:.2e}",
              "✓" if prop3_error < 1e-10 else "✗")
        
        # Property 4: (A⁺ A) should be symmetric
        prop4_result = A_plus_svd @ A
        prop4_error = np.linalg.norm(prop4_result - prop4_result.T)
        print(f"  4. A⁺ A symmetric: Error = {prop4_error:.2e}",
              "✓" if prop4_error < 1e-10 else "✗")
        
        return A_plus_svd, A_plus_normal
    
    def linear_regression_demo(self):
        """
        Demonstrate that Linear Regression uses pseudoinverse.
        
        Mathematical Insight:
        Linear regression solution: β = (XᵀX)⁻¹Xᵀy
        This is equivalent to: β = X⁺ y
        
        This demonstration shows all three approaches give
        essentially the same result.
        """
        print("\n" + "=" * 60)
        print("ML APPLICATION: LINEAR REGRESSION")
        print("=" * 60)
        
        # Create synthetic regression data
        np.random.seed(42)
        n_samples = 100
        n_features = 3
        
        print(f"\n[Data] Generating synthetic dataset")
        print(f"       Samples: {n_samples}")
        print(f"       Features: {n_features}")
        print("       Adding Gaussian noise to simulate real data")
        
        # True regression coefficients
        true_beta = np.array([2.5, -1.3, 0.7])
        print(f"[Truth] True coefficients: {true_beta}")
        
        # Design matrix with intercept
        X_features = np.random.randn(n_samples, n_features - 1)
        X = np.column_stack([np.ones(n_samples), X_features])  # Add intercept
        
        # Generate target with noise
        noise = np.random.randn(n_samples) * 0.5
        y = X @ true_beta + noise
        
        print(f"\n[Methods] Comparing different approaches:")
        
        # Method 1: Direct pseudoinverse
        print("\n1. Direct Pseudoinverse:")
        print("   β = X⁺ y")
        self.method = 'svd'
        X_plus = self.pseudoinverse(X)
        beta_pseudo = X_plus @ y
        
        # Method 2: Normal equations
        print("\n2. Normal Equations (Traditional):")
        print("   β = (XᵀX)⁻¹Xᵀy")
        beta_normal = np.linalg.inv(X.T @ X) @ X.T @ y
        
        # Method 3: NumPy's built-in
        print("\n3. NumPy Least Squares:")
        print("   np.linalg.lstsq(X, y, rcond=None)")
        beta_lstsq, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        
        # Display comparison
        print("\n" + "-" * 70)
        print("COMPARISON OF REGRESSION COEFFICIENTS")
        print("-" * 70)
        
        results = []
        headers = ["Method", "Intercept", "Feature 1", "Feature 2", "Error"]
        
        for name, beta in [
            ("True Values", true_beta),
            ("Pseudoinverse", beta_pseudo),
            ("Normal Eqs", beta_normal),
            ("NumPy lstsq", beta_lstsq)
        ]:
            error = np.linalg.norm(beta - true_beta)
            results.append([name] + list(beta.round(4)) + [f"{error:.6f}"])
        
        print(tabulate(results, headers=headers, tablefmt="grid"))
        
        print("\n[Conclusion] All methods produce similar results")
        print("             Linear Regression = Pseudoinverse application")
        
        # Visualize results
        self._plot_regression_results(X, y, true_beta, beta_pseudo)
        
        return beta_pseudo
    
    def _plot_regression_results(self, X, y, true_beta, estimated_beta):
        """
        Create visualization of regression results.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Design matrix
        y : numpy.ndarray
            Target vector
        true_beta : numpy.ndarray
            True regression coefficients
        estimated_beta : numpy.ndarray
            Estimated coefficients
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: True vs Predicted values
        y_pred = X @ estimated_beta
        axes[0].scatter(y, y_pred, alpha=0.6, edgecolors='black', linewidth=0.5)
        axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 
                    'r--', alpha=0.5, label='Perfect prediction')
        axes[0].set_xlabel('True y values', fontsize=11)
        axes[0].set_ylabel('Predicted y values', fontsize=11)
        axes[0].set_title('Prediction Accuracy', fontsize=13)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot 2: Coefficient comparison
        n_coeffs = len(true_beta)
        x_positions = np.arange(n_coeffs)
        bar_width = 0.35
        
        axes[1].bar(x_positions - bar_width/2, true_beta, bar_width, 
                   label='True Coefficients', alpha=0.8)
        axes[1].bar(x_positions + bar_width/2, estimated_beta, bar_width,
                   label='Estimated Coefficients', alpha=0.8)
        axes[1].set_xlabel('Coefficient Index', fontsize=11)
        axes[1].set_ylabel('Value', fontsize=11)
        axes[1].set_title('Coefficient Estimation', fontsize=13)
        axes[1].set_xticks(x_positions)
        axes[1].set_xticklabels([f'β{i}' for i in range(n_coeffs)])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Linear Regression via Pseudoinverse', 
                    fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.show()


def run_demo():
    """
    Main demonstration function.
    
    Runs through comprehensive examples of pseudoinverse applications:
    1. Over-determined systems
    2. Under-determined systems  
    3. Method comparison
    4. Linear regression application
    """
    print("\n" + "*" * 70)
    print("MOORE-PENROSE PSEUDOINVERSE DEMONSTRATION")
    print("*" * 70)
    
    print("\nCreated by: @kira-ml")
    print("Machine Learning Student | Numerical Methods Series")
    print("\nThis demonstration covers:")
    print("  • Solving over-determined linear systems")
    print("  • Solving under-determined linear systems")
    print("  • Method comparison: SVD vs Normal Equations")
    print("  • Linear regression as pseudoinverse application")
    
    # Initialize calculator with SVD method
    pinv_calc = PseudoinversePro(method='svd')
    
    # Example 1: Over-determined system
    print("\n\n" + "=" * 60)
    print("EXAMPLE 1: OVER-DETERMINED SYSTEM")
    print("=" * 60)
    print("Finding best fit line through 4 points with 2 parameters")
    
    A1 = np.array([[1, 2], 
                   [3, 4], 
                   [5, 6], 
                   [7, 8]])
    b1 = np.array([1, 2, 3, 4])
    
    solution1 = pinv_calc.solve_system(A1, b1)
    
    # Example 2: Under-determined system
    print("\n\n" + "=" * 60)
    print("EXAMPLE 2: UNDER-DETERMINED SYSTEM")
    print("=" * 60)
    print("Infinite solutions - finding minimum norm solution")
    
    A2 = np.array([[1, 2, 3, 4]])
    b2 = np.array([10])
    
    solution2 = pinv_calc.solve_system(A2, b2)
    
    # Example 3: Method comparison
    print("\n\n" + "=" * 60)
    print("EXAMPLE 3: METHOD COMPARISON")
    print("=" * 60)
    
    A3 = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [10, 11, 12]])
    
    pinv_calc.compare_methods(A3)
    
    # Example 4: Linear regression application
    print("\n\n" + "=" * 60)
    print("EXAMPLE 4: LINEAR REGRESSION APPLICATION")
    print("=" * 60)
    
    regression_coeffs = pinv_calc.linear_regression_demo()
    
    # Summary and learning outcomes
    print("\n" + "*" * 70)
    print("LEARNING OUTCOMES SUMMARY")
    print("*" * 70)
    
    print("\n✓ Can solve over-determined systems (least squares)")
    print("✓ Can solve under-determined systems (minimum norm)")
    print("✓ Understand SVD-based pseudoinverse (numerically stable)")
    print("✓ Understand Normal Equations method (traditional)")
    print("✓ Recognize linear regression uses pseudoinverse")
    print("✓ Can compare methods and assess numerical stability")
    
    print("\n" + "-" * 70)
    print("Created by @kira-ml")
    print("GitHub: Machine Learning Student | Numerical Methods Enthusiast")
    print("-" * 70)


if __name__ == "__main__":
    # Clear console for clean output
    print("\033c", end="")
    
    # Run the demonstration
    run_demo()