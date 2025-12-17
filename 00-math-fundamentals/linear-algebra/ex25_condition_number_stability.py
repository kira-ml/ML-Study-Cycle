"""
CONDITION NUMBER SIMULATOR: ML Stability Analysis

Condition number measures sensitivity to input changes in ML models.
High condition number = Small data changes cause large prediction variations.

Created by: @kira-ml (GitHub ML Student)
#MachineLearning #DataScience #NumericalStability #MathForML
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hilbert


class ConditionNumberDemo:
    """Demonstrates condition number concepts for ML stability analysis."""
    
    def __init__(self):
        np.random.seed(42)
    
    def compute_condition_number_demo(self):
        """Compare well-conditioned and ill-conditioned matrices."""
        print("\n" + "=" * 80)
        print("EPISODE 1: STABLE VS. ILL-CONDITIONED MATRICES")
        print("=" * 80)
        
        # Example 1: Well-conditioned matrix
        print("\nEXAMPLE 1: WELL-CONDITIONED MATRIX")
        print("Stable system - small input changes yield small output changes")
        
        A1 = np.array([[2, 1], [1, 2]])
        b1 = np.array([3, 3])
        cond_A1 = np.linalg.cond(A1)
        x1 = np.linalg.solve(A1, b1)
        
        print(f"\nMatrix A:")
        print(A1)
        print(f"Condition number: {cond_A1:.2f} (Low)")
        print(f"True solution x: {x1}")
        
        # Add perturbation
        b1_perturbed = b1 + np.array([0.01, -0.01])
        x1_perturbed = np.linalg.solve(A1, b1_perturbed)
        error1 = np.linalg.norm(x1_perturbed - x1) / np.linalg.norm(x1)
        
        print(f"\nPerturbed b: {b1_perturbed}")
        print(f"New solution: {x1_perturbed}")
        print(f"Relative error: {error1:.2%}")
        print("Observation: Small input change → Small output change")
        
        # Example 2: Ill-conditioned Hilbert matrix
        print("\n\nEXAMPLE 2: ILL-CONDITIONED HILBERT MATRIX")
        print("Unstable system - known for numerical instability")
        
        n = 5
        A2 = hilbert(n)
        x_true = np.ones(n)
        b2 = A2 @ x_true
        cond_A2 = np.linalg.cond(A2)
        x2 = np.linalg.solve(A2, b2)
        
        print(f"\nHilbert Matrix (size {n}x{n}):")
        print(f"Condition number: {cond_A2:.2e} (Very High)")
        print(f"True solution x: {x_true}")
        print(f"Computed solution: {x2}")
        print(f"Relative error: {np.linalg.norm(x2 - x_true) / np.linalg.norm(x_true):.2%}")
        print("Observation: Perfect input → Significant computational error")
        
        # Example 3: Nearly singular matrix
        print("\n\nEXAMPLE 3: NEARLY SINGULAR MATRIX")
        print("Almost linearly dependent rows")
        
        A3 = np.array([[1, 1], [1, 1.0001]])
        b3 = np.array([2, 2.0001])
        cond_A3 = np.linalg.cond(A3)
        x3 = np.linalg.solve(A3, b3)
        
        print(f"\nMatrix A:")
        print(A3)
        print(f"Condition number: {cond_A3:.2f} (High)")
        print(f"Solution x: {x3}")
        
        # Add small perturbation
        b3_perturbed = b3 + np.array([0.001, 0])
        x3_perturbed = np.linalg.solve(A3, b3_perturbed)
        error3 = np.linalg.norm(x3_perturbed - x3) / np.linalg.norm(x3)
        
        print(f"\nPerturbed b: {b3_perturbed}")
        print(f"New solution: {x3_perturbed}")
        print(f"Relative error: {error3:.2%}")
        print("Observation: Tiny input change → Large output change")
    
    def condition_number_vs_error(self):
        """Visualize relationship between condition number and numerical error."""
        print("\n" + "=" * 80)
        print("EPISODE 2: CONDITION NUMBER VS. NUMERICAL ERROR")
        print("=" * 80)
        
        sizes = range(3, 12)
        condition_numbers = []
        relative_errors = []
        
        print("\nEXPERIMENT: Hilbert Matrices of Increasing Size")
        
        for n in sizes:
            A = hilbert(n)
            x_true = np.ones(n)
            b = A @ x_true
            
            cond_num = np.linalg.cond(A)
            condition_numbers.append(cond_num)
            
            x_computed = np.linalg.solve(A, b)
            error = np.linalg.norm(x_computed - x_true) / np.linalg.norm(x_true)
            relative_errors.append(error)
            
            print(f"\nn={n}x{n} Hilbert Matrix:")
            print(f"  Condition number: {cond_num:.2e}")
            print(f"  Relative error: {error:.2e}")
            
            if cond_num > 1e10:
                print("  WARNING: Condition number > 10^10 - Extreme instability")
            elif cond_num > 1e6:
                print("  WARNING: Condition number > 10^6 - High instability")
        
        # Create visualization
        plt.figure(figsize=(14, 6))
        
        # Plot 1: Condition Number Growth
        plt.subplot(1, 2, 1)
        plt.semilogy(sizes, condition_numbers, 'r^-', linewidth=2, markersize=8)
        plt.fill_between(sizes, condition_numbers, alpha=0.2, color='red')
        plt.xlabel('Matrix Size (n x n)', fontsize=11)
        plt.ylabel('Condition Number (log scale)', fontsize=11)
        plt.title('Condition Number Growth with Matrix Size', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=1e10, color='red', linestyle='--', alpha=0.5, label='Critical Zone')
        plt.axhline(y=1e6, color='orange', linestyle='--', alpha=0.5, label='Warning Zone')
        plt.legend()
        
        # Plot 2: Error Growth
        plt.subplot(1, 2, 2)
        plt.semilogy(sizes, relative_errors, 'bs-', linewidth=2, markersize=8)
        plt.fill_between(sizes, relative_errors, alpha=0.2, color='blue')
        plt.xlabel('Matrix Size (n x n)', fontsize=11)
        plt.ylabel('Relative Error (log scale)', fontsize=11)
        plt.title('Numerical Error Growth', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.suptitle('Condition Number Impact on Numerical Stability', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        print("\nKEY INSIGHT:")
        print("  Larger Hilbert matrices → Higher condition numbers → Greater numerical errors")
        print("  This demonstrates why correlated features cause instability in ML models")
    
    def stability_analysis(self):
        """Demonstrate regularization for improving stability."""
        print("\n" + "=" * 80)
        print("EPISODE 3: REGULARIZATION FOR IMPROVED STABILITY")
        print("=" * 80)
        
        # Simulate ML dataset with correlated features
        n_samples, n_features = 100, 10
        
        print("\nSETUP: Simulating ML Features with Multicollinearity")
        print("Creating correlated features (common in real datasets)")
        
        X = np.random.randn(n_samples, n_features)
        # Introduce correlation between features
        X[:, 1] = X[:, 0] + 0.01 * np.random.randn(n_samples)
        X[:, 2] = X[:, 0] + 0.02 * np.random.randn(n_samples)
        
        # Normal equation: XᵀX w = Xᵀy
        XTX = X.T @ X
        cond_original = np.linalg.cond(XTX)
        
        print(f"\nXᵀX Matrix Analysis:")
        print(f"  Condition number: {cond_original:.2e}")
        
        if cond_original > 1e10:
            print("  CRITICAL: Condition > 10^10 - Model predictions will be unstable")
        elif cond_original > 1e6:
            print("  WARNING: Condition > 10^6 - Risk of overfitting")
        else:
            print("  ACCEPTABLE: Model should be numerically stable")
        
        # Regularization experiment
        print("\n\nREGULARIZATION EXPERIMENT:")
        print("Adding λI to XᵀX (Ridge Regression)")
        print("λ = regularization strength")
        
        lambda_vals = [0, 1e-8, 1e-6, 1e-4, 1e-2]
        
        print("\n" + "-" * 70)
        print(f"{'λ':<12} {'Condition Number':<20} {'Improvement':<15} {'Effect'}")
        print("-" * 70)
        
        for lambda_val in lambda_vals:
            XTX_regularized = XTX + lambda_val * np.eye(n_features)
            cond_regularized = np.linalg.cond(XTX_regularized)
            improvement = cond_original / cond_regularized
            
            if improvement > 1000:
                effect = "Excellent stability"
            elif improvement > 100:
                effect = "Good stability"
            elif improvement > 10:
                effect = "Moderate improvement"
            elif improvement > 2:
                effect = "Slight improvement"
            else:
                effect = "Minimal effect"
            
            print(f"{lambda_val:<12.0e} {cond_regularized:<20.2e} {improvement:<15.1f}x {effect}")
        
        print("\nREGULARIZATION TRADE-OFF:")
        print("  Higher λ → Better stability but potentially biased solutions")
        print("  Lower λ → Better fit but risk of numerical instability")
        print("  Typical range: λ = 1e-4 to 1e-2 for many applications")
    
    def run(self):
        """Execute the complete demonstration."""
        print("\n" + "*" * 80)
        print("CONDITION NUMBER ANALYSIS FOR MACHINE LEARNING")
        print("Understanding Numerical Stability in ML Models")
        print("*" * 80)
        
        print("\nAuthor: @kira-ml (GitHub ML Student)")
        print("This tutorial demonstrates numerical stability concepts in ML")
        
        print("\nDEMONSTRATION OVERVIEW:")
        print("  1. Matrix Conditioning Examples")
        print("  2. Condition Number vs. Numerical Error")
        print("  3. Regularization for Stability")
        
        input("\nPress Enter to begin Part 1...")
        self.compute_condition_number_demo()
        
        input("\nPress Enter for Part 2 (visualization)...")
        self.condition_number_vs_error()
        
        input("\nPress Enter for Part 3 (ML applications)...")
        self.stability_analysis()
        
        # Conclusion
        print("\n" + "*" * 80)
        print("CONCLUSION: KEY TAKEAWAYS")
        print("*" * 80)
        
        print("\nESSENTIAL CONCEPTS:")
        print("  1. Condition number quantifies sensitivity to input changes")
        print("  2. High condition numbers lead to numerical instability")
        print("  3. Regularization improves stability at cost of some bias")
        
        print("\nPRACTICAL APPLICATIONS:")
        print("  • Check condition number of XᵀX in linear models")
        print("  • Use Ridge or Lasso regression for correlated features")
        print("  • Monitor gradient stability in neural networks")
        
        print("\nNEXT STEPS:")
        print("  1. Experiment with sklearn's Ridge() and Lasso()")
        print("  2. Analyze your datasets with np.linalg.cond()")
        print("  3. Study SVD and eigenvalue analysis for deeper understanding")
        
        print("\n" + "=" * 80)
        print("Remember: Robust ML models require numerical stability awareness")
        print("=" * 80)


def main():
    """Main execution function."""
    # Clear screen
    print("\033c", end="")
    
    # Run demonstration
    demo = ConditionNumberDemo()
    demo.run()


if __name__ == "__main__":
    main()