import numpy as np
from typing import Tuple

"""
Power Iteration Method Implementation
A foundational algorithm for finding dominant eigenvalues/eigenvectors.

Core Concept: Repeated matrix-vector multiplication amplifies the 
component aligned with the largest-magnitude eigenvalue, while other 
components get suppressed.

Real-world applications: Google's PageRank, Principal Component Analysis (PCA),
spectral clustering, and vibration analysis in mechanical systems.
"""

def power_iteration(
    matrix: np.ndarray,
    max_iter: int = 100,
    tolerance: float = 1e-10,
    normalize: bool = True
) -> Tuple[float, np.ndarray]:
    """
    Computes the dominant eigenvalue and eigenvector via power iteration.
    
    Parameters:
    -----------
    matrix : np.ndarray
        Square matrix for eigenvalue analysis
    max_iter : int
        Maximum iterations before halting
    tolerance : float
        Convergence threshold for eigenvalue stability
    normalize : bool
        Whether to normalize the eigenvector each iteration
    
    Returns:
    --------
    eigenvalue : float
        Approximation of dominant eigenvalue (largest magnitude)
    eigenvector : np.ndarray
        Corresponding eigenvector approximation
    """
    
    # Input validation
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square for eigenvalue computation")
    
    # Initialize with random vector (unit norm for numerical stability)
    current_vector = np.random.randn(matrix.shape[1])
    current_vector = current_vector / np.linalg.norm(current_vector)
    
    prev_eigenvalue = 0.0
    
    for iteration in range(max_iter):
        # Core iteration: project vector onto matrix column space
        next_vector = matrix @ current_vector
        
        # Normalization prevents numerical overflow/underflow
        if normalize:
            next_vector = next_vector / np.linalg.norm(next_vector)
        
        # Rayleigh quotient: optimal eigenvalue estimate for current vector
        eigenvalue_estimate = (
            current_vector.T @ (matrix @ current_vector)
        ) / (current_vector.T @ current_vector)
        
        # Convergence check: eigenvalue change below threshold
        if np.abs(eigenvalue_estimate - prev_eigenvalue) < tolerance:
            print(f"Converged after {iteration + 1} iterations")
            break
        
        # Update for next iteration
        current_vector = next_vector
        prev_eigenvalue = eigenvalue_estimate
    
    # Final eigenvalue computation
    dominant_eigenvalue = (
        current_vector.T @ (matrix @ current_vector)
    ) / (current_vector.T @ current_vector)
    
    return dominant_eigenvalue, current_vector


def demonstrate_power_iteration() -> None:
    """
    Educational demonstration of the power iteration algorithm.
    
    Shows:
    1. Random matrix generation
    2. Power iteration execution
    3. Verification against numpy's eigen decomposition
    4. Spectral properties explanation
    """
    
    print("=" * 60)
    print("POWER ITERATION DEMONSTRATION")
    print("=" * 60)
    
    # Generate symmetric matrix (real eigenvalues guaranteed)
    dimension = 5
    random_matrix = np.random.randn(dimension, dimension)
    symmetric_matrix = (random_matrix + random_matrix.T) / 2
    
    print(f"\nMatrix Shape: {symmetric_matrix.shape}")
    print("\nGenerated Matrix (symmetric for real eigenvalues):")
    print(symmetric_matrix)
    
    # Execute power iteration
    print("\n" + "-" * 40)
    print("Executing Power Iteration...")
    print("-" * 40)
    
    eigenvalue, eigenvector = power_iteration(
        matrix=symmetric_matrix,
        max_iter=100,
        tolerance=1e-12
    )
    
    # Display results
    print(f"\n✓ Dominant Eigenvalue Estimate: {eigenvalue:.8f}")
    print(f"\n✓ Corresponding Eigenvector (unit norm):")
    print(np.round(eigenvector, 6))
    
    # Verification using numpy's eigen decomposition
    print("\n" + "-" * 40)
    print("Verification (NumPy Reference Implementation)")
    print("-" * 40)
    
    reference_eigenvalues, reference_eigenvectors = np.linalg.eig(symmetric_matrix)
    dominant_idx = np.argmax(np.abs(reference_eigenvalues))
    
    print(f"\nNumPy Dominant Eigenvalue: {reference_eigenvalues[dominant_idx]:.8f}")
    print(f"Absolute Error: {np.abs(eigenvalue - reference_eigenvalues[dominant_idx]):.2e}")
    
    # Eigenvector alignment check (absolute value for sign invariance)
    alignment = np.abs(np.dot(eigenvector, reference_eigenvectors[:, dominant_idx]))
    print(f"Eigenvector Alignment: {alignment:.8f} (1.0 = perfect match)")
    
    # Spectral gap analysis
    sorted_eigenvalues = np.sort(np.abs(reference_eigenvalues))[::-1]
    spectral_gap = sorted_eigenvalues[0] - sorted_eigenvalues[1]
    
    print(f"\n📊 Spectral Analysis:")
    print(f"   Spectral Gap: {spectral_gap:.4f}")
    print(f"   Convergence Rate: ~{spectral_gap/sorted_eigenvalues[0]:.4f}")
    print("   Note: Larger spectral gap → faster convergence")


def explain_convergence() -> None:
    """
    Educational explanation of power iteration convergence properties.
    """
    print("\n" + "=" * 60)
    print("CONVERGENCE ANALYSIS")
    print("=" * 60)
    
    concepts = {
        "Why It Works": (
            "Repeated multiplication amplifies the dominant eigenvector component "
            "while suppressing others (geometric progression effect)."
        ),
        "Convergence Rate": (
            "Depends on |λ₂/λ₁| where λ₁ is dominant, λ₂ is second largest. "
            "Smaller ratio → faster convergence."
        ),
        "Numerical Stability": (
            "Regular normalization prevents overflow/underflow and maintains "
            "numerical precision throughout iterations."
        ),
        "Limitations": (
            "Only finds dominant eigenvalue. Requires |λ₁| > |λ₂|. "
            "May converge slowly for nearly degenerate eigenvalues."
        )
    }
    
    for concept, explanation in concepts.items():
        print(f"\n• {concept}:")
        print(f"  {explanation}")


if __name__ == "__main__":
    """
    Main execution block demonstrating professional implementation.
    """
    np.random.seed(42)  # Reproducibility for educational purposes
    
    demonstrate_power_iteration()
    explain_convergence()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
    1. Power iteration is computationally efficient: O(n²) per iteration
    2. Memory efficient: only stores current vector, not full history
    3. Foundation for advanced methods (inverse iteration, QR algorithm)
    4. Understanding this method builds intuition for spectral methods
    """)