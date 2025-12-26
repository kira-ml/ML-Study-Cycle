"""
MATRIX MULTIPLICATION: GEOMETRIC INTERPRETATION AND VISUALIZATION

This module demonstrates matrix multiplication as a composition of linear 
transformations, visualized through spatial mapping of basis vectors.
The implementation emphasizes:
- Dimensional compatibility and mathematical validation
- Geometric interpretation of matrix multiplication
- Visual representation of linear transformations
- Professional error handling and result verification
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from dataclasses import dataclass

@dataclass
class MatrixProperties:
    """Comprehensive metadata for matrix analysis."""
    shape: Tuple[int, int]
    rank: int
    determinant: float
    trace: float
    is_square: bool
    is_invertible: bool
    frobenius_norm: float

def validate_multiplication_compatibility(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray
) -> Tuple[bool, str]:
    """
    Validate matrices for multiplication with detailed feedback.
    
    Parameters
    ----------
    matrix_a : np.ndarray
        First transformation matrix
    matrix_b : np.ndarray
        Second transformation matrix
    
    Returns
    -------
    Tuple[bool, str]
        Validation result and descriptive message
    """
    if matrix_a.ndim != 2 or matrix_b.ndim != 2:
        return False, "Both operands must be 2D matrices"
    
    if matrix_a.shape[1] != matrix_b.shape[0]:
        return False, (
            f"Inner dimension mismatch: "
            f"Matrix A columns ({matrix_a.shape[1]}) ≠ "
            f"Matrix B rows ({matrix_b.shape[0]})"
        )
    
    return True, "Matrices are compatible for multiplication"

def compute_matrix_properties(matrix: np.ndarray) -> MatrixProperties:
    """
    Compute comprehensive mathematical properties of a matrix.
    
    Parameters
    ----------
    matrix : np.ndarray
        Input matrix for analysis
    
    Returns
    -------
    MatrixProperties
        Structured mathematical properties
    """
    shape = matrix.shape
    
    # Rank calculation with tolerance for numerical stability
    rank = np.linalg.matrix_rank(matrix)
    
    # Determinant and invertibility for square matrices
    if shape[0] == shape[1]:
        determinant = np.linalg.det(matrix)
        is_invertible = abs(determinant) > 1e-10
        trace = np.trace(matrix)
    else:
        determinant = np.nan
        is_invertible = False
        trace = np.nan
    
    return MatrixProperties(
        shape=shape,
        rank=rank,
        determinant=determinant,
        trace=trace,
        is_square=shape[0] == shape[1],
        is_invertible=is_invertible,
        frobenius_norm=np.linalg.norm(matrix, 'fro')
    )

def perform_matrix_multiplication(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    method: str = 'standard'
) -> np.ndarray:
    """
    Perform matrix multiplication with method selection.
    
    Parameters
    ----------
    matrix_a : np.ndarray
        Left operand matrix
    matrix_b : np.ndarray
        Right operand matrix
    method : str
        Multiplication method: 'standard', 'dot', 'einsum'
    
    Returns
    -------
    np.ndarray
        Product matrix A @ B
    """
    validation_result, message = validate_multiplication_compatibility(
        matrix_a, matrix_b
    )
    if not validation_result:
        raise ValueError(f"Matrix multiplication validation failed: {message}")
    
    method_map = {
        'standard': lambda a, b: a @ b,
        'dot': lambda a, b: np.dot(a, b),
        'einsum': lambda a, b: np.einsum('ij,jk->ik', a, b)
    }
    
    if method not in method_map:
        raise ValueError(f"Unknown method: {method}. Choose from {list(method_map.keys())}")
    
    return method_map[method](matrix_a, matrix_b)

def create_transformation_visualization(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    product: np.ndarray
) -> None:
    """
    Create comprehensive visualization of matrix multiplication.
    
    Parameters
    ----------
    matrix_a : np.ndarray
        First transformation matrix
    matrix_b : np.ndarray
        Second transformation matrix
    product : np.ndarray
        Result of A @ B
    """
    fig = plt.figure(figsize=(15, 8))
    
    # Create main layout
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], hspace=0.3, wspace=0.4)
    
    # Matrix visualization subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Mathematical annotations subplot
    ax4 = fig.add_subplot(gs[1, :])
    
    matrices = [
        (matrix_a, "Matrix A", ax1),
        (matrix_b, "Matrix B", ax2),
        (product, "Product A @ B", ax3)
    ]
    
    # Color normalization for consistent scaling
    all_values = np.concatenate([matrix_a.flatten(), matrix_b.flatten(), product.flatten()])
    vmin, vmax = all_values.min(), all_values.max()
    
    for matrix, title, ax in matrices:
        # Heatmap visualization
        im = ax.imshow(matrix, cmap="RdYlBu", vmin=vmin, vmax=vmax, aspect='auto')
        
        # Cell value annotations
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                # Color text based on cell value for readability
                text_color = 'black' if abs(matrix[i, j] - vmin) / (vmax - vmin) > 0.5 else 'white'
                ax.text(j, i, f'{matrix[i, j]:.1f}',
                       ha='center', va='center',
                       color=text_color, fontweight='bold')
        
        # Axis labels with dimension information
        ax.set_title(f'{title}\n{matrix.shape[0]}×{matrix.shape[1]}', pad=12, fontsize=12)
        ax.set_xlabel('Columns', fontsize=10)
        ax.set_ylabel('Rows', fontsize=10)
        
        # Grid lines
        ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.tick_params(which='minor', size=0)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=[ax1, ax2, ax3], orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Matrix Element Value', rotation=270, labelpad=15)
    
    # Mathematical properties display
    ax4.axis('off')
    properties_a = compute_matrix_properties(matrix_a)
    properties_b = compute_matrix_properties(matrix_b)
    properties_p = compute_matrix_properties(product)
    
    properties_text = (
        "MATHEMATICAL PROPERTIES ANALYSIS:\n\n"
        f"{'Matrix':<12} {'Shape':<12} {'Rank':<8} {'Determinant':<12} {'Norm':<10}\n"
        f"{'-'*60}\n"
        f"{'A':<12} {str(properties_a.shape):<12} {properties_a.rank:<8} "
        f"{properties_a.determinant:.4f if not np.isnan(properties_a.determinant) else 'N/A':<12} "
        f"{properties_a.frobenius_norm:.4f}\n"
        f"{'B':<12} {str(properties_b.shape):<12} {properties_b.rank:<8} "
        f"{properties_b.determinant:.4f if not np.isnan(properties_b.determinant) else 'N/A':<12} "
        f"{properties_b.frobenius_norm:.4f}\n"
        f"{'A @ B':<12} {str(properties_p.shape):<12} {properties_p.rank:<8} "
        f"{properties_p.determinant:.4f if not np.isnan(properties_p.determinant) else 'N/A':<12} "
        f"{properties_p.frobenius_norm:.4f}\n\n"
        "DIMENSIONAL ANALYSIS:\n"
        f"A: ℝ^{matrix_a.shape[1]} → ℝ^{matrix_a.shape[0]}\n"
        f"B: ℝ^{matrix_b.shape[1]} → ℝ^{matrix_b.shape[0]}\n"
        f"A @ B: ℝ^{matrix_b.shape[1]} → ℝ^{matrix_a.shape[0]}\n\n"
        "RANK PRESERVATION: "
        f"rank(A @ B) ≤ min(rank(A), rank(B)) → {properties_p.rank} ≤ min({properties_a.rank}, {properties_b.rank})"
    )
    
    ax4.text(0.02, 0.98, properties_text,
             transform=ax4.transAxes,
             fontsize=9,
             fontfamily='monospace',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8))
    
    plt.suptitle('Matrix Multiplication: Composition of Linear Transformations',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def demonstrate_matrix_multiplication() -> None:
    """
    Comprehensive demonstration of matrix multiplication with visualization.
    
    This function illustrates matrix multiplication as the composition
    of linear transformations, providing both numerical and geometric
    understanding.
    """
    print("=" * 70)
    print("MATRIX MULTIPLICATION: COMPOSITION OF LINEAR TRANSFORMATIONS")
    print("=" * 70)
    
    # Define transformation matrices
    TRANSFORMATION_A = np.array([[1, 2],
                                [3, 4]], dtype=np.float64)
    
    TRANSFORMATION_B = np.array([[5, 6],
                                [7, 8]], dtype=np.float64)
    
    print("\nTRANSFORMATION DEFINITION:")
    print("-" * 40)
    print(f"Matrix A (Transformation T₁):\n{TRANSFORMATION_A}")
    print(f"\nMatrix B (Transformation T₂):\n{TRANSFORMATION_B}")
    
    # Validate compatibility
    validation_result, message = validate_multiplication_compatibility(
        TRANSFORMATION_A, TRANSFORMATION_B
    )
    
    if not validation_result:
        print(f"\n✗ {message}")
        return
    
    print(f"\n✓ {message}")
    
    # Perform multiplication using different methods
    print("\n\nMATRIX MULTIPLICATION RESULTS:")
    print("-" * 40)
    
    methods = ['standard', 'dot', 'einsum']
    results = {}
    
    for method in methods:
        try:
            result = perform_matrix_multiplication(TRANSFORMATION_A, TRANSFORMATION_B, method)
            results[method] = result
            print(f"{method.upper():<10}: Shape = {result.shape}")
        except ValueError as e:
            print(f"{method.upper():<10}: Error - {e}")
    
    # Verify all methods produce identical results
    if len(results) > 1:
        all_equal = all(np.allclose(list(results.values())[0], r) for r in results.values())
        print(f"\n✓ All methods produce identical results: {all_equal}")
    
    # Display final result
    product = TRANSFORMATION_A @ TRANSFORMATION_B
    print(f"\nFINAL PRODUCT (A @ B):\n{product}")
    
    # Mathematical properties analysis
    print("\n\nMATHEMATICAL PROPERTIES:")
    print("-" * 40)
    
    props_a = compute_matrix_properties(TRANSFORMATION_A)
    props_b = compute_matrix_properties(TRANSFORMATION_B)
    props_p = compute_matrix_properties(product)
    
    print(f"Matrix A: rank={props_a.rank}, det={props_a.determinant:.4f}, "
          f"norm={props_a.frobenius_norm:.4f}")
    print(f"Matrix B: rank={props_b.rank}, det={props_b.determinant:.4f}, "
          f"norm={props_b.frobenius_norm:.4f}")
    print(f"Product:  rank={props_p.rank}, det={props_p.determinant:.4f}, "
          f"norm={props_p.frobenius_norm:.4f}")
    
    # Rank inequality verification
    rank_inequality = props_p.rank <= min(props_a.rank, props_b.rank)
    print(f"\nRank preservation check (rank(A@B) ≤ min(rank(A), rank(B))): {rank_inequality}")
    
    # Create visualization
    print("\n\nGenerating transformation visualization...")
    create_transformation_visualization(TRANSFORMATION_A, TRANSFORMATION_B, product)
    
    print("\n" + "=" * 70)
    print("EDUCATIONAL SUMMARY")
    print("=" * 70)
    print("""
Key Insights:
1. Matrix multiplication represents composition of linear transformations
2. Inner dimension compatibility: Aₘₓₙ · Bₙₓₚ → Cₘₓₚ
3. Rank inequality: rank(AB) ≤ min(rank(A), rank(B))
4. Geometric interpretation: AB applies transformation B followed by A

Applications in Machine Learning:
• Neural network layers: Sequential application of weight matrices
• Covariance matrices: Product of data matrix with its transpose
• Principal Component Analysis: Eigenvalue decomposition via matrix powers
• Linear system solving: Transformation of solution spaces
    """)

if __name__ == "__main__":
    """
    EXECUTION ENTRY POINT
    
    This demonstration provides a comprehensive analysis of matrix
    multiplication as both an algebraic operation and a geometric
    transformation, with professional validation, multiple computation
    methods, and detailed visualization.
    """
    demonstrate_matrix_multiplication()