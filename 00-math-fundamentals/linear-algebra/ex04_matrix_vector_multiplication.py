"""
MATRIX-VECTOR MULTIPLICATION: DUAL INTERPRETATION IMPLEMENTATION

This module demonstrates the fundamental linear transformation operation
through two complementary mathematical perspectives:
1. Computational: Row-wise dot product via np.dot()
2. Geometric: Linear combination of matrix column vectors

These interpretations form the basis for:
- Neural network forward propagation
- Linear transformations in computer graphics
- Feature mapping in dimensionality reduction
- System of linear equations representation
"""

import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass

@dataclass
class TransformationMetadata:
    """Metadata container for linear transformation analysis."""
    input_dimension: int
    output_dimension: int
    transformation_type: str
    is_consistent: bool
    column_space_rank: int
    matrix_determinant: float

def validate_transformation_compatibility(
    matrix: np.ndarray,
    vector: np.ndarray
) -> None:
    """
    Validate dimensional compatibility for matrix-vector multiplication.
    
    Parameters
    ----------
    matrix : np.ndarray
        Transformation matrix of shape (m, n)
    vector : np.ndarray
        Input vector of shape (n,) or (n, 1)
    
    Raises
    ------
    ValueError
        If inner dimensions do not match for matrix multiplication
    """
    if matrix.ndim != 2:
        raise ValueError(f"Matrix must be 2-dimensional. Got {matrix.ndim}D")
    
    if vector.ndim > 2:
        raise ValueError(f"Vector must be 1D or 2D column vector. Got {vector.ndim}D")
    
    if matrix.shape[1] != vector.shape[0]:
        raise ValueError(
            f"Dimension mismatch: matrix columns ({matrix.shape[1]}) "
            f"must equal vector length ({vector.shape[0]})"
        )

def compute_linear_combination(
    matrix: np.ndarray,
    coefficients: np.ndarray
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute linear combination of matrix columns explicitly.
    
    Parameters
    ----------
    matrix : np.ndarray
        Matrix whose columns form the basis for combination
    coefficients : np.ndarray
        Scalar weights for each column
    
    Returns
    -------
    Tuple[np.ndarray, Dict[str, np.ndarray]]
        Combined result and dictionary of scaled column vectors
    """
    scaled_columns = {}
    
    for i in range(matrix.shape[1]):
        column_label = f"column_{i+1}"
        scaled_columns[column_label] = matrix[:, i] * coefficients[i]
    
    # Sum all scaled columns to produce final linear combination
    linear_combination = sum(scaled_columns.values())
    
    return linear_combination, scaled_columns

def analyze_transformation_properties(
    matrix: np.ndarray,
    vector: np.ndarray
) -> TransformationMetadata:
    """
    Analyze mathematical properties of the linear transformation.
    
    Parameters
    ----------
    matrix : np.ndarray
        Transformation matrix
    vector : np.ndarray
        Input vector
    
    Returns
    -------
    TransformationMetadata
        Comprehensive analysis of transformation characteristics
    """
    # Compute matrix rank (dimension of column space)
    matrix_rank = np.linalg.matrix_rank(matrix)
    
    # Determine transformation type
    if matrix.shape[0] > matrix.shape[1]:
        transformation_type = "Overdetermined System"
    elif matrix.shape[0] < matrix.shape[1]:
        transformation_type = "Underdetermined System"
    else:
        transformation_type = "Square Transformation"
    
    # Check consistency (whether vector is in column space)
    column_space_consistent = matrix_rank == np.linalg.matrix_rank(
        np.column_stack([matrix, vector])
    )
    
    # Compute determinant for square matrices
    if matrix.shape[0] == matrix.shape[1]:
        determinant = np.linalg.det(matrix)
    else:
        determinant = np.nan
    
    return TransformationMetadata(
        input_dimension=matrix.shape[1],
        output_dimension=matrix.shape[0],
        transformation_type=transformation_type,
        is_consistent=column_space_consistent,
        column_space_rank=matrix_rank,
        matrix_determinant=determinant
    )

def demonstrate_dual_interpretation() -> None:
    """
    Comprehensive demonstration of matrix-vector multiplication duality.
    
    This function illustrates the equivalence between computational
    matrix multiplication and geometric linear combination interpretations.
    """
    
    print("=" * 70)
    print("MATRIX-VECTOR MULTIPLICATION: DUAL INTERPRETATION ANALYSIS")
    print("=" * 70)
    
    # Define demonstration system
    TRANSFORMATION_MATRIX = np.array([
        [2, 3],
        [4, 5],
        [6, 7]
    ], dtype=np.float64)
    
    INPUT_VECTOR = np.array([1, 2], dtype=np.float64)
    
    print("\nSYSTEM DEFINITION:")
    print("-" * 40)
    print(f"Transformation Matrix A ({TRANSFORMATION_MATRIX.shape[0]}×{TRANSFORMATION_MATRIX.shape[1]}):")
    print(TRANSFORMATION_MATRIX)
    print(f"\nInput Vector v (ℝ^{INPUT_VECTOR.shape[0]}):")
    print(INPUT_VECTOR)
    
    # Validate dimensional compatibility
    try:
        validate_transformation_compatibility(TRANSFORMATION_MATRIX, INPUT_VECTOR)
        print("\n✓ Dimension validation passed")
    except ValueError as e:
        print(f"\n✗ Validation error: {e}")
        return
    
    # Compute transformation metadata
    metadata = analyze_transformation_properties(TRANSFORMATION_MATRIX, INPUT_VECTOR)
    
    print("\n\nTRANSFORMATION ANALYSIS:")
    print("-" * 40)
    print(f"• Input dimension: ℝ^{metadata.input_dimension}")
    print(f"• Output dimension: ℝ^{metadata.output_dimension}")
    print(f"• Column space rank: {metadata.column_space_rank}")
    print(f"• Transformation type: {metadata.transformation_type}")
    print(f"• System consistency: {'Yes' if metadata.is_consistent else 'No'}")
    
    if not np.isnan(metadata.matrix_determinant):
        print(f"• Matrix determinant: {metadata.matrix_determinant:.4f}")
        if abs(metadata.matrix_determinant) < 1e-10:
            print("  Warning: Near-singular matrix (possible rank deficiency)")
    
    # Method 1: Computational approach (dot product)
    print("\n\n1. COMPUTATIONAL INTERPRETATION:")
    print("-" * 40)
    print("Matrix-vector multiplication as row-wise dot products:")
    
    computational_result = np.dot(TRANSFORMATION_MATRIX, INPUT_VECTOR)
    print(f"\nA · v = {computational_result}")
    
    # Show dot product calculations for each row
    print("\nRow-wise calculations:")
    for i in range(TRANSFORMATION_MATRIX.shape[0]):
        row_dot = np.dot(TRANSFORMATION_MATRIX[i], INPUT_VECTOR)
        print(f"  Row {i+1}: {TRANSFORMATION_MATRIX[i]} · {INPUT_VECTOR} = {row_dot:.1f}")
    
    # Method 2: Geometric approach (linear combination)
    print("\n\n2. GEOMETRIC INTERPRETATION:")
    print("-" * 40)
    print("Matrix-vector multiplication as linear combination of columns:")
    
    geometric_result, scaled_columns = compute_linear_combination(
        TRANSFORMATION_MATRIX,
        INPUT_VECTOR
    )
    
    print(f"\nColumn vectors of A:")
    for i in range(TRANSFORMATION_MATRIX.shape[1]):
        print(f"  c{i+1} = {TRANSFORMATION_MATRIX[:, i]}")
    
    print(f"\nScaled by vector components v = {INPUT_VECTOR}:")
    for label, scaled_column in scaled_columns.items():
        col_num = int(label.split('_')[1])
        print(f"  v[{col_num-1}] × c{col_num} = {INPUT_VECTOR[col_num-1]} × {TRANSFORMATION_MATRIX[:, col_num-1]}")
        print(f"                = {scaled_column}")
    
    print(f"\nLinear combination: {' + '.join(scaled_columns.keys())}")
    print(f"Result = {geometric_result}")
    
    # Verification of equivalence
    print("\n\n3. MATHEMATICAL EQUIVALENCE VERIFICATION:")
    print("-" * 40)
    
    are_equivalent = np.allclose(computational_result, geometric_result)
    print(f"✓ Computational method ≡ Geometric method: {are_equivalent}")
    
    if are_equivalent:
        print("\nBoth interpretations produce identical results, confirming that:")
        print("  A · v = Σ (v[i] × column_i(A))")
    else:
        print("\n✗ Discrepancy detected between methods")
    
    # Dimensional analysis
    print("\n\n4. DIMENSIONAL ANALYSIS:")
    print("-" * 40)
    print(f"Input space:  ℝ^{metadata.input_dimension}")
    print(f"Output space: ℝ^{metadata.output_dimension}")
    print(f"Transformation: T: ℝ^{metadata.input_dimension} → ℝ^{metadata.output_dimension}")
    
    if metadata.column_space_rank < min(TRANSFORMATION_MATRIX.shape):
        print(f"\nColumn space dimension: {metadata.column_space_rank}")
        print("Note: Transformation is not full rank")
    
    print("\n" + "=" * 70)
    print("EDUCATIONAL SUMMARY")
    print("=" * 70)
    print("""
Key Conceptual Insights:
1. Matrix-vector multiplication transforms vectors between spaces ℝⁿ → ℝᵐ
2. Computational interpretation: Dot product of matrix rows with input vector
3. Geometric interpretation: Linear combination of matrix column vectors
4. The two interpretations are mathematically equivalent (A·v = Σ vᵢcᵢ)

Applications in Machine Learning:
• Neural network layers: Weight matrices transform feature vectors
• Linear regression: Design matrix multiplied by parameter vector
• Feature extraction: Projection onto column space of transformation matrix
• Dimensionality reduction: Mapping from high- to lower-dimensional spaces
    """)

if __name__ == "__main__":
    """
    EXECUTION ENTRY POINT
    
    This demonstration provides a comprehensive analysis of matrix-vector
    multiplication from both computational and geometric perspectives,
    reinforcing fundamental linear algebra concepts through explicit
    calculation and verification.
    """
    demonstrate_dual_interpretation()
    
    # Supplementary example: Non-compatible dimensions
    print("\n\n" + "=" * 70)
    print("SUPPLEMENTARY: DIMENSION COMPATIBILITY CHECK")
    print("=" * 70)
    
    incompatible_matrix = np.array([[1, 2, 3], [4, 5, 6]])
    incompatible_vector = np.array([1, 2])  # Wrong dimension
    
    print(f"\nMatrix shape: {incompatible_matrix.shape}")
    print(f"Vector shape: {incompatible_vector.shape}")
    
    try:
        result = np.dot(incompatible_matrix, incompatible_vector)
    except ValueError as e:
        print(f"\nExpected error: {e}")
        print("Reason: Inner dimensions must match (matrix columns = vector length)")