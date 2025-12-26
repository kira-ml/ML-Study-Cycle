"""
MATRIX OPERATIONS: A NUMERICAL LINEAR ALGEBRA FOUNDATION

This module demonstrates fundamental matrix operations that form the computational
basis for linear transformations in scientific computing. These operations are
essential building blocks for:
- Linear system solving (Ax = b)
- Eigenvalue decomposition in principal component analysis
- Neural network layer transformations
- Computer graphics transformation matrices
"""

import numpy as np
from typing import Tuple, Dict, Union
from dataclasses import dataclass

@dataclass
class MatrixMetadata:
    """Container for matrix properties and validation data."""
    shape: Tuple[int, int]
    dtype: np.dtype
    rank: int
    is_square: bool
    trace: float
    frobenius_norm: float

def validate_matrix_operations(matrix_a: np.ndarray, matrix_b: np.ndarray) -> None:
    """
    Validate matrices for element-wise operations.
    
    Parameters
    ----------
    matrix_a : np.ndarray
        First input matrix
    matrix_b : np.ndarray
        Second input matrix
    
    Raises
    ------
    ValueError
        If matrices have incompatible shapes for element-wise operations
    """
    if matrix_a.shape != matrix_b.shape:
        raise ValueError(
            f"Matrices must have identical shapes for element-wise operations. "
            f"Got shapes {matrix_a.shape} and {matrix_b.shape}."
        )
    
    if matrix_a.size == 0 or matrix_b.size == 0:
        raise ValueError("Matrices must be non-empty")

def compute_matrix_metadata(matrix: np.ndarray) -> MatrixMetadata:
    """
    Compute comprehensive metadata for matrix analysis.
    
    Parameters
    ----------
    matrix : np.ndarray
        Input matrix for analysis
    
    Returns
    -------
    MatrixMetadata
        Structured metadata containing mathematical properties
    """
    return MatrixMetadata(
        shape=matrix.shape,
        dtype=matrix.dtype,
        rank=np.linalg.matrix_rank(matrix),
        is_square=matrix.shape[0] == matrix.shape[1],
        trace=np.trace(matrix) if matrix.shape[0] == matrix.shape[1] else np.nan,
        frobenius_norm=np.linalg.norm(matrix, 'fro')
    )

def perform_matrix_operations(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    scalar: Union[int, float] = 2
) -> Dict[str, np.ndarray]:
    """
    Execute fundamental matrix operations with validation.
    
    Parameters
    ----------
    matrix_a : np.ndarray
        First operand matrix
    matrix_b : np.ndarray
        Second operand matrix
    scalar : Union[int, float]
        Scaling factor for scalar multiplication
    
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing results of all matrix operations
    """
    # Validate operation compatibility
    validate_matrix_operations(matrix_a, matrix_b)
    
    operations = {
        "addition": matrix_a + matrix_b,           # Element-wise sum
        "subtraction": matrix_a - matrix_b,        # Element-wise difference
        "scalar_multiplication": scalar * matrix_a, # Uniform scaling
        "hadamard_product": matrix_a * matrix_b,    # Element-wise multiplication
        "transpose_a": matrix_a.T,                 # Row-column interchange
        "transpose_b": matrix_b.T                  # Secondary transpose for comparison
    }
    
    return operations

def format_matrix_output(
    matrix: np.ndarray,
    name: str,
    metadata: MatrixMetadata = None
) -> str:
    """
    Generate formatted string representation of matrix with metadata.
    
    Parameters
    ----------
    matrix : np.ndarray
        Matrix to format
    name : str
        Descriptive name for the matrix
    metadata : MatrixMetadata, optional
        Precomputed matrix properties
    
    Returns
    -------
    str
        Formatted string with matrix data and metadata
    """
    if metadata is None:
        metadata = compute_matrix_metadata(matrix)
    
    output = [
        f"{name} ({metadata.shape[0]}×{metadata.shape[1]}):",
        f"{matrix}",
        f"Properties: rank={metadata.rank}, "
        f"norm={metadata.frobenius_norm:.4f}, "
        f"dtype={metadata.dtype}"
    ]
    
    if metadata.is_square:
        output.append(f"trace={metadata.trace:.4f}")
    
    return "\n".join(output)

def demonstrate_matrix_algebra() -> None:
    """
    Comprehensive demonstration of matrix operations with educational context.
    
    This function orchestrates a complete workflow showing:
    1. Matrix initialization and validation
    2. Fundamental algebraic operations
    3. Mathematical property analysis
    4. Professional results presentation
    """
    
    print("=" * 70)
    print("MATRIX ALGEBRA: FUNDAMENTAL OPERATIONS DEMONSTRATION")
    print("=" * 70)
    
    # Initialize demonstration matrices
    MATRIX_A = np.array([[1, 2, 3],
                         [4, 5, 6]], dtype=np.float64)
    
    MATRIX_B = np.array([[6, 5, 4],
                         [3, 2, 1]], dtype=np.float64)
    
    SCALAR_FACTOR = 2
    
    # Compute metadata for both matrices
    print("\nINITIAL MATRIX SPECIFICATIONS:")
    print("-" * 40)
    
    metadata_a = compute_matrix_metadata(MATRIX_A)
    metadata_b = compute_matrix_metadata(MATRIX_B)
    
    print(format_matrix_output(MATRIX_A, "Matrix A", metadata_a))
    print("\n" + format_matrix_output(MATRIX_B, "Matrix B", metadata_b))
    
    # Perform and display operations
    print("\n\nMATRIX OPERATIONS:")
    print("-" * 40)
    
    try:
        results = perform_matrix_operations(MATRIX_A, MATRIX_B, SCALAR_FACTOR)
        
        print("1. MATRIX ADDITION (Commutative: A + B = B + A):")
        print(f"{results['addition']}")
        print(f"   Verification: {(MATRIX_A + MATRIX_B == MATRIX_B + MATRIX_A).all()}")
        
        print("\n2. MATRIX SUBTRACTION (Non-commutative):")
        print(f"{results['subtraction']}")
        
        print(f"\n3. SCALAR MULTIPLICATION (Distributive: {SCALAR_FACTOR}×A):")
        print(f"{results['scalar_multiplication']}")
        print(f"   Linearity check: {SCALAR_FACTOR}×(A+B) = {SCALAR_FACTOR}×A + {SCALAR_FACTOR}×B")
        
        print("\n4. HADAMARD PRODUCT (Element-wise multiplication):")
        print(f"{results['hadamard_product']}")
        
        print("\n5. MATRIX TRANSPOSITION:")
        print("   Original A shape:", MATRIX_A.shape)
        print("   Transposed A shape:", results['transpose_a'].shape)
        print("   Double transpose property: (Aᵀ)ᵀ == A")
        
    except ValueError as e:
        print(f"OPERATION ERROR: {e}")
        return
    
    # Demonstrate mathematical properties
    print("\n\nMATHEMATICAL PROPERTIES DEMONSTRATION:")
    print("-" * 40)
    
    # Commutativity of addition
    commutativity = np.allclose(MATRIX_A + MATRIX_B, MATRIX_B + MATRIX_A)
    print(f"• Addition commutativity: {commutativity}")
    
    # Distributivity of scalar multiplication
    distributive = np.allclose(
        SCALAR_FACTOR * (MATRIX_A + MATRIX_B),
        SCALAR_FACTOR * MATRIX_A + SCALAR_FACTOR * MATRIX_B
    )
    print(f"• Scalar distributivity: {distributive}")
    
    # Transposition properties
    double_transpose = np.allclose(MATRIX_A, MATRIX_A.T.T)
    print(f"• Double transpose identity: {double_transpose}")
    
    print("\n" + "=" * 70)
    print("EDUCATIONAL SUMMARY")
    print("=" * 70)
    print("""
Key Concepts Demonstrated:
1. Matrix Shape Compatibility: Element-wise operations require identical dimensions
2. Operation Properties: Commutativity (addition) vs non-commutativity (subtraction)
3. Scalar Multiplication: Linear transformation preserving matrix structure
4. Transposition: Fundamental operation for changing data orientation

Applications in Machine Learning:
• Matrix addition/subtraction: Gradient updates in optimization
• Scalar multiplication: Learning rate application
• Transposition: Weight matrix alignment in neural networks
• Hadamard product: Element-wise activation functions
    """)

if __name__ == "__main__":
    """
    EXECUTION ENTRY POINT
    
    This demonstration showcases the implementation of basic matrix algebra
    with professional validation, metadata computation, and educational
    annotations. The code follows software engineering best practices while
    maintaining mathematical rigor.
    """
    
    demonstrate_matrix_algebra()
    
    # Additional educational example: Matrix shapes and broadcasting
    print("\n\n" + "=" * 70)
    print("SUPPLEMENTARY: MATRIX SHAPE CONSIDERATIONS")
    print("=" * 70)
    
    # Demonstrate shape requirements
    incompatible_a = np.array([[1, 2], [3, 4]])
    incompatible_b = np.array([[1, 2, 3], [4, 5, 6]])
    
    print("\nAttempting operation with incompatible shapes:")
    print(f"Matrix A shape: {incompatible_a.shape}")
    print(f"Matrix B shape: {incompatible_b.shape}")
    
    try:
        invalid_result = incompatible_a + incompatible_b
    except ValueError as e:
        print(f"Expected error: {e}")