"""
MATRIX DETERMINANT: COMPUTATIONAL METHODS AND MATHEMATICAL PROPERTIES

This module demonstrates determinant calculation through multiple methodologies,
providing insight into both theoretical foundations and practical implementations.
The determinant is analyzed as:
1. A scalar invariant encoding linear transformation properties
2. A computational primitive for matrix invertibility testing
3. A geometric measure of volume scaling and orientation

Applications in machine learning include:
- Covariance matrix analysis in multivariate statistics
- Jacobian determinant computation in normalizing flows
- Eigenvalue decomposition in principal component analysis
- Numerical stability assessment in optimization algorithms
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum

class DeterminantMethod(Enum):
    """Enumeration of determinant computation methodologies."""
    NUMPY = "numpy"           # LU decomposition with partial pivoting
    COFACTOR = "cofactor"     # Laplace expansion (theoretical demonstration)
    LAPLACE = "laplace"       # Generalized Laplace expansion
    RECURSIVE = "recursive"   # Divide-and-conquer approach

@dataclass
class MatrixProperties:
    """Comprehensive mathematical properties of a matrix."""
    determinant: float
    is_singular: bool
    is_invertible: bool
    condition_number: float
    trace: float
    rank: int
    volume_scaling: float
    orientation_preserved: bool

def validate_square_matrix(matrix: np.ndarray) -> Tuple[bool, str]:
    """
    Validate that a matrix is square and suitable for determinant computation.
    
    Parameters
    ----------
    matrix : np.ndarray
        Input matrix to validate
    
    Returns
    -------
    Tuple[bool, str]
        Validation result and descriptive message
    """
    if matrix.ndim != 2:
        return False, f"Matrix must be 2-dimensional. Got {matrix.ndim}D"
    
    if matrix.shape[0] != matrix.shape[1]:
        return False, (
            f"Matrix must be square. Got shape {matrix.shape} "
            f"(rows={matrix.shape[0]}, columns={matrix.shape[1]})"
        )
    
    if matrix.size == 0:
        return False, "Matrix is empty"
    
    return True, "Matrix is valid for determinant computation"

def compute_determinant_2x2(matrix: np.ndarray) -> float:
    """
    Compute determinant of 2x2 matrix using closed-form formula.
    
    For matrix: [[a, b],
                 [c, d]]
    Determinant: ad - bc
    
    This serves as the base case for recursive determinant algorithms.
    """
    if matrix.shape != (2, 2):
        raise ValueError(f"Matrix must be 2x2. Got shape {matrix.shape}")
    
    return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]

def compute_determinant_cofactor(matrix: np.ndarray, 
                                expansion_row: int = 0) -> Tuple[float, List[dict]]:
    """
    Compute determinant via cofactor expansion with detailed step tracking.
    
    Parameters
    ----------
    matrix : np.ndarray
        Square matrix for determinant computation
    expansion_row : int
        Row index for Laplace expansion (default: first row)
    
    Returns
    -------
    Tuple[float, List[dict]]
        Determinant value and list of computational steps for analysis
    """
    n = matrix.shape[0]
    
    # Base case: 2x2 matrix
    if n == 2:
        determinant = compute_determinant_2x2(matrix)
        steps = [{
            'matrix_size': 2,
            'method': 'direct',
            'determinant': determinant
        }]
        return determinant, steps
    
    determinant = 0.0
    steps = []
    
    # Laplace expansion along specified row
    for col in range(n):
        # Cofactor sign: (-1)^(row + col)
        sign = 1 if (expansion_row + col) % 2 == 0 else -1
        
        # Extract minor by removing row and column
        minor_rows = [i for i in range(n) if i != expansion_row]
        minor_cols = [j for j in range(n) if j != col]
        minor = matrix[np.ix_(minor_rows, minor_cols)]
        
        # Recursively compute minor determinant
        minor_det, minor_steps = compute_determinant_cofactor(minor, 0)
        
        # Cofactor contribution
        cofactor = sign * matrix[expansion_row, col] * minor_det
        determinant += cofactor
        
        # Record step for educational analysis
        steps.append({
            'expansion_row': expansion_row,
            'expansion_col': col,
            'element': matrix[expansion_row, col],
            'sign': sign,
            'minor_size': minor.shape[0],
            'minor_determinant': minor_det,
            'cofactor_contribution': cofactor,
            'cumulative_sum': determinant,
            'minor_steps': minor_steps
        })
    
    return determinant, steps

def analyze_matrix_properties(matrix: np.ndarray, 
                            determinant: float) -> MatrixProperties:
    """
    Compute comprehensive mathematical properties based on determinant.
    
    Parameters
    ----------
    matrix : np.ndarray
        Original matrix
    determinant : float
        Computed determinant value
    
    Returns
    -------
    MatrixProperties
        Structured analysis of matrix characteristics
    """
    # Determine singularity threshold
    SINGULARITY_THRESHOLD = 1e-10
    is_singular = abs(determinant) < SINGULARITY_THRESHOLD
    
    # Compute condition number (measure of numerical stability)
    try:
        condition_number = np.linalg.cond(matrix)
    except np.linalg.LinAlgError:
        condition_number = np.inf
    
    # Volume scaling factor (absolute determinant)
    volume_scaling = abs(determinant)
    
    # Orientation preservation (sign of determinant)
    orientation_preserved = determinant > 0
    
    return MatrixProperties(
        determinant=determinant,
        is_singular=is_singular,
        is_invertible=not is_singular,
        condition_number=condition_number,
        trace=np.trace(matrix),
        rank=np.linalg.matrix_rank(matrix),
        volume_scaling=volume_scaling,
        orientation_preserved=orientation_preserved
    )

def format_cofactor_expansion_steps(steps: List[dict], depth: int = 0) -> str:
    """
    Generate formatted output of cofactor expansion steps.
    
    Parameters
    ----------
    steps : List[dict]
        Step-by-step computation records
    depth : int
        Recursion depth for indentation
    
    Returns
    -------
    str
        Formatted string representation of computational steps
    """
    indent = "  " * depth
    
    if not steps:
        return ""
    
    output_lines = []
    
    for i, step in enumerate(steps):
        line = (
            f"{indent}Step {i+1}: Expand at element a[{step['expansion_row']+1},{step['expansion_col']+1}] = {step['element']}\n"
            f"{indent}  Sign: {'+' if step['sign'] == 1 else '-'} (cofactor = (-1)^({step['expansion_row']+1}+{step['expansion_col']+1}))\n"
            f"{indent}  Minor determinant: {step['minor_determinant']}\n"
            f"{indent}  Contribution: {step['sign']} × {step['element']} × {step['minor_determinant']} = {step['cofactor_contribution']}\n"
            f"{indent}  Cumulative sum: {step['cumulative_sum']}"
        )
        output_lines.append(line)
        
        # Recursively format minor steps
        if step.get('minor_steps'):
            output_lines.append(f"\n{indent}  Minor matrix steps:")
            output_lines.append(format_cofactor_expansion_steps(step['minor_steps'], depth + 2))
    
    return "\n".join(output_lines)

def demonstrate_determinant_computation() -> None:
    """
    Comprehensive demonstration of determinant computation methodologies.
    
    This function illustrates multiple approaches to determinant calculation,
    providing both theoretical insight and practical implementation guidance.
    """
    print("=" * 70)
    print("DETERMINANT COMPUTATION: THEORETICAL AND NUMERICAL ANALYSIS")
    print("=" * 70)
    
    # Define demonstration matrices
    MATRIX_2x2 = np.array([[1, 2],
                          [3, 4]], dtype=np.float64)
    
    MATRIX_3x3 = np.array([[1, 2, 3],
                          [0, 1, 4],
                          [5, 6, 0]], dtype=np.float64)
    
    MATRIX_4x4 = np.array([[1, 2, 3, 4],
                          [0, 1, 0, 2],
                          [2, 0, 1, 3],
                          [1, 1, 1, 1]], dtype=np.float64)
    
    matrices = {
        "2×2 Matrix": MATRIX_2x2,
        "3×3 Matrix": MATRIX_3x3,
        "4×4 Matrix": MATRIX_4x4
    }
    
    for name, matrix in matrices.items():
        print(f"\n{'='*60}")
        print(f"ANALYZING {name.upper()}")
        print(f"{'='*60}")
        
        print(f"\nMatrix:\n{matrix}")
        
        # Validate matrix
        is_valid, message = validate_square_matrix(matrix)
        if not is_valid:
            print(f"\n✗ Validation failed: {message}")
            continue
        
        print(f"\n✓ {message}")
        print(f"Matrix shape: {matrix.shape[0]}×{matrix.shape[1]}")
        
        # Method 1: NumPy optimized computation
        print("\n1. NUMPY OPTIMIZED COMPUTATION (LU DECOMPOSITION):")
        print("-" * 40)
        
        try:
            det_numpy = np.linalg.det(matrix)
            print(f"   Determinant: {det_numpy:.8f}")
        except np.linalg.LinAlgError as e:
            print(f"   Computation failed: {e}")
            det_numpy = None
        
        # Method 2: Cofactor expansion (educational)
        print("\n2. COFACTOR EXPANSION (EDUCATIONAL DEMONSTRATION):")
        print("-" * 40)
        
        if matrix.shape[0] <= 4:  # Limit for demonstration
            try:
                det_cofactor, steps = compute_determinant_cofactor(matrix)
                print(f"   Determinant: {det_cofactor:.8f}")
                print(f"\n   Step-by-step expansion:")
                print(format_cofactor_expansion_steps(steps))
                
                # Verify equivalence with NumPy result
                if det_numpy is not None:
                    difference = abs(det_numpy - det_cofactor)
                    print(f"\n   Numerical equivalence check:")
                    print(f"   |NumPy - Cofactor| = {difference:.2e}")
                    print(f"   Within tolerance (1e-10): {difference < 1e-10}")
            except Exception as e:
                print(f"   Cofactor expansion failed: {e}")
        else:
            print("   (Cofactor demonstration limited to matrices ≤ 4×4)")
        
        # Method 3: Direct formula for 2x2
        if matrix.shape == (2, 2):
            print("\n3. DIRECT 2×2 FORMULA:")
            print("-" * 40)
            det_direct = compute_determinant_2x2(matrix)
            print(f"   Determinant: {det_direct:.8f}")
            print(f"   Formula: a₁₁·a₂₂ - a₁₂·a₂₁ = {matrix[0,0]}×{matrix[1,1]} - {matrix[0,1]}×{matrix[1,0]}")
        
        # Matrix properties analysis
        if det_numpy is not None:
            print("\n4. MATHEMATICAL PROPERTIES ANALYSIS:")
            print("-" * 40)
            
            properties = analyze_matrix_properties(matrix, det_numpy)
            
            print(f"   Singular: {properties.is_singular}")
            print(f"   Invertible: {properties.is_invertible}")
            print(f"   Condition number: {properties.condition_number:.2e}")
            print(f"   Volume scaling factor: {properties.volume_scaling:.4f}")
            print(f"   Orientation preserved: {properties.orientation_preserved}")
            print(f"   Matrix rank: {properties.rank}")
            print(f"   Matrix trace: {properties.trace}")
    
    # Computational complexity analysis
    print(f"\n{'='*60}")
    print("COMPUTATIONAL COMPLEXITY ANALYSIS")
    print(f"{'='*60}")
    
    print("""
Methodology Comparison:
• NumPy (LU decomposition): O(n³) operations, numerically stable
• Cofactor expansion: O(n!) operations, exponential complexity
• Recursive algorithms: O(n³) with proper implementation

Practical Considerations:
1. Use NumPy for all production computations (numerical stability)
2. Cofactor expansion is primarily educational (theoretical insight)
3. Determinant magnitude indicates numerical conditioning
4. Small determinants (< 1e-10) suggest potential numerical issues
    """)
    
    # Geometric interpretation demonstration
    print(f"\n{'='*60}")
    print("GEOMETRIC INTERPRETATION")
    print(f"{'='*60}")
    
    # Create a simple transformation matrix
    theta = np.pi / 6  # 30 degrees
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    scaling_matrix = np.array([[2, 0],
                              [0, 3]])
    
    composite_matrix = rotation_matrix @ scaling_matrix
    
    for mat_name, matrix in [("Rotation", rotation_matrix),
                            ("Scaling", scaling_matrix),
                            ("Composite", composite_matrix)]:
        det = np.linalg.det(matrix)
        print(f"\n{mat_name} Matrix (determinant = {det:.4f}):")
        print(f"• Area scaling: {abs(det):.2f}×")
        print(f"• Orientation: {'preserved' if det > 0 else 'reversed'}")
    
    print(f"\n{'='*70}")
    print("EDUCATIONAL SUMMARY")
    print(f"{'='*70}")
    print("""
Key Mathematical Insights:
1. Determinant Properties:
   - Multiplicative: det(AB) = det(A)det(B)
   - Scaling: det(kA) = kⁿ det(A) for n×n matrix
   - Transpose: det(Aᵀ) = det(A)

2. Geometric Significance:
   - Absolute value: Volume scaling factor
   - Sign: Orientation preservation/reversal
   - Zero determinant: Collapsed dimension

3. Machine Learning Applications:
   - Jacobian determinants in normalizing flows
   - Covariance matrix analysis in multivariate statistics
   - Numerical stability assessment in optimization
   - Feature importance in linear transformations
    """)

if __name__ == "__main__":
    """
    EXECUTION ENTRY POINT
    
    This comprehensive demonstration showcases determinant computation through
    multiple methodologies, providing both theoretical understanding and
    practical implementation guidance for linear algebra applications in
    scientific computing and machine learning.
    """
    demonstrate_determinant_computation()