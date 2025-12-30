"""
Row Echelon Form Computation
============================
Educational implementation of Gaussian elimination to compute Row Echelon Form (REF).

The REF algorithm transforms a matrix into upper triangular form using elementary
row operations. This is fundamental for:
- Determining matrix rank
- Solving linear systems
- Understanding linear independence
- Matrix decomposition techniques

Mathematical Foundation:
A matrix is in Row Echelon Form when:
1. All zero rows are at the bottom
2. The leading entry (pivot) of each nonzero row is 1
3. Each pivot is to the right of pivots above it
4. Entries below pivots are zero
"""

import numpy as np
from typing import Optional, Tuple


def compute_row_echelon_form(
    matrix: np.ndarray, 
    epsilon: float = 1e-10,
    track_operations: bool = False
) -> Tuple[np.ndarray, Optional[list]]:
    """
    Transform matrix to Row Echelon Form using Gaussian elimination.
    
    Parameters
    ----------
    matrix : np.ndarray
        Input matrix of shape (m, n)
    epsilon : float, optional
        Numerical tolerance for pivot detection
    track_operations : bool, optional
        If True, return list of elementary row operations performed
        
    Returns
    -------
    Tuple[np.ndarray, Optional[list]]
        REF matrix and list of operations if tracking is enabled
        
    Examples
    --------
    >>> A = np.array([[2, 4, -2], [4, 9, -3], [-2, -3, 7]], dtype=float)
    >>> ref_A, ops = compute_row_echelon_form(A, track_operations=True)
    >>> print(f"Rank: {np.sum(np.any(np.abs(ref_A) > 1e-10, axis=1))}")
    """
    # Create a copy to avoid modifying the original
    A = matrix.astype(np.float64).copy()
    m, n = A.shape
    
    if track_operations:
        operations = []
        operations.append({
            'type': 'initial',
            'matrix': A.copy(),
            'description': 'Starting matrix'
        })
    
    print("=" * 70)
    print("ROW ECHELON FORM COMPUTATION")
    print(f"Matrix shape: {m} × {n}")
    print("=" * 70)
    print(f"\nOriginal matrix:\n{A}\n")
    
    pivot_row = 0  # Current pivot row index
    
    # Iterate through columns (potential pivot positions)
    for col in range(min(m, n)):  # Only need to process min(m, n) columns
        print(f"\n{'─' * 60}")
        print(f"PROCESSING COLUMN {col}")
        print(f"Current pivot row: {pivot_row}")
        print(f"Pivot column entries: {A[pivot_row:, col]}")
        
        # Step 1: Find pivot - first nonzero entry below/at current pivot row
        pivot_index = _find_pivot(A, col, pivot_row, epsilon)
        
        if pivot_index is None:
            print(f"  → Column {col} has no pivot (linearly dependent)")
            continue
        
        # Step 2: Swap rows if necessary
        if pivot_index != pivot_row:
            print(f"  ↳ Swapping rows {pivot_row} ↔ {pivot_index}")
            A[[pivot_row, pivot_index]] = A[[pivot_index, pivot_row]]
            
            if track_operations:
                operations.append({
                    'type': 'swap',
                    'rows': (pivot_row, pivot_index),
                    'matrix': A.copy(),
                    'description': f'Swap rows {pivot_row} and {pivot_index}'
                })
            
            print(f"  After row swap:\n{A[pivot_row:, :]}")
        
        # Step 3: Normalize pivot row (make pivot = 1)
        pivot_value = A[pivot_row, col]
        print(f"  ↳ Normalizing row {pivot_row} (pivot = {pivot_value:.6f})")
        A[pivot_row] /= pivot_value
        
        if track_operations:
            operations.append({
                'type': 'scale',
                'row': pivot_row,
                'scalar': 1 / pivot_value,
                'matrix': A.copy(),
                'description': f'Scale row {pivot_row} by {1/pivot_value:.4f}'
            })
        
        print(f"  After normalization:\n{A[pivot_row:, :]}")
        
        # Step 4: Eliminate entries below the pivot
        elimination_count = 0
        for r in range(pivot_row + 1, m):
            if abs(A[r, col]) > epsilon:
                factor = A[r, col]
                print(f"  ↳ Eliminating row {r} using row {pivot_row} (factor = {factor:.6f})")
                A[r] -= factor * A[pivot_row]
                elimination_count += 1
                
                if track_operations:
                    operations.append({
                        'type': 'eliminate',
                        'target_row': r,
                        'source_row': pivot_row,
                        'factor': factor,
                        'matrix': A.copy(),
                        'description': f'Row {r} ← Row {r} - {factor:.4f} × Row {pivot_row}'
                    })
        
        if elimination_count > 0:
            print(f"  Eliminated entries in {elimination_count} rows below pivot")
            print(f"  Current state:\n{A[pivot_row:, :]}")
        
        # Move to next pivot row
        pivot_row += 1
        
        # Stop if we've processed all rows
        if pivot_row >= m:
            print(f"\n  → Reached bottom of matrix (row {pivot_row-1})")
            break
    
    print(f"\n{'═' * 70}")
    print("FINAL ROW ECHELON FORM")
    print(f"{'═' * 70}")
    
    # Display final REF with visual indicators
    _display_ref_with_pivots(A, epsilon)
    
    # Compute and display matrix properties
    _analyze_matrix_properties(A, epsilon)
    
    return (A, operations if track_operations else None)


def _find_pivot(
    matrix: np.ndarray, 
    col: int, 
    start_row: int, 
    epsilon: float
) -> Optional[int]:
    """
    Find the first row with a non-zero entry in the specified column.
    
    Uses numerical tolerance to detect effectively zero entries.
    """
    for row in range(start_row, matrix.shape[0]):
        if abs(matrix[row, col]) > epsilon:
            return row
    return None


def _display_ref_with_pivots(matrix: np.ndarray, epsilon: float):
    """
    Display the REF matrix with visual indicators for pivots.
    """
    m, n = matrix.shape
    
    print("\nRow Echelon Form (REF):")
    for i in range(m):
        row_str = "["
        for j in range(n):
            value = matrix[i, j]
            
            # Check if this is a pivot position
            is_pivot = False
            if abs(value - 1.0) < epsilon:
                # Check if this is the first non-zero in the row
                row_nonzeros = np.where(np.abs(matrix[i, :]) > epsilon)[0]
                if len(row_nonzeros) > 0 and j == row_nonzeros[0]:
                    is_pivot = True
            
            # Format the value
            if abs(value) < epsilon:
                formatted = " 0.0000 "
            elif is_pivot:
                formatted = f" \033[1m{value:7.4f}\033[0m "  # Bold for pivot
            else:
                formatted = f" {value:7.4f} "
            
            row_str += formatted
        row_str += "]"
        print(row_str)
    
    # Highlight pivot columns
    pivot_cols = []
    for i in range(m):
        row_nonzeros = np.where(np.abs(matrix[i, :]) > epsilon)[0]
        if len(row_nonzeros) > 0:
            pivot_cols.append(row_nonzeros[0])
    
    if pivot_cols:
        pivot_marker = " " * 4  # Initial offset
        for j in range(n):
            if j in pivot_cols:
                pivot_marker += "   ↑    "
            else:
                pivot_marker += "        "
        print("\033[90m" + pivot_marker + "← Pivot columns\033[0m")


def _analyze_matrix_properties(matrix: np.ndarray, epsilon: float):
    """
    Analyze and display key properties of the REF matrix.
    """
    m, n = matrix.shape
    
    # Count non-zero rows (rank)
    non_zero_rows = 0
    for i in range(m):
        if np.any(np.abs(matrix[i, :]) > epsilon):
            non_zero_rows += 1
    
    rank = non_zero_rows
    nullity = n - rank
    
    print(f"\n{'─' * 40}")
    print("MATRIX PROPERTIES")
    print(f"{'─' * 40}")
    print(f"• Rank:               {rank}")
    print(f"• Nullity:            {nullity}")
    print(f"• Dimension:          {m} × {n}")
    print(f"• Full rank:          {rank == min(m, n)}")
    
    if rank < min(m, n):
        print(f"• Linear dependencies: {min(m, n) - rank}")
    
    # Identify pivot columns
    pivot_columns = []
    for i in range(m):
        row_nonzeros = np.where(np.abs(matrix[i, :]) > epsilon)[0]
        if len(row_nonzeros) > 0:
            pivot_columns.append(row_nonzeros[0])
    
    print(f"• Pivot columns:      {sorted(pivot_columns)}")
    
    # Special cases
    if rank == 0:
        print("  → Zero matrix")
    elif rank == m == n:
        print("  → Invertible (full rank square matrix)")
    elif rank < n:
        print("  → Columns are linearly dependent")
    elif rank < m:
        print("  → Rows are linearly dependent")


def verify_ref_properties(matrix: np.ndarray, epsilon: float = 1e-10) -> dict:
    """
    Verify that a matrix satisfies Row Echelon Form properties.
    
    Returns a dictionary with verification results and explanations.
    """
    m, n = matrix.shape
    violations = []
    
    # Property 1: Zero rows at bottom
    found_zero_row = False
    for i in range(m):
        if np.all(np.abs(matrix[i, :]) < epsilon):
            found_zero_row = True
        elif found_zero_row:  # Non-zero row after zero row
            violations.append(f"Non-zero row {i} after zero row")
            break
    
    # Property 2: Leading entry of each non-zero row is 1
    for i in range(m):
        if np.any(np.abs(matrix[i, :]) > epsilon):
            # Find first non-zero entry
            for j in range(n):
                if abs(matrix[i, j]) > epsilon:
                    if abs(matrix[i, j] - 1.0) > epsilon:
                        violations.append(f"Row {i} leading entry is {matrix[i, j]:.4f}, not 1")
                    break
    
    # Property 3: Each leading 1 is to the right of leading 1s above
    last_pivot_col = -1
    for i in range(m):
        if np.any(np.abs(matrix[i, :]) > epsilon):
            for j in range(n):
                if abs(matrix[i, j]) > epsilon:
                    if j <= last_pivot_col:
                        violations.append(f"Row {i} pivot at column {j}, not right of previous at {last_pivot_col}")
                    last_pivot_col = j
                    break
    
    return {
        'is_valid_ref': len(violations) == 0,
        'violations': violations,
        'rank': np.sum([np.any(np.abs(matrix[i, :]) > epsilon) for i in range(m)])
    }


# ============================================================================
# DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    # Example 1: Full rank matrix
    print("\n" + "="*70)
    print("EXAMPLE 1: FULL RANK 3×3 MATRIX")
    print("="*70)
    
    A1 = np.array([
        [2, 4, -2],
        [4, 9, -3],
        [-2, -3, 7]
    ], dtype=float)
    
    ref_A1, operations1 = compute_row_echelon_form(A1, track_operations=True)
    verification1 = verify_ref_properties(ref_A1)
    
    print(f"\nREF Verification: {'✓ PASS' if verification1['is_valid_ref'] else '✗ FAIL'}")
    if verification1['violations']:
        for violation in verification1['violations']:
            print(f"  - {violation}")
    
    # Example 2: Rank-deficient matrix
    print("\n\n" + "="*70)
    print("EXAMPLE 2: RANK-DEFICIENT 4×3 MATRIX")
    print("="*70)
    
    A2 = np.array([
        [1, 2, 3],
        [2, 4, 6],  # Linearly dependent (2 × row 1)
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=float)
    
    ref_A2, _ = compute_row_echelon_form(A2, track_operations=False)
    
    # Example 3: Educational comparison with numpy.linalg.matrix_rank
    print("\n" + "="*70)
    print("EDUCATIONAL INSIGHTS")
    print("="*70)
    
    print("\nComparison with NumPy's SVD-based rank computation:")
    print(f"• Our REF rank:      {verification1['rank']}")
    print(f"• NumPy matrix_rank: {np.linalg.matrix_rank(A1)}")
    
    print("\nKey Learnings:")
    print("1. REF reveals the matrix's rank and linear dependencies")
    print("2. Pivot columns correspond to linearly independent columns")
    print("3. Computational complexity: O(m × n × min(m, n))")
    print("4. Numerical stability requires careful pivot selection")
    print("5. REF is not unique (unlike Reduced Row Echelon Form)")
    
    print("\nApplications in Machine Learning:")
    print("• Feature selection (identifying redundant features)")
    print("• Dimensionality reduction (finding intrinsic dimension)")
    print("• Understanding model capacity and overfitting")
    print("• Solving normal equations in linear regression")