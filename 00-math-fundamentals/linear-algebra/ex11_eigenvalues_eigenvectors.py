import numpy as np


def compute_trace_and_determinant(matrix: np.ndarray) -> tuple:
    """
    Compute the trace and determinant of a square matrix.

    Args:
        matrix (np.ndarray): A square matrix (e.g., 2x2).

    Returns:
        tuple: (trace, determinant)
    """
    trace = np.trace(matrix)
    determinant = np.linalg.det(matrix)
    return trace, determinant


def solve_characteristic_polynomial(trace: float, determinant: float) -> np.ndarray:
    """
    Solve the characteristic polynomial of a 2x2 matrix: λ² - trace·λ + det = 0

    Args:
        trace (float): Trace of the matrix.
        determinant (float): Determinant of the matrix.

    Returns:
        np.ndarray: The eigenvalues (real or complex).
    """
    coeffs = [1, -trace, determinant]
    return np.roots(coeffs)


def compute_eigenvector(matrix: np.ndarray, eigenvalue: float) -> np.ndarray:
    """
    Compute a normalized eigenvector for a given eigenvalue of a 2x2 matrix.

    Args:
        matrix (np.ndarray): The original square matrix.
        eigenvalue (float): The eigenvalue to compute the eigenvector for.

    Returns:
        np.ndarray: A normalized eigenvector.
    """
    identity = np.eye(matrix.shape[0])
    shifted_matrix = matrix - eigenvalue * identity

    # Use the first row to construct an eigenvector [1, x] (if possible)
    if np.abs(shifted_matrix[0, 1]) > 1e-8:
        x = (eigenvalue - matrix[0, 0]) / matrix[0, 1]
        vec = np.array([1, x])
    else:
        # Fall back to using second row if the first row is degenerate
        x = (eigenvalue - matrix[1, 1]) / matrix[1, 0]
        vec = np.array([x, 1])

    return vec / np.linalg.norm(vec)


def diagonalize_matrix(matrix: np.ndarray) -> tuple:
    """
    Diagonalize a 2x2 matrix by computing its eigenvalues and eigenvectors.

    Args:
        matrix (np.ndarray): The input matrix (must be 2x2 and diagonalizable).

    Returns:
        tuple: (D, P) where D is diagonal matrix of eigenvalues,
               and P contains eigenvectors as columns.
    """
    trace, det = compute_trace_and_determinant(matrix)
    eigenvalues = solve_characteristic_polynomial(trace, det)

    v1 = compute_eigenvector(matrix, eigenvalues[0])
    v2 = compute_eigenvector(matrix, eigenvalues[1])

    D = np.diag(eigenvalues)
    P = np.column_stack((v1, v2))

    return D, P, eigenvalues, [v1, v2]


def reconstruct_matrix(P: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    Reconstruct the original matrix from its eigendecomposition.

    Args:
        P (np.ndarray): Matrix of eigenvectors.
        D (np.ndarray): Diagonal matrix of eigenvalues.

    Returns:
        np.ndarray: Reconstructed matrix.
    """
    return P @ D @ np.linalg.inv(P)


if __name__ == "__main__":
    # Example matrix
    A = np.array([[4, 1], [2, 3]], dtype=float)
    print("Original Matrix A:\n", A)

    # Diagonalization
    D, P, eigenvalues, eigenvectors = diagonalize_matrix(A)

    print("\nEigenvalues:")
    print(eigenvalues)

    print("\nNormalized Eigenvectors:")
    for i, v in enumerate(eigenvectors, 1):
        print(f"v{i} (for λ{i}):", v)

    print("\nDiagonal Matrix D:")
    print(D)

    print("\nMatrix P (eigenvectors as columns):")
    print(P)

    # Reconstruct A using A = P D P⁻¹
    A_reconstructed = reconstruct_matrix(P, D)
    print("\nReconstructed A from eigendecomposition:")
    print(A_reconstructed)
