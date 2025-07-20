
"""
ex22_orthogonal_projections.py
-----------------------------
In this implementation, I demonstrate two foundational techniques for orthogonal projection in linear algebra:
1. Projecting a vector onto another vector.
2. Projecting a vector onto a subspace using a projection matrix.

These operations are essential in many machine learning and data science workflows, including least squares regression, dimensionality reduction, and feature engineering. The code is structured to be both educational and practical, following conventions used in high-quality ML research and production codebases.
"""

import numpy as np



def projected_vector_onto_vector(v: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Project vector v onto vector u using the standard vector projection formula.

    Parameters
    ----------
    v : np.ndarray
        The vector to be projected.
    u : np.ndarray
        The vector onto which v is projected.

    Returns
    -------
    np.ndarray
        The projection of v onto u.

    Notes
    -----
    This function implements the formula:
        proj_u(v) = (<v, u> / <u, u>) * u
    where <.,.> denotes the dot product.
    This is a common operation in ML pipelines, e.g., for feature selection or geometric interpretation of data.
    """
    # Compute the scalar projection (how much of v lies in the direction of u)
    scalar_projection = np.dot(v, u) / np.dot(u, u)
    # Scale u by the scalar projection to get the actual projected vector
    projection = scalar_projection * u
    return projection



direction_vector = np.array([1, 0])

# Example: Project [3, 4] onto the x-axis ([1, 0])
vector_to_project = np.array([3, 4])
direction_vector = np.array([1, 0])
projected_vector = projected_vector_onto_vector(vector_to_project, direction_vector)
print("Projected vector:", projected_vector)




def compute_projection_matrix(A: np.ndarray) -> np.ndarray:
    """
    Compute the orthogonal projection matrix for projecting onto the column space of A.

    Parameters
    ----------
    A : np.ndarray
        The matrix whose column space defines the subspace for projection.

    Returns
    -------
    np.ndarray
        The projection matrix P.

    Notes
    -----
    The formula used is:
        P = A (A^T A)^{-1} A^T
    This is widely used in least squares problems and subspace projections in ML.
    The resulting matrix P is symmetric and idempotent (P^2 = P).
    """
    A_T = A.T
    projection_matrix = A @ np.linalg.inv(A_T @ A) @ A_T
    return projection_matrix




# Example: Project [3, 4] onto the subspace spanned by columns of basic_matrix
basic_matrix = np.array([
    [1, 1],
    [0, 1]
])
P = compute_projection_matrix(basic_matrix)
vector = np.array([3, 4])
projected_onto_subspace = P @ vector
print("Projected onto subspace:", projected_onto_subspace)