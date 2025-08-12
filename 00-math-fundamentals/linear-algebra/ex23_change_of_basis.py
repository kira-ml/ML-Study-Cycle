"""
Change of Basis Visualization Toolkit
=====================================

This module provides tools for visualizing and computing change-of-basis transformations
in both 2D and 3D vector spaces. The implementation demonstrates fundamental concepts
in linear algebra through interactive visualizations and numerical computations.

The change-of-basis transformation is a core concept in linear algebra that allows us
to express vectors and linear transformations in different coordinate systems. This is
particularly important in machine learning for feature transformations, dimensionality
reduction, and understanding how algorithms behave under different representations.

Key Features:
- Compute change-of-basis matrices between arbitrary bases
- Transform vectors between coordinate systems
- Visualize transformations in 2D and 3D space
- Verify transformation correctness through round-trip validation

Example Usage:
    >>> cob_matrix, transformed_vector = visualize_2d_change_of_basis()
    >>> print(f"Transformation matrix: {cob_matrix}")

Note:
    This implementation uses NumPy for numerical computations and Matplotlib for
    visualization. The 3D visualization extends Matplotlib's FancyArrowPatch to
    properly render 3D arrows in projected space.

Author: kira-ml (machine learning student)

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    """
    Custom 3D arrow class for rendering directional vectors in 3D matplotlib plots.
    
    This class extends FancyArrowPatch to handle 3D coordinate transformations
    required for proper projection onto 2D display surfaces. It's essential for
    accurately visualizing vector directions in three-dimensional space.
    
    The implementation follows matplotlib's 3D projection conventions, ensuring
    that arrows maintain proper perspective and alignment with the 3D coordinate system.
    """
    
    def __init__(self, xs, ys, zs, *args, **kwargs):
        """
        Initialize a 3D arrow with start and end coordinates.
        
        Parameters
        ----------
        xs : array-like
            X-coordinates of arrow start and end points [x_start, x_end]
        ys : array-like
            Y-coordinates of arrow start and end points [y_start, y_end]
        zs : array-like
            Z-coordinates of arrow start and end points [z_start, z_end]
        *args : tuple
            Additional positional arguments passed to FancyArrowPatch
        **kwargs : dict
            Additional keyword arguments passed to FancyArrowPatch
        """
        # Initialize parent class with dummy 2D coordinates
        # These will be replaced during the draw operation
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        """
        Render the 3D arrow by projecting 3D coordinates to 2D display space.
        
        This method performs the critical transformation from 3D world coordinates
        to 2D screen coordinates using matplotlib's projection matrix. This ensures
        that the arrow appears correctly oriented and scaled in the 3D visualization.
        
        Parameters
        ----------
        renderer : matplotlib renderer
            The rendering context used for coordinate transformation
        """
        # Extract 3D vertex coordinates
        xs3d, ys3d, zs3d = self._verts3d
        
        # Project 3D coordinates to 2D display coordinates
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        
        # Update arrow positions with projected 2D coordinates
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        
        # Render the arrow using the parent class method
        FancyArrowPatch.draw(self, renderer)


def compute_change_of_basis_matrix(old_basis, new_basis):
    """
    Compute the change-of-basis matrix from old_basis to new_basis.
    
    The change-of-basis matrix transforms coordinates from one basis to another.
    Given a vector expressed in the old basis, multiplying by this matrix gives
    the same vector expressed in the new basis.
    
    Mathematical Foundation:
        If B_old and B_new are matrices whose columns are basis vectors,
        then the change-of-basis matrix C is: C = B_new^(-1) @ B_old
        
        This follows from the relationship: [v]_new = C @ [v]_old
    
    Parameters
    ----------
    old_basis : numpy.ndarray
        Matrix whose columns are the old basis vectors (d x d)
    new_basis : numpy.ndarray
        Matrix whose columns are the new basis vectors (d x d)
        
    Returns
    -------
    numpy.ndarray
        Change-of-basis matrix (d x d)
        
    Raises
    ------
    numpy.linalg.LinAlgError
        If new_basis is singular (non-invertible)
        
    Examples
    --------
    >>> old_basis = np.eye(2)  # Standard basis
    >>> new_basis = np.array([[2, 1], [1, 2]]).T
    >>> cob_matrix = compute_change_of_basis_matrix(old_basis, new_basis)
    >>> print(cob_matrix.shape)
    (2, 2)
    """
    return np.linalg.inv(new_basis) @ old_basis


def transform_vector(vector, cob_matrix):
    """
    Transform a vector using the change-of-basis matrix.
    
    This function applies the change-of-basis transformation to convert
    a vector's coordinates from the old basis representation to the new
    basis representation.
    
    The transformation follows the standard linear algebra relationship:
        v_new = C @ v_old
        
    where C is the change-of-basis matrix and v_old is the original vector.
    
    Parameters
    ----------
    vector : numpy.ndarray
        Vector in the old basis (d-dimensional)
    cob_matrix : numpy.ndarray
        Change-of-basis matrix (d x d)
        
    Returns
    -------
    numpy.ndarray
        Vector in the new basis (d-dimensional)
        
    Examples
    --------
    >>> v_old = np.array([3, 2])
    >>> cob_matrix = np.array([[0.667, -0.333], [-0.333, 0.667]])
    >>> v_new = transform_vector(v_old, cob_matrix)
    >>> print(v_new)
    [1.334 0.667]
    """
    return cob_matrix @ vector


def visualize_2d_change_of_basis():
    """
    Visualize change of basis transformation in 2D space.
    
    This function demonstrates the geometric interpretation of change-of-basis
    by showing the same vector represented in two different coordinate systems.
    The visualization helps build intuition about how basis vectors define
    coordinate systems and how vectors transform between them.
    
    The example uses:
    - Old basis: Standard basis {[1,0], [0,1]}
    - New basis: Custom basis {[2,1], [1,2]}
    - Test vector: [3, 2] in standard coordinates
    
    Returns
    -------
    tuple
        (change_of_basis_matrix, transformed_vector) - The transformation
        matrix and the vector expressed in the new basis
        
    Notes
    -----
    The visualization uses matplotlib's quiver plots to represent vectors
    as arrows. Different colors distinguish basis vectors (red, green) from
    the test vector (blue). The side-by-side comparison clearly shows how
    the same geometric object appears different in different coordinate systems.
    """
    # Define old basis (standard basis) - identity matrix
    # Each column represents a basis vector: e1=[1,0], e2=[0,1]
    old_basis = np.array([
        [1, 0],  # e1 - unit vector along x-axis
        [0, 1]   # e2 - unit vector along y-axis
    ]).T
    
    # Define new basis - linearly independent vectors that span R²
    # These vectors define a new coordinate system where:
    # b1' = [2,1] and b2' = [1,2]
    new_basis = np.array([
        [2, 1],   # b1' - first new basis vector
        [1, 2]    # b2' - second new basis vector
    ]).T
    
    # Vector to transform - expressed in old (standard) basis
    # This represents the point (3, 2) in standard coordinates
    v_old = np.array([3, 2])
    
    # Compute the change-of-basis matrix from old to new basis
    # This matrix will convert coordinates from standard basis to new basis
    cob_matrix = compute_change_of_basis_matrix(old_basis, new_basis)
    
    # Transform the vector to the new coordinate system
    # The resulting vector has the same geometric meaning but different coordinates
    v_new = transform_vector(v_old, cob_matrix)
    
    # Create side-by-side visualization of both coordinate systems
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Vector in old (standard) basis
    # Red arrow: e1 basis vector (x-axis)
    ax1.quiver(0, 0, old_basis[0, 0], old_basis[1, 0], 
               angles='xy', scale_units='xy', scale=1, color='r', width=0.005,
               label='e1')
    # Green arrow: e2 basis vector (y-axis)
    ax1.quiver(0, 0, old_basis[0, 1], old_basis[1, 1], 
               angles='xy', scale_units='xy', scale=1, color='g', width=0.005,
               label='e2')
    # Blue arrow: Test vector v in standard coordinates
    ax1.quiver(0, 0, v_old[0], v_old[1], 
               angles='xy', scale_units='xy', scale=1, color='b', width=0.005,
               label='v')
    
    # Configure first subplot appearance
    ax1.grid(True)
    ax1.set_xlim(-1, 4)
    ax1.set_ylim(-1, 4)
    ax1.set_aspect('equal')
    ax1.legend()
    ax1.set_title('Vector in Old Basis')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    # Plot 2: Vector in new basis
    # Red arrow: b1' new basis vector
    ax2.quiver(0, 0, new_basis[0, 0], new_basis[1, 0], 
               angles='xy', scale_units='xy', scale=1, color='r', width=0.005,
               label="b1'")
    # Green arrow: b2' new basis vector
    ax2.quiver(0, 0, new_basis[0, 1], new_basis[1, 1], 
               angles='xy', scale_units='xy', scale=1, color='g', width=0.005,
               label="b2'")
    # Blue arrow: Test vector v in new coordinates
    ax2.quiver(0, 0, v_new[0], v_new[1], 
               angles='xy', scale_units='xy', scale=1, color='b', width=0.005,
               label="v'")
    
    # Configure second subplot appearance
    ax2.grid(True)
    ax2.set_xlim(-1, 4)
    ax2.set_ylim(-1, 4)
    ax2.set_aspect('equal')
    ax2.legend()
    ax2.set_title('Vector in New Basis')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    # Optimize layout and display
    plt.tight_layout()
    plt.show()
    
    return cob_matrix, v_new


def visualize_3d_change_of_basis():
    """
    Visualize change of basis transformation in 3D space.
    
    This function extends the 2D concept to three dimensions, demonstrating
    how vectors and coordinate systems behave in higher-dimensional spaces.
    The visualization uses custom 3D arrow rendering to properly project
    three-dimensional vectors onto the 2D display surface.
    
    The example uses:
    - Old basis: Standard basis {[1,0,0], [0,1,0], [0,0,1]}
    - New basis: Rotated basis vectors that maintain linear independence
    - Test vector: [2, 1, 1] in standard coordinates
    
    Returns
    -------
    tuple
        (change_of_basis_matrix, transformed_vector) - The transformation
        matrix and the vector expressed in the new basis
        
    Notes
    -----
    The 3D visualization requires special handling of arrow rendering due to
    matplotlib's 2D nature. The Arrow3D class handles the projection from
    3D world coordinates to 2D screen coordinates, ensuring accurate visual
    representation of spatial relationships.
    """
    # Define old basis (standard basis) - 3x3 identity matrix
    # Basis vectors: e1=[1,0,0], e2=[0,1,0], e3=[0,0,1]
    old_basis = np.eye(3)
    
    # Define new basis - linearly independent 3D vectors
    # These vectors span R³ and define a rotated coordinate system
    new_basis = np.array([
        [1, 1, 0],   # b1' - first new basis vector
        [1, -1, 1],  # b2' - second new basis vector
        [0, 1, 1]    # b3' - third new basis vector
    ]).T
    
    # Vector to transform - expressed in old (standard) basis
    # Represents the point (2, 1, 1) in standard 3D coordinates
    v_old = np.array([2, 1, 1])
    
    # Compute the change-of-basis matrix from old to new basis
    cob_matrix = compute_change_of_basis_matrix(old_basis, new_basis)
    
    # Transform the vector to the new coordinate system
    v_new = transform_vector(v_old, cob_matrix)
    
    # Create side-by-side 3D visualization
    fig = plt.figure(figsize=(12, 5))
    
    # Plot 1: Vector in old (standard) basis
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Red arrow: e1 basis vector (x-axis)
    ax1.quiver(0, 0, 0, old_basis[0, 0], old_basis[1, 0], old_basis[2, 0], 
               color='r', arrow_length_ratio=0.1, label='e1')
    # Green arrow: e2 basis vector (y-axis)
    ax1.quiver(0, 0, 0, old_basis[0, 1], old_basis[1, 1], old_basis[2, 1], 
               color='g', arrow_length_ratio=0.1, label='e2')
    # Blue arrow: e3 basis vector (z-axis)
    ax1.quiver(0, 0, 0, old_basis[0, 2], old_basis[1, 2], old_basis[2, 2], 
               color='b', arrow_length_ratio=0.1, label='e3')
    # Black arrow: Test vector v in standard coordinates
    ax1.quiver(0, 0, 0, v_old[0], v_old[1], v_old[2], 
               color='k', arrow_length_ratio=0.1, label='v')
    
    # Configure first 3D subplot
    ax1.set_xlim([-1, 3])
    ax1.set_ylim([-1, 3])
    ax1.set_zlim([-1, 3])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    ax1.set_title('Vector in Old Basis')
    
    # Plot 2: Vector in new basis
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Red arrow: b1' new basis vector
    ax2.quiver(0, 0, 0, new_basis[0, 0], new_basis[1, 0], new_basis[2, 0], 
               color='r', arrow_length_ratio=0.1, label="b1'")
    # Green arrow: b2' new basis vector
    ax2.quiver(0, 0, 0, new_basis[0, 1], new_basis[1, 1], new_basis[2, 1], 
               color='g', arrow_length_ratio=0.1, label="b2'")
    # Blue arrow: b3' new basis vector
    ax2.quiver(0, 0, 0, new_basis[0, 2], new_basis[1, 2], new_basis[2, 2], 
               color='b', arrow_length_ratio=0.1, label="b3'")
    # Black arrow: Test vector v in new coordinates
    ax2.quiver(0, 0, 0, v_new[0], v_new[1], v_new[2], 
               color='k', arrow_length_ratio=0.1, label="v'")
    
    # Configure second 3D subplot
    ax2.set_xlim([-1, 3])
    ax2.set_ylim([-1, 3])
    ax2.set_zlim([-1, 3])
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    ax2.set_title('Vector in New Basis')
    
    # Optimize layout and display
    plt.tight_layout()
    plt.show()
    
    return cob_matrix, v_new


def main():
    """
    Main execution function demonstrating change-of-basis transformations.
    
    This function orchestrates the complete workflow:
    1. Executes 2D change-of-basis visualization and analysis
    2. Executes 3D change-of-basis visualization and analysis
    3. Performs verification of transformation correctness
    
    The verification step is crucial for understanding that change-of-basis
    transformations are invertible - we can transform back to recover the
    original vector, confirming the mathematical correctness of our implementation.
    """
    print("Change of Basis Transformation")
    print("=" * 40)
    
    # Example 1: 2D transformation demonstration
    print("\n2D Example:")
    cob_matrix_2d, v_new_2d = visualize_2d_change_of_basis()
    print(f"Change-of-basis matrix:\n{cob_matrix_2d}")
    print(f"Vector in new basis: {v_new_2d}")
    
    # Example 2: 3D transformation demonstration
    print("\n3D Example:")
    cob_matrix_3d, v_new_3d = visualize_3d_change_of_basis()
    print(f"Change-of-basis matrix:\n{cob_matrix_3d}")
    print(f"Vector in new basis: {v_new_3d}")
    
    # Verification: Demonstrate that transformations are invertible
    print("\nVerification:")
    print("To verify, we can transform back using the inverse:")
    
    # Transform the vector back to the original basis
    v_original = np.linalg.inv(cob_matrix_2d) @ v_new_2d
    print(f"Transformed back: {v_original}")
    print(f"Original vector: {np.array([3, 2])}")
    
    # Check numerical equivalence (accounting for floating-point precision)
    is_close = np.allclose(v_original, np.array([3, 2]))
    print(f"Close enough: {is_close}")


# Execute main function when script is run directly
if __name__ == "__main__":
    main()