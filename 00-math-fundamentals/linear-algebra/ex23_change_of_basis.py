import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def compute_change_of_basis_matrix(old_basis, new_basis):
    """
    Compute the change-of-basis matrix from old_basis to new_basis.
    
    Parameters:
    old_basis (np.array): Matrix whose columns are the old basis vectors
    new_basis (np.array): Matrix whose columns are the new basis vectors
    
    Returns:
    np.array: Change-of-basis matrix
    """
    return np.linalg.inv(new_basis) @ old_basis

def transform_vector(vector, cob_matrix):
    """
    Transform a vector using the change-of-basis matrix.
    
    Parameters:
    vector (np.array): Vector in the old basis
    cob_matrix (np.array): Change-of-basis matrix
    
    Returns:
    np.array: Vector in the new basis
    """
    return cob_matrix @ vector

def visualize_2d_change_of_basis():
    """Visualize change of basis in 2D."""
    # Define old basis (standard basis)
    old_basis = np.array([
        [1, 0],  # e1
        [0, 1]   # e2
    ]).T
    
    # Define new basis
    new_basis = np.array([
        [2, 1],   # b1'
        [1, 2]    # b2'
    ]).T
    
    # Vector to transform (in old basis)
    v_old = np.array([3, 2])
    
    # Compute change-of-basis matrix
    cob_matrix = compute_change_of_basis_matrix(old_basis, new_basis)
    
    # Transform vector
    v_new = transform_vector(v_old, cob_matrix)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Old basis
    ax1.quiver(0, 0, old_basis[0,0], old_basis[1,0], 
               angles='xy', scale_units='xy', scale=1, color='r', width=0.005,
               label='e1')
    ax1.quiver(0, 0, old_basis[0,1], old_basis[1,1], 
               angles='xy', scale_units='xy', scale=1, color='g', width=0.005,
               label='e2')
    ax1.quiver(0, 0, v_old[0], v_old[1], 
               angles='xy', scale_units='xy', scale=1, color='b', width=0.005,
               label='v')
    ax1.grid(True)
    ax1.set_xlim(-1, 4)
    ax1.set_ylim(-1, 4)
    ax1.set_aspect('equal')
    ax1.legend()
    ax1.set_title('Vector in Old Basis')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    # New basis
    ax2.quiver(0, 0, new_basis[0,0], new_basis[1,0], 
               angles='xy', scale_units='xy', scale=1, color='r', width=0.005,
               label="b1'")
    ax2.quiver(0, 0, new_basis[0,1], new_basis[1,1], 
               angles='xy', scale_units='xy', scale=1, color='g', width=0.005,
               label="b2'")
    ax2.quiver(0, 0, v_new[0], v_new[1], 
               angles='xy', scale_units='xy', scale=1, color='b', width=0.005,
               label="v'")
    ax2.grid(True)
    ax2.set_xlim(-1, 4)
    ax2.set_ylim(-1, 4)
    ax2.set_aspect('equal')
    ax2.legend()
    ax2.set_title('Vector in New Basis')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    plt.tight_layout()
    plt.show()
    
    return cob_matrix, v_new

def visualize_3d_change_of_basis():
    """Visualize change of basis in 3D."""
    # Define old basis (standard basis)
    old_basis = np.eye(3)
    
    # Define new basis (rotated basis)
    new_basis = np.array([
        [1, 1, 0],
        [1, -1, 1],
        [0, 1, 1]
    ]).T
    
    # Vector to transform (in old basis)
    v_old = np.array([2, 1, 1])
    
    # Compute change-of-basis matrix
    cob_matrix = compute_change_of_basis_matrix(old_basis, new_basis)
    
    # Transform vector
    v_new = transform_vector(v_old, cob_matrix)
    
    # Visualization
    fig = plt.figure(figsize=(12, 5))
    
    # Old basis
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.quiver(0, 0, 0, old_basis[0,0], old_basis[1,0], old_basis[2,0], 
               color='r', arrow_length_ratio=0.1, label='e1')
    ax1.quiver(0, 0, 0, old_basis[0,1], old_basis[1,1], old_basis[2,1], 
               color='g', arrow_length_ratio=0.1, label='e2')
    ax1.quiver(0, 0, 0, old_basis[0,2], old_basis[1,2], old_basis[2,2], 
               color='b', arrow_length_ratio=0.1, label='e3')
    ax1.quiver(0, 0, 0, v_old[0], v_old[1], v_old[2], 
               color='k', arrow_length_ratio=0.1, label='v')
    ax1.set_xlim([-1, 3])
    ax1.set_ylim([-1, 3])
    ax1.set_zlim([-1, 3])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    ax1.set_title('Vector in Old Basis')
    
    # New basis
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.quiver(0, 0, 0, new_basis[0,0], new_basis[1,0], new_basis[2,0], 
               color='r', arrow_length_ratio=0.1, label="b1'")
    ax2.quiver(0, 0, 0, new_basis[0,1], new_basis[1,1], new_basis[2,1], 
               color='g', arrow_length_ratio=0.1, label="b2'")
    ax2.quiver(0, 0, 0, new_basis[0,2], new_basis[1,2], new_basis[2,2], 
               color='b', arrow_length_ratio=0.1, label="b3'")
    ax2.quiver(0, 0, 0, v_new[0], v_new[1], v_new[2], 
               color='k', arrow_length_ratio=0.1, label="v'")
    ax2.set_xlim([-1, 3])
    ax2.set_ylim([-1, 3])
    ax2.set_zlim([-1, 3])
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    ax2.set_title('Vector in New Basis')
    
    plt.tight_layout()
    plt.show()
    
    return cob_matrix, v_new

def main():
    print("Change of Basis Transformation")
    print("=" * 40)
    
    # Example 1: 2D transformation
    print("\n2D Example:")
    cob_matrix_2d, v_new_2d = visualize_2d_change_of_basis()
    print(f"Change-of-basis matrix:\n{cob_matrix_2d}")
    print(f"Vector in new basis: {v_new_2d}")
    
    # Example 2: 3D transformation
    print("\n3D Example:")
    cob_matrix_3d, v_new_3d = visualize_3d_change_of_basis()
    print(f"Change-of-basis matrix:\n{cob_matrix_3d}")
    print(f"Vector in new basis: {v_new_3d}")
    
    # Verification
    print("\nVerification:")
    print("To verify, we can transform back using the inverse:")
    v_original = np.linalg.inv(cob_matrix_2d) @ v_new_2d
    print(f"Transformed back: {v_original}")
    print(f"Original vector: {np.array([3, 2])}")
    print(f"Close enough: {np.allclose(v_original, np.array([3, 2]))}")

if __name__ == "__main__":
    main()