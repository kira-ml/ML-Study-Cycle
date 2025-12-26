"""
VECTOR ANGLE COMPUTATION: A NUMERICAL LINEAR ALGEBRA IMPLEMENTATION

This module demonstrates the calculation of the angle between two vectors 
using fundamental linear algebra operations. The implementation follows
mathematical principles that form the basis for:
- Cosine similarity in machine learning
- Orthogonality testing in numerical analysis
- Geometric transformations in computer graphics
"""

import numpy as np
from typing import Tuple, Dict

def compute_vector_angle(
    vector_a: np.ndarray, 
    vector_b: np.ndarray
) -> Dict[str, float]:
    """
    Calculate the angle between two vectors using the dot product method.
    
    Parameters
    ----------
    vector_a : np.ndarray
        First vector in n-dimensional space
    vector_b : np.ndarray
        Second vector in n-dimensional space
    
    Returns
    -------
    Dict[str, float]
        Dictionary containing all computed quantities:
        - dot_product: Scalar projection
        - magnitudes: Euclidean norms
        - cos_theta: Cosine similarity
        - angle_rad: Angle in radians
        - angle_deg: Angle in degrees
    """
    
    # Validate input vectors
    if vector_a.shape != vector_b.shape:
        raise ValueError("Vectors must have identical dimensions")
    
    if not (vector_a.any() and vector_b.any()):
        raise ValueError("Vectors must be non-zero")
    
    # Computation of dot product (inner product)
    # The dot product measures the projection of one vector onto another
    dot_product = np.dot(vector_a, vector_b)
    
    # Calculation of vector magnitudes (Euclidean norms)
    # The norm represents the length of the vector in space
    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)
    
    # Application of Cauchy-Schwarz inequality for validation
    # This fundamental inequality ensures numerical stability
    cauchy_schwarz_bound = magnitude_a * magnitude_b
    if abs(dot_product) > cauchy_schwarz_bound + 1e-10:  # Tolerance for floating-point errors
        raise ArithmeticError("Cauchy-Schwarz inequality violation detected")
    
    # Computation of cosine similarity
    # This ratio represents the cosine of the angle between vectors
    cos_theta = dot_product / cauchy_schwarz_bound
    
    # Numerical stabilization: clip values to valid arccos domain [-1, 1]
    cos_theta_clipped = np.clip(cos_theta, -1.0, 1.0)
    
    # Angle calculation using inverse trigonometric function
    angle_radians = np.arccos(cos_theta_clipped)
    angle_degrees = np.degrees(angle_radians)
    
    return {
        "dot_product": dot_product,
        "magnitude_a": magnitude_a,
        "magnitude_b": magnitude_b,
        "cosine_similarity": cos_theta,
        "angle_radians": angle_radians,
        "angle_degrees": angle_degrees,
        "is_orthogonal": abs(cos_theta) < 1e-10,
        "is_parallel": abs(abs(cos_theta) - 1.0) < 1e-10
    }

def format_vector_analysis(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
    results: Dict[str, float]
) -> str:
    """
    Generate formatted output for vector angle analysis.
    
    Parameters
    ----------
    vector_a : np.ndarray
        First input vector
    vector_b : np.ndarray
        Second input vector
    results : Dict[str, float]
        Computed metrics from compute_vector_angle
    
    Returns
    -------
    str
        Formatted string with analysis results
    """
    
    output = [
        "=" * 60,
        "VECTOR ANGLE ANALYSIS REPORT",
        "=" * 60,
        f"Vector A (dimensions: {vector_a.shape}):",
        f"  {vector_a}",
        f"Vector B (dimensions: {vector_b.shape}):",
        f"  {vector_b}",
        "",
        "COMPUTED METRICS:",
        "-" * 40,
        f"Dot Product (A·B):          {results['dot_product']:>12.6f}",
        f"Magnitude ‖A‖:              {results['magnitude_a']:>12.6f}",
        f"Magnitude ‖B‖:              {results['magnitude_b']:>12.6f}",
        f"Cosine Similarity (cos θ):  {results['cosine_similarity']:>12.6f}",
        f"Angle (radians):            {results['angle_radians']:>12.6f}",
        f"Angle (degrees):            {results['angle_degrees']:>12.6f}",
        "",
        "GEOMETRIC INTERPRETATION:",
        "-" * 40,
    ]
    
    # Add geometric interpretation
    if results['is_orthogonal']:
        output.append("  ⟶ Vectors are orthogonal (θ ≈ 90°)")
    elif results['is_parallel']:
        if results['cosine_similarity'] > 0:
            output.append("  ⟶ Vectors are parallel (θ ≈ 0°)")
        else:
            output.append("  ⟶ Vectors are anti-parallel (θ ≈ 180°)")
    else:
        if results['cosine_similarity'] > 0:
            output.append(f"  ⟶ Vectors form an acute angle (θ < 90°)")
        else:
            output.append(f"  ⟶ Vectors form an obtuse angle (θ > 90°)")
    
    output.append("=" * 60)
    return "\n".join(output)

def analyze_vector_relationship(
    vector_a: np.ndarray,
    vector_b: np.ndarray
) -> None:
    """
    Complete analysis workflow for vector relationship.
    
    This function orchestrates the entire computation and presentation
    of vector angle analysis, demonstrating best practices in numerical
    computing and scientific visualization.
    """
    
    print("\n" + "=" * 60)
    print("INITIATING VECTOR ANGLE ANALYSIS")
    print("=" * 60)
    
    # Display input vectors with metadata
    print(f"\nINPUT SPECIFICATIONS:")
    print(f"  Vector dimensionality: {vector_a.shape[0]}")
    print(f"  Data type: {vector_a.dtype}")
    print(f"  Vector space: ℝ^{vector_a.shape[0]}")
    
    try:
        # Perform core computation
        print(f"\nPERFORMING COMPUTATIONS...")
        results = compute_vector_angle(vector_a, vector_b)
        
        # Display formatted results
        print(format_vector_analysis(vector_a, vector_b, results))
        
        # Additional derived insights
        print("\nDERIVED INSIGHTS:")
        print("-" * 40)
        print(f"  • Euclidean distance between vectors: "
              f"{np.linalg.norm(vector_a - vector_b):.6f}")
        print(f"  • Ratio of magnitudes (‖A‖/‖B‖): "
              f"{results['magnitude_a']/results['magnitude_b']:.6f}")
        print(f"  • Projection of A onto B: "
              f"{results['dot_product']/results['magnitude_b']:.6f}")
        
    except ValueError as e:
        print(f"\nVALIDATION ERROR: {e}")
    except ArithmeticError as e:
        print(f"\nNUMERICAL ERROR: {e}")
    finally:
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)

# Demonstration with example vectors
if __name__ == "__main__":
    """
    EXECUTION EXAMPLE:
    
    This demonstration uses linearly dependent vectors to illustrate
    fundamental geometric relationships. Vector B is exactly 4 times
    Vector A, resulting in parallel vectors with zero angle.
    """
    
    # Define demonstration vectors
    DEMO_VECTOR_A = np.array([2, 4, 6], dtype=np.float64)
    DEMO_VECTOR_B = np.array([8, 10, 12], dtype=np.float64)  # Not a scalar multiple
    
    # Execute comprehensive analysis
    analyze_vector_relationship(DEMO_VECTOR_A, DEMO_VECTOR_B)
    
    # Additional educational examples
    print("\n\n" + "=" * 60)
    print("SUPPLEMENTARY EXAMPLES")
    print("=" * 60)
    
    # Example 1: Orthogonal vectors
    orth_a = np.array([1, 0, 0])
    orth_b = np.array([0, 1, 0])
    print("\n1. ORTHOGONAL VECTORS (90° angle):")
    print(f"   A = {orth_a}, B = {orth_b}")
    print(f"   Computed angle: {np.degrees(np.arccos(np.dot(orth_a, orth_b))):.2f}°")
    
    # Example 2: Parallel vectors
    para_a = np.array([1, 2, 3])
    para_b = np.array([2, 4, 6])  # Scalar multiple
    print("\n2. PARALLEL VECTORS (0° angle):")
    print(f"   A = {para_a}, B = {para_b}")
    print(f"   Computed angle: {np.degrees(np.arccos(np.dot(para_a, para_b)/(np.linalg.norm(para_a)*np.linalg.norm(para_b)))):.2f}°")