import numpy as np
import time

def vector_dot(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute the dot product of two vectors manually.
    
    Formula: sum(v1[i] * v2[i]) for i in range(n)
    
    Args:
        v1: First vector (1D array)
        v2: Second vector (1D array)
    
    Returns:
        Dot product as a scalar float
    
    Raises:
        ValueError: If vectors have different lengths
    """
    # Check if vectors have same length
    if len(v1) != len(v2):
        raise ValueError(f"Vectors must have same length: {len(v1)} != {len(v2)}")
    
    # Initialize result
    result = 0.0
    
    # Manual dot product computation
    for i in range(len(v1)):
        result += v1[i] * v2[i]
    
    return result


def vector_outer(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Compute the outer product of two vectors manually.
    
    Formula: result[i, j] = v1[i] * v2[j]
    
    Args:
        v1: First vector (1D array) of length m
        v2: Second vector (1D array) of length n
    
    Returns:
        Matrix of shape (m, n) containing outer product
    """
    # Get dimensions
    m = len(v1)
    n = len(v2)
    
    # Initialize result matrix with zeros
    result = np.zeros((m, n))
    
    # Manual outer product computation
    for i in range(m):
        for j in range(n):
            result[i, j] = v1[i] * v2[j]
    
    return result


def compare_with_numpy():
    """
    Compare our manual implementations with NumPy's built-in functions.
    """
    print("=" * 60)
    print("COMPARING MANUAL VS NUMPY IMPLEMENTATIONS")
    print("=" * 60)
    
    # Test case 1: Simple vectors
    print("\nTest 1: Simple vectors")
    v1 = np.array([1, 2, 3], dtype=float)
    v2 = np.array([4, 5, 6], dtype=float)
    
    # Manual computation
    manual_dot = vector_dot(v1, v2)
    manual_outer = vector_outer(v1, v2)
    
    # NumPy computation
    numpy_dot = np.dot(v1, v2)
    numpy_outer = np.outer(v1, v2)
    
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(f"\nDot Product:")
    print(f"  Manual: {manual_dot}")
    print(f"  NumPy:  {numpy_dot}")
    print(f"  Match:  {np.isclose(manual_dot, numpy_dot)}")
    
    print(f"\nOuter Product:")
    print(f"  Manual shape: {manual_outer.shape}")
    print(f"  NumPy shape:  {numpy_outer.shape}")
    print(f"  Matrices equal: {np.allclose(manual_outer, numpy_outer)}")
    
    # Test case 2: Random vectors
    print("\n" + "-" * 40)
    print("Test 2: Random vectors (size 5)")
    np.random.seed(42)  # For reproducibility
    v1_rand = np.random.randn(5)
    v2_rand = np.random.randn(5)
    
    manual_dot_rand = vector_dot(v1_rand, v2_rand)
    numpy_dot_rand = np.dot(v1_rand, v2_rand)
    
    print(f"\nDot Product of random vectors:")
    print(f"  Manual: {manual_dot_rand:.6f}")
    print(f"  NumPy:  {numpy_dot_rand:.6f}")
    print(f"  Difference: {abs(manual_dot_rand - numpy_dot_rand):.10f}")
    
    # Test case 3: Performance comparison
    print("\n" + "-" * 40)
    print("Test 3: Performance comparison (size 1000)")
    
    # Create larger vectors
    size = 1000
    large_v1 = np.random.randn(size)
    large_v2 = np.random.randn(size)
    
    # Time manual dot product
    start_time = time.time()
    manual_result = vector_dot(large_v1, large_v2)
    manual_time = time.time() - start_time
    
    # Time NumPy dot product
    start_time = time.time()
    numpy_result = np.dot(large_v1, large_v2)
    numpy_time = time.time() - start_time
    
    print(f"\nDot Product Performance:")
    print(f"  Manual time: {manual_time:.6f} seconds")
    print(f"  NumPy time:  {numpy_time:.6f} seconds")
    print(f"  Speedup factor: {manual_time/numpy_time:.1f}x")
    print(f"  Results match: {np.isclose(manual_result, numpy_result)}")


def explain_linear_algebra():
    """
    Educational explanation of the linear algebra concepts.
    """
    print("\n" + "=" * 60)
    print("LINEAR ALGEBRA CONCEPTS EXPLAINED")
    print("=" * 60)
    
    print("\n1. VECTOR DOT PRODUCT (Inner Product)")
    print("-" * 40)
    print("The dot product combines two vectors into a scalar.")
    print("Geometric interpretation: v1 · v2 = ||v1|| * ||v2|| * cos(θ)")
    print("where θ is the angle between the vectors.")
    print("\nExample: v1 = [1, 2, 3], v2 = [4, 5, 6]")
    print("v1 · v2 = (1×4) + (2×5) + (3×6) = 4 + 10 + 18 = 32")
    
    print("\n2. VECTOR OUTER PRODUCT")
    print("-" * 40)
    print("The outer product combines two vectors into a matrix.")
    print("If v1 has length m and v2 has length n,")
    print("the result is an m × n matrix where:")
    print("  result[i, j] = v1[i] × v2[j]")
    
    print("\nExample: v1 = [1, 2], v2 = [3, 4, 5]")
    v1_example = np.array([1, 2])
    v2_example = np.array([3, 4, 5])
    result_example = vector_outer(v1_example, v2_example)
    print(f"\nv1 = {v1_example}")
    print(f"v2 = {v2_example}")
    print(f"Outer product = \n{result_example}")
    
    print("\n3. TIME COMPLEXITY ANALYSIS")
    print("-" * 40)
    print("Dot Product: O(n) - one loop through n elements")
    print("Outer Product: O(m×n) - nested loops through m and n elements")
    
    print("\n4. APPLICATION IN DEEP LEARNING")
    print("-" * 40)
    print("Dot Product Used For:")
    print("  • Computing weighted sums in neural networks")
    print("  • Attention mechanisms in transformers")
    print("  • Similarity measurements")
    
    print("\nOuter Product Used For:")
    print("  • Computing covariance matrices")
    print("  • Attention score calculations")
    print("  • Feature interaction terms")


def interactive_examples():
    """
    Interactive examples for hands-on learning.
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE EXAMPLES")
    print("=" * 60)
    
    while True:
        print("\nChoose an operation:")
        print("1. Compute dot product")
        print("2. Compute outer product")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '3':
            print("Exiting interactive mode.")
            break
            
        try:
            # Get vector inputs
            v1_input = input("Enter first vector (comma-separated numbers): ")
            v2_input = input("Enter second vector (comma-separated numbers): ")
            
            # Convert to numpy arrays
            v1 = np.array([float(x.strip()) for x in v1_input.split(',')])
            v2 = np.array([float(x.strip()) for x in v2_input.split(',')])
            
            if choice == '1':
                print(f"\nv1 = {v1}")
                print(f"v2 = {v2}")
                print(f"Length v1 = {len(v1)}, Length v2 = {len(v2)}")
                
                try:
                    result = vector_dot(v1, v2)
                    print(f"Manual dot product: {result}")
                    
                    # Compare with NumPy
                    numpy_result = np.dot(v1, v2)
                    print(f"NumPy dot product:  {numpy_result}")
                    print(f"Match: {np.isclose(result, numpy_result)}")
                    
                    # Geometric interpretation
                    norm_v1 = np.linalg.norm(v1)
                    norm_v2 = np.linalg.norm(v2)
                    if norm_v1 > 0 and norm_v2 > 0:
                        cos_theta = result / (norm_v1 * norm_v2)
                        cos_theta = np.clip(cos_theta, -1, 1)  # Numerical safety
                        angle_deg = np.degrees(np.arccos(cos_theta))
                        print(f"Angle between vectors: {angle_deg:.2f}°")
                        
                except ValueError as e:
                    print(f"Error: {e}")
                    
            elif choice == '2':
                print(f"\nv1 = {v1} (length {len(v1)})")
                print(f"v2 = {v2} (length {len(v2)})")
                
                result = vector_outer(v1, v2)
                print(f"\nManual outer product shape: {result.shape}")
                print("Result matrix:")
                print(result)
                
                # Compare with NumPy
                numpy_result = np.outer(v1, v2)
                print(f"\nNumPy outer product matches: {np.allclose(result, numpy_result)}")
                
                # Show the pattern
                print("\nPattern: Each element result[i,j] = v1[i] × v2[j]")
                print("First row: v1[0] × each element of v2")
                print(f"  {v1[0]} × {v2} = {result[0, :]}")
                
        except ValueError:
            print("Invalid input! Please enter numbers separated by commas.")
        except Exception as e:
            print(f"Error: {e}")


# Main execution
if __name__ == "__main__":
    print("LINEAR ALGEBRA IN DEEP LEARNING - EDUCATIONAL IMPLEMENTATION")
    print("=" * 60)
    
    # Run comparisons
    compare_with_numpy()
    
    # Explain concepts
    explain_linear_algebra()
    
    # Run interactive examples
    interactive_examples()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Dot product: Combines two vectors → scalar")
    print("2. Outer product: Combines two vectors → matrix")
    print("3. Manual implementation helps understand the math")
    print("4. NumPy is optimized and much faster")
    print("5. These operations are fundamental to deep learning")
    
    print("\nPractice Exercise:")
    print("Try modifying the code to:")
    print("1. Add vector addition and subtraction")
    print("2. Implement matrix multiplication")
    print("3. Add scalar multiplication")
    print("4. Compute vector norms (length)")