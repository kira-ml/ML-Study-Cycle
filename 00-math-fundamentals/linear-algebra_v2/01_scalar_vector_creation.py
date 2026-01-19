"""
NumPy Fundamentals: Scalars vs. Vectors
---------------------------------------
Demonstrates the structural and operational distinctions between
scalar values (0-D) and vector arrays (1-D) in numerical computing.
"""

import sys
from typing import Final

import numpy as np
import numpy.typing as npt

# Enforce explicit precision
DTYPE: Final = np.float64

def print_header(title: str) -> None:
    """Utility to print formatted section headers."""
    print(f"\n{title}")
    print("=" * 60)

def demonstrate_scalar(value: float) -> None:
    """
    Analyze and display properties of a scalar value.

    Parameters
    ----------
    value : float
        The scalar input to analyze.
    """
    print_header("PART 1: Scalar Values (0-Dimensional)")
    
    # Structural properties
    print(f"Value          : {value}")
    print(f"Python Type    : {type(value).__name__}")
    print(f"Dimensionality : 0 (Point data)")
    
    # Operation demonstration
    offset = 5.0
    result = value + offset
    print(f"Operation      : {value} + {offset} = {result}")
    print("Constraint     : Cannot be indexed or sliced.")

def demonstrate_vector(vec: npt.NDArray[DTYPE]) -> None:
    """
    Analyze and display properties of a vector array.

    Parameters
    ----------
    vec : npt.NDArray[np.float64]
        The 1-D vector array to analyze.
    """
    print_header("PART 2: Vector Arrays (1-Dimensional)")

    # Structural properties
    print(f"Vector Content : {vec}")
    print(f"NumPy Type     : {type(vec).__name__}")
    print(f"Data Type      : {vec.dtype}")
    print(f"Shape          : {vec.shape} (1-D array with {vec.size} elements)")
    print(f"Dimensionality : {vec.ndim}-D")
    print(f"Memory Usage   : {vec.nbytes} bytes")

    # Access patterns
    print("\n[Access Patterns]")
    print(f"Index [0]      : {vec[0]}")
    print(f"Slice [0:2]    : {vec[0:2]}")

    # Vectorization (Broadcasting)
    # This is the key optimization over scalars: applying ops to all elements at once
    offset = 5.0
    broadcast_result = vec + offset
    
    print("\n[Vectorized Operation - Broadcasting]")
    print(f"Logic          : array + {offset} (Applied to all elements)")
    print(f"Result         : {broadcast_result}")

def main() -> None:
    """Main execution entry point."""
    
    # 1. Define inputs with explicit precision
    # Scalar: Temperature reading
    scalar_temp: float = 98.6
    
    # Vector: Feature set (e.g., [feature_1, feature_2, feature_3, feature_4])
    feature_vector: npt.NDArray[DTYPE] = np.array(
        [2.5, -1.0, 4.7, 0.3], 
        dtype=DTYPE
    )

    # 2. execute demonstrations
    try:
        demonstrate_scalar(scalar_temp)
        demonstrate_vector(feature_vector)
        
        # 3. Summary
        print_header("Summary")
        print("• Scalar : Single value, no shape, minimal overhead.")
        print("• Vector : Ordered collection, enables SIMD/Broadcasting operations.")
        print("• Memory : Vectors store data contiguously for CPU cache optimization.")
        print()
        
    except Exception as e:
        print(f"Critical execution error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()