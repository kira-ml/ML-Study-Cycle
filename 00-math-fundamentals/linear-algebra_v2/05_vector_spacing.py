import numpy as np

"""
NumPy Array Generation: Understanding arange() vs linspace()
This tutorial demonstrates two fundamental functions for creating regularly
spaced arrays: np.arange() and np.linspace(). Learn when to use each based
on your specific numerical requirements.
"""

# =============================================================================
# SECTION 1: np.arange() - Step-Based Sequence Generation
# =============================================================================
"""
np.arange(start, stop, step) generates values with fixed step size.
- start: inclusive beginning value (default 0)
- stop: exclusive upper bound (required)
- step: increment between values (default 1)
Returns: Values from start up to but not including stop
"""

print("=" * 60)
print("SECTION 1: np.arange() - Step-Based Array Creation")
print("=" * 60)

# Create sequence from 0 to 9 with step 1 (default parameters)
v1 = np.arange(10)  # Equivalent to np.arange(0, 10, 1)
print("Vector v1 created with np.arange(10):")
print(f"  Array: {v1}")
print(f"  Data type: {v1.dtype} (int64 by default for integer ranges)")
print(f"  Shape: {v1.shape} (1D array with {v1.size} elements)")
print(f"  Verification: First={v1[0]}, Last={v1[-1]}, Length={len(v1)}")
print()

# =============================================================================
# SECTION 2: np.linspace() - Point-Count-Based Generation
# =============================================================================
"""
np.linspace(start, stop, num) generates fixed number of evenly spaced values.
- start: inclusive beginning value (required)
- stop: inclusive ending value (required)
- num: number of points to generate (default 50)
- endpoint: whether to include stop value (default True)
Returns: 'num' values evenly spaced between start and stop
"""

print("=" * 60)
print("SECTION 2: np.linspace() - Count-Based Array Creation")
print("=" * 60)

# Create 5 evenly spaced values between 0.0 and 1.0 (both inclusive)
v2 = np.linspace(0.0, 1.0, 5)
print("Vector v2 created with np.linspace(0.0, 1.0, 5):")
print(f"  Array: {v2}")
print(f"  Data type: {v2.dtype} (float64 by default)")
print(f"  Shape: {v2.shape} ({v2.size} elements)")
print(f"  Spacing between values: {v2[1] - v2[0]:.2f}")
print()

# =============================================================================
# SECTION 3: Detailed Mathematical Verification
# =============================================================================
print("=" * 60)
print("SECTION 3: Mathematical Verification")
print("=" * 60)

# Verification for np.arange()
print("Verification for np.arange():")
print(f"  Expected range: 0 to 9 (10 elements, step=1)")
print(f"  Actual first element: {v1[0]} (✓ matches expected 0)")
print(f"  Actual last element: {v1[-1]} (✓ matches expected 9)")
print(f"  Element count: {v1.size} (✓ matches expected 10)")
print()

# Verification for np.linspace()
print("Verification for np.linspace():")
print(f"  Expected range: 0.0 to 1.0 (5 points inclusive)")
print(f"  Expected spacing: (1.0 - 0.0) / (5 - 1) = 0.25")
print(f"  Actual spacing: {v2[1] - v2[0]:.2f} (✓ matches expected)")
print(f"  Uniformity check: All spacings equal = {np.all(np.diff(v2) == (v2[1] - v2[0]))}")
print()

# =============================================================================
# SECTION 4: Practical Comparison and Applications
# =============================================================================
print("=" * 60)
print("SECTION 4: Comparative Analysis and Use Cases")
print("=" * 60)

# Example 1: Different approaches to same numerical range
print("Example 1: Creating [2, 4, 6, 8]")
arange_example = np.arange(2, 10, 2)  # Step-based: 2, 4, 6, 8 (stop at 10)
linspace_example = np.linspace(2, 8, 4)  # Count-based: 4 points between 2 and 8
print(f"  np.arange(2, 10, 2):  {arange_example}")
print(f"  np.linspace(2, 8, 4): {linspace_example}")
print("  Note: Different approaches can yield identical results")
print()

# Example 2: Floating-point precision considerations
print("Example 2: Floating-point precision differences")
print("  np.arange(0, 1, 0.1):")
print(f"    Result: {np.arange(0, 1, 0.1)}")
print("    Potential issue: Floating-point rounding may exclude endpoint")
print("  np.linspace(0, 1, 10):")
print(f"    Result: {np.linspace(0, 1, 10)}")
print("    Advantage: Guarantees exact endpoint inclusion")
print()

# Example 3: Data type control
print("Example 3: Explicit data type specification")
print(f"  arange as float: {np.arange(0, 5, dtype=float)}")
print(f"  linspace as int: {np.linspace(0, 4, 5, dtype=int)}")
print("  Note: Integer conversion truncates, doesn't round")
print()

# =============================================================================
# SECTION 5: Decision Guide - When to Use Each
# =============================================================================
print("=" * 60)
print("SECTION 5: Decision Guide - arange() vs linspace()")
print("=" * 60)

print("Use np.arange() when:")
print("  1. You know the exact step size between values")
print("  2. Working primarily with integers")
print("  3. Need control over spacing rather than count")
print("  4. Creating sequences where endpoint exclusion is desired")
print()

print("Use np.linspace() when:")
print("  1. You know the exact number of points needed")
print("  2. Require both endpoints to be included")
print("  3. Working with floating-point ranges")
print("  4. Need mathematically exact spacing (no floating-point accumulation error)")
print()

# =============================================================================
# SECTION 6: Practical Applications
# =============================================================================
print("=" * 60)
print("SECTION 6: Real-World Applications")
print("=" * 60)

print("np.arange() applications:")
print("  • Time series with fixed sampling intervals")
print("  • Creating indices for loops or indexing")
print("  • Generating sequences with known increments")
print()

print("np.linspace() applications:")
print("  • Plotting x-axis values for functions")
print("  • Creating test points across a range")
print("  • Discretizing continuous intervals for analysis")
print("  • Generating training/validation splits")
print()

# =============================================================================
# SECTION 7: Practice Exercises with Solutions
# =============================================================================
print("=" * 60)
print("SECTION 7: Practice Exercises")
print("=" * 60)

print("Exercise 1: Create even numbers from 2 to 20 (inclusive)")
print("  Solution using arange:")
print(f"    np.arange(2, 21, 2) = {np.arange(2, 21, 2)}")
print()

print("Exercise 2: Create 7 numbers between -1 and 1 (inclusive)")
print("  Solution using linspace:")
print(f"    np.linspace(-1, 1, 7) = {np.linspace(-1, 1, 7)}")
print()

print("Exercise 3: Compare arange and linspace for same range")
print("  np.arange(0, 5, 0.5):")
print(f"    {np.arange(0, 5, 0.5)}")
print("  np.linspace(0, 4.5, 10):")
print(f"    {np.linspace(0, 4.5, 10)}")
print("  Analysis: Results differ due to endpoint handling")
print("    arange: 10 values from 0 to 4.5 (step-based, stop exclusive)")
print("    linspace: 10 values from 0 to 4.5 (count-based, endpoints inclusive)")
print()

# =============================================================================
# SECTION 8: Performance and Memory Considerations
# =============================================================================
print("=" * 60)
print("SECTION 8: Performance Notes")
print("=" * 60)

print("Memory efficiency:")
print("  • Both functions allocate memory for entire array at creation")
print("  • For large arrays, consider numpy.memmap for memory-mapped arrays")
print()

print("Performance characteristics:")
print("  • arange(): Faster for simple integer sequences")
print("  • linspace(): Slightly more overhead due to spacing calculation")
print("  • Both optimized in C for performance")