import numpy as np


v1 = np.arange(10)

print("Vector v1 (using np.arange):")
print(v1)
print(f"Type of v1: {v1.dtype}")
print(f"Shape of v1: {v1.shape}")
print(f"Number of elements in v1: {len(v1)}")
print()


v2 = np.linspace(0.0, 1.0, 5)


print("Vector v2 (using np.linspace):")
print(v2)
print(f"Type of v2: {v2.dtype}")
print(f"Shape of v2: {v2.shape}")
print(f"Spacing between values: {(v2[1] - v2[0]):.2f}")
print()

print("=" * 50)
print("VERIFICATION:")
print("=" * 50)


print("Verifying v1:")
print(f"First element: {v1[0]} (should be 0)")
print(f"Last element: {v1[-1]} (should be 9)")
print(f"Is v1[0] == 0? {v1[0] == 0}")
print(f"Is v1[-1] == 9? {v1[-1] == 9}")
print()


print("Verifying v2:")
print(f"First element: {v2[0]:.1f} (should be 0.0)")
print(f"Last element: {v2[-1]:.1f} (should be 1.0)")
print(f"Number of elements: {len(v2)} (should be 5)")

spacing = v2[1] - v2[0]
print(f"Actual spacing between elements: {spacing:.2f}")
print(f"Expected spacing: {(1.0 - 0.0) / (5 - 1):.2f}")
print()




print("=" * 50)
print("ADDITIONAL EXAMPLES:")
print("=" * 50)

# Example 1: Different ways to create the same vector
print("\nExample 1: Creating [2, 4, 6, 8]")
print("Using arange: ", np.arange(2, 10, 2))  # Start=2, Stop=10 (exclusive), Step=2
print("Using linspace:", np.linspace(2, 8, 4))  # Start=2, Stop=8, 4 points

# Example 2: Watch out for floating point issues with arange
print("\nExample 2: Floating point arange (can be tricky!)")
print("arange(0, 1, 0.1): ", np.arange(0, 1, 0.1))
print("linspace(0, 1, 10): ", np.linspace(0, 1, 10))

# Example 3: Specifying data types
print("\nExample 3: Controlling data types")
print("arange with float: ", np.arange(0, 5, dtype=float))
print("linspace with int: ", np.linspace(0, 4, 5, dtype=int))  # Converts to integers

# =============================================================================
# PART 6: Common Use Cases
# =============================================================================
"""
Real-world applications of regularly spaced vectors:
1. Time series analysis: Creating time points
2. Plotting: Creating x-axis values
3. Signal processing: Sampling signals
4. Physics simulations: Discretizing space or time
5. Machine learning: Creating test/train splits
"""

print("\n" + "=" * 50)
print("SUMMARY:")
print("=" * 50)
print("1. np.arange(start, stop, step):")
print("   - Creates values from start (inclusive) to stop (exclusive)")
print("   - You control the step size")
print("   - Best for integer sequences or when you know the spacing")
print()
print("2. np.linspace(start, stop, num):")
print("   - Creates 'num' values from start to stop (both inclusive)")
print("   - You control the number of points")
print("   - Best for sampling or when you need a specific number of points")
print()
print("3. Key takeaway:")
print("   Use arange when you know the spacing between values.")
print("   Use linspace when you know how many values you need.")

# =============================================================================
# PART 7: Exercise (Try it yourself!)
# =============================================================================
"""
Try these exercises to test your understanding:

Exercise 1: Create a vector of even numbers from 2 to 20 (inclusive)
Exercise 2: Create a vector of 7 numbers between -1 and 1 (inclusive)
Exercise 3: What's the difference between:
    np.arange(0, 5, 0.5) and np.linspace(0, 4.5, 10)?
"""

print("\n" + "=" * 50)
print("TRY THESE EXERCISES:")
print("=" * 50)
print("Exercise 1: Create a vector of even numbers from 2 to 20 (inclusive)")
print("    Hint: Use np.arange with appropriate parameters")
print("    Expected: [ 2,  4,  6,  8, 10, 12, 14, 16, 18, 20]")
print()
print("Exercise 2: Create a vector of 7 numbers between -1 and 1 (inclusive)")
print("    Hint: Use np.linspace")
print("    Expected: [-1. , -0.66666667, -0.33333333,  0. ,  0.33333333,  0.66666667,  1. ]")
print()
print("Exercise 3: Compare np.arange(0, 5, 0.5) and np.linspace(0, 4.5, 10)")
print("    Are they the same? Try it and see!")