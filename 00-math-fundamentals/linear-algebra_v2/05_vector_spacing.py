import numpy as np

"""
NumPy Array Generation: Understanding arange() vs linspace()
Author: Kira-ML
Teaching Philosophy: I believe in learning through comparison and practical intuition.
This tutorial helps you build mental models for when to use each function.
"""

# =============================================================================
# MENTAL MODEL: The Difference in One Sentence
# =============================================================================
"""
Think of np.arange() as "count by steps" and np.linspace() as "divide the distance."
arange: "Start at 0, add 1 each time, stop before 10" → [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
linspace: "Between 0 and 1, give me 5 equally spaced points" → [0.0, 0.25, 0.5, 0.75, 1.0]
"""

print("=" * 60)
print("ARANGE vs LINSPACE: Building Intuition")
print("=" * 60)

# Core Mental Model Demonstration
print("MENTAL MODEL VISUALIZATION:")
print("np.arange(0, 10, 2) means:")
print("  Start at 0, take steps of size 2, stop before 10")
print(f"  Result: {np.arange(0, 10, 2)}")
print()
print("np.linspace(0, 10, 3) means:")
print("  Divide the distance 0→10 into 2 equal segments")
print("  Place 3 points at the boundaries")
print(f"  Result: {np.linspace(0, 10, 3)}")
print()

# =============================================================================
# PART 1: arange() - The "Step Counter"
# =============================================================================
print("=" * 60)
print("PART 1: np.arange() - When You Know Your Step Size")
print("=" * 60)

# I use arange when I want to count in specific increments
print("USE CASE 1: Creating time intervals")
print("Imagine measuring temperature every 30 minutes:")
temp_times = np.arange(0, 180, 30)  # 0 to 180 minutes, every 30 min
print(f"  Measurement times (minutes): {temp_times}")
print(f"  This gives me {len(temp_times)} measurements")
print()

print("USE CASE 2: Looping with indices")
print("When I need to process elements in an array:")
indices = np.arange(5)
print(f"  Array indices: {indices}")
print(f"  I can use these to access: element[0], element[1], ... element[4]")
print()

# Common pitfall I've encountered
print("WATCH OUT: Floating-point with arange")
print("  np.arange(0, 1, 0.1) might surprise you:")
float_seq = np.arange(0, 1, 0.1)
print(f"  Result: {float_seq}")
print(f"  Length: {len(float_seq)} (not 10 due to floating-point precision)")
print("  This is why I often use linspace for float ranges")
print()

# =============================================================================
# PART 2: linspace() - The "Divider"
# =============================================================================
print("=" * 60)
print("PART 2: np.linspace() - When You Know How Many Points")
print("=" * 60)

# I use linspace when I need exact control over point count
print("USE CASE 1: Creating smooth plots")
print("When plotting sin(x) from 0 to 2π:")
x_smooth = np.linspace(0, 2*np.pi, 100)
print(f"  I get {len(x_smooth)} x-values for a smooth curve")
print(f"  First 5 points: {x_smooth[:5]:.3f}")
print("  This ensures my plot has enough resolution")
print()

print("USE CASE 2: Creating test splits")
print("For machine learning, splitting data 70-30:")
split_point = np.linspace(0, 100, 11)  # 0%, 10%, 20%, ..., 100%
print(f"  Percentage points: {split_point}")
print(f"  70% split at index: {np.where(split_point == 70)[0][0]}")
print()

# Parameter I use often: endpoint
print("CONTROLLING ENDPOINTS:")
print("  By default, endpoint=True (include stop):")
with_end = np.linspace(0, 1, 5)
print(f"    np.linspace(0, 1, 5) = {with_end}")
print()
print("  Sometimes I want endpoint=False (exclude stop):")
without_end = np.linspace(0, 1, 5, endpoint=False)
print(f"    np.linspace(0, 1, 5, endpoint=False) = {without_end}")
print("  Useful when creating cyclic or periodic ranges")
print()

# =============================================================================
# PART 3: Side-by-Side Comparison
# =============================================================================
print("=" * 60)
print("PART 3: Decision Framework - Which Should I Use?")
print("=" * 60)

print("QUESTION 1: Do you know the STEP SIZE or NUMBER OF POINTS?")
print("  Step size known → arange")
print("  Number of points known → linspace")
print()

print("EXAMPLE: Creating values from 0 to π")
print("  With arange (step-based):")
arange_pi = np.arange(0, np.pi, np.pi/4)
print(f"    np.arange(0, π, π/4) = {arange_pi:.3f}")
print(f"    Gives me {len(arange_pi)} points")
print()
print("  With linspace (count-based):")
linspace_pi = np.linspace(0, np.pi, 5)
print(f"    np.linspace(0, π, 5) = {linspace_pi:.3f}")
print(f"    Always gives me exactly 5 points")
print()

print("QUESTION 2: Do you need the ENDPOINT INCLUDED?")
print("  Need endpoint → linspace(endpoint=True)")
print("  Don't need endpoint → arange or linspace(endpoint=False)")
print()

# =============================================================================
# PART 4: My Practical Applications
# =============================================================================
print("=" * 60)
print("PART 4: How I Actually Use These in ML Projects")
print("=" * 60)

print("APPLICATION 1: Hyperparameter tuning")
print("  When tuning learning rates:")
learning_rates = np.logspace(-3, 0, 4)  # Creates [0.001, 0.01, 0.1, 1.0]
print(f"  np.logspace(-3, 0, 4) = {learning_rates:.3f}")
print("  (logspace is linspace's logarithmic cousin)")
print()

print("APPLICATION 2: Creating training batches")
print("  For batch indices:")
batch_size = 32
n_samples = 100
batch_starts = np.arange(0, n_samples, batch_size)
print(f"  Batch start indices: {batch_starts}")
print("  Each batch: data[start:start+batch_size]")
print()

print("APPLICATION 3: Visualization grids")
print("  For heatmaps or 3D plots:")
x_grid = np.linspace(-2, 2, 50)
y_grid = np.linspace(-2, 2, 50)
print(f"  Created {len(x_grid)}×{len(y_grid)} = {len(x_grid)*len(y_grid)} grid points")
print("  Perfect for evaluating functions over 2D space")
print()

# =============================================================================
# PART 5: Quick Reference & Memory Aids
# =============================================================================
print("=" * 60)
print("PART 5: Quick Reference - My Mental Cheat Sheet")
print("=" * 60)

print("ARANGE (Step-based):")
print("  Syntax: np.arange(start, stop, step)")
print("  Think: 'range' with 'a' for 'array'")
print("  Memory aid: 'a-range' → array version of Python's range()")
print()

print("LINSPACE (Count-based):")
print("  Syntax: np.linspace(start, stop, num)")
print("  Think: 'linear space'")
print("  Memory aid: 'lin' for linear, 'space' for spacing")
print()

print("MY RULE OF THUMB:")
print("  1. Working with indices or steps → arange")
print("  2. Creating plot points or test values → linspace")
print("  3. Unsure about floating-point issues → linspace")
print("  4. Need exact endpoint control → linspace with endpoint parameter")
print()

# =============================================================================
# PART 6: Practice with Purpose
# =============================================================================
print("=" * 60)
print("PART 6: Build Your Intuition - Try These")
print("=" * 60)

print("EXERCISE 1: Create a 24-hour timeline")
print("  You need hourly measurements from 0:00 to 23:00")
print("  Which function? Why?")
print("  My solution:")
hours = np.arange(0, 24, 1)
print(f"    np.arange(0, 24, 1) = {hours}")
print("    Reason: Known step size (1 hour), don't need 24:00")
print()

print("EXERCISE 2: Create color gradient")
print("  You need 256 values from black (0) to white (1)")
print("  Which function? Why?")
print("  My solution:")
gradient = np.linspace(0, 1, 256)
print(f"    First 5 values: {gradient[:5]:.3f}")
print(f"    Last 5 values: {gradient[-5:]:.3f}")
print("    Reason: Need exact number of points, both endpoints included")
print()

print("EXERCISE 3: The edge case")
print("  What happens with arange(0, 0.6, 0.1)?")
edge_case = np.arange(0, 0.6, 0.1)
print(f"  Result: {edge_case}")
print(f"  Length: {len(edge_case)} (notice 0.5 is included)")
print("  This is why I'm careful with float stop values")
print()

print("=" * 60)
print("FINAL THOUGHT FROM KIRA-ML:")
print("=" * 60)
print("Don't memorize syntax - build intuition.")
print("Ask: 'Do I want to count steps or divide distance?'")
print("Both are tools; knowing when to use each is the skill.")
print()