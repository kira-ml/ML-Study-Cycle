import numpy as np

"""
NumPy Array Slicing Tutorial
Demonstrates array slicing techniques, syntax variations, and memory behavior.
Key concepts: slice notation [start:stop:step], view vs. copy behavior.
"""

# Initialize a sample vector for slicing demonstrations
# This creates a 1-dimensional array with 7 elements
v = np.array([0, 5, 10, 15, 20, 25, 30])

# Display original array properties
print(f"Original vector: {v}")
print(f"Shape of v: {v.shape}")
print(f"Array length: {len(v)}")
print()

# SECTION 1: Basic Slicing Examples
# ----------------------------------
# Slicing syntax: array[start:stop:step]
# - start: inclusive beginning index (default 0)
# - stop: exclusive ending index (default array length)
# - step: increment between elements (default 1)

print("=" * 60)
print("SECTION 1: Basic Slicing Operations")
print("=" * 60)

# Example 1: Extract first three elements
# v[0:3] or v[:3] - both yield same result
first_three = v[0:3]
print(f"1. First three elements (v[0:3]): {first_three}")
print(f"   Interpretation: indices 0 through 2 (stop index 3 is exclusive)")
print(f"   Equivalent syntax: v[:3]")
print()

# Example 2: Middle segment extraction
middle_section = v[2:5]
print(f"2. Middle section (v[2:5]): {middle_section}")
print(f"   Interpretation: indices 2, 3, 4")
print(f"   Note: Index 5 is exclusive boundary")
print()

# Example 3: Strided extraction with step parameter
every_other = v[::2]
print(f"3. Every other element (v[::2]): {every_other}")
print(f"   Interpretation: start=0, stop=end, step=2")
print(f"   Result indices: {[i for i in range(0, len(v), 2)]}")
print()

# SECTION 2: Advanced Slicing Patterns
# ------------------------------------
# Demonstrates negative indexing, reverse stepping, and omission patterns

print("=" * 60)
print("SECTION 2: Advanced Slicing Patterns")
print("=" * 60)

# From specific index to end (omitted stop parameter)
from_index_3 = v[3:]
print(f"v[3:]  → Elements from index 3 to end: {from_index_3}")

# Negative indexing for end-relative slicing
last_three = v[-3:]
print(f"v[-3:] → Last three elements: {last_three}")

# Reverse array with negative step
reversed_v = v[::-1]
print(f"v[::-1] → Reversed array: {reversed_v}")

# Strided extraction from non-zero start
every_other_from_1 = v[1::2]
print(f"v[1::2] → Every other element from index 1: {every_other_from_1}")
print()

# SECTION 3: Memory Behavior - Views vs. Copies
# ---------------------------------------------
# Critical concept: Slicing creates views (shared memory) by default
# The .copy() method creates independent copies

print("=" * 60)
print("SECTION 3: Memory Behavior - Views vs. Copies")
print("=" * 60)

# Demonstration of view behavior
print("\nMemory Note: Default slicing creates VIEWS (shared memory)")
slice_view = v[2:5]  # Creates view into v's memory
original_value = v[2]  # Store for comparison
slice_view[0] = 999    # Modify through view

print(f"After slice_view[0] = 999:")
print(f"  slice_view: {slice_view}")
print(f"  Original v: {v}")
print("  Note: Modification through view affects original array")
print("  Reason: slice_view shares memory with v")

# Reset for clarity
v[2] = original_value
print(f"\nReset v[2] to original value {original_value}")

# Demonstration of independent copy behavior
print("\nCreating independent copy with .copy() method:")
slice_copy = v[2:5].copy()  # Explicit copy, independent memory
slice_copy[0] = 888        # Modify copy only

print(f"After slice_copy[0] = 888:")
print(f"  slice_copy: {slice_copy}")
print(f"  Original v: {v}")
print("  Note: Copy modifications do not affect original")
print("  Use .copy() when independent array is required")
print()

# SECTION 4: Practical Considerations
# -----------------------------------
print("=" * 60)
print("SECTION 4: Practical Recommendations")
print("=" * 60)

print("When to use views (default slicing):")
print("  - Read-only operations")
print("  - Memory efficiency critical")
print("  - Temporary data inspection")
print()

print("When to use copies (.copy() method):")
print("  - Modifications without affecting original")
print("  - Data persistence requirements")
print("  - Function return values")
print()

print("Performance Note:")
print("  - Views: O(1) time/space (memory efficient)")
print("  - Copies: O(n) space (memory intensive for large arrays)")