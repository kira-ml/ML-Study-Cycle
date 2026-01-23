"""
Vector Indexing Operations in NumPy
Author: kira-ml

This module demonstrates fundamental array indexing and modification techniques
using NumPy, a core library for numerical computing in Python.
"""

import numpy as np

# Initialize a one-dimensional NumPy array
data = np.array([10, 20, 30, 40, 50])

# Display initial array state
print(f"Original vector: {data}")
print()

# ============================================================================
# Positive Indexing Operation
# ============================================================================
"""
Positive indexing accesses elements from the array beginning.
Indices range from 0 to n-1 for an array of length n.
"""
second_element = data[1]  # Retrieve element at index position 1

print(f"1. Element at index 1: {second_element}")
print(f"   Method: data[1] accesses the second array element")
print()

# ============================================================================
# Negative Indexing Modification
# ============================================================================
"""
Negative indexing provides access from the array end.
Index -1 references the final element, -2 the penultimate, etc.
"""
print("2. Modifying terminal element from 50 to 99...")

data[-1] = 99  # Update the last element using negative index notation

print(f"   Operation: data[-1] = 99")
print(f"   Note: Negative indices provide end-relative positioning")
print()

# ============================================================================
# Verification of Modification
# ============================================================================
print("Modified vector:", data)
print(f"   Current terminal element: {data[-1]}")