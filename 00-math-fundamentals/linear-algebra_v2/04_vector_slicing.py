"""
NumPy Array Slicing: Advanced Operations and Memory Management
Author: kira-ml

This module demonstrates professional array slicing techniques in NumPy,
highlighting both syntactic variations and critical memory behavior patterns.
Essential for scientific computing, data analysis, and machine learning workflows.
"""

import numpy as np

"""
OVERVIEW:
Array slicing provides efficient access to data subsets without full copies.
Understanding the distinction between views (shared memory) and copies 
(independent memory) is crucial for performance optimization and data integrity.
"""

# Initialize demonstration array
# 7-element vector with linear progression for clear indexing visualization
v = np.array([0, 5, 10, 15, 20, 25, 30])

# Display array metadata
print(f"Original vector: {v}")
print(f"Array dimensions: {v.shape}")
print(f"Element count: {len(v)}")
print()

# ============================================================================
# SECTION 1: FOUNDATIONAL SLICING SYNTAX
# ============================================================================
"""
SLICE SYNTAX: array[start:stop:step]
- start: inclusive starting index (default=0)
- stop: exclusive ending index (default=array length)
- step: sampling interval (default=1)

Edge cases and defaults demonstrate Python's design philosophy:
"Explicit is better than implicit, but practicality beats purity."
"""

print("=" * 60)
print("SECTION 1: Fundamental Slicing Patterns")
print("=" * 60)

# Pattern 1: Prefix extraction
first_three = v[0:3]
print(f"1. Prefix slice v[0:3]: {first_three}")
print(f"   Zero-based indexing: positions 0, 1, 2")
print(f"   Alternative: v[:3] (implicit start=0)")
print()

# Pattern 2: Contiguous subsequence
middle_section = v[2:5]
print(f"2. Contiguous slice v[2:5]: {middle_section}")
print(f"   Mathematical notation: v[2], v[3], v[4]")
print(f"   Exclusive boundary: index 5 not included")
print()

# Pattern 3: Strided sampling
every_other = v[::2]
print(f"3. Strided slice v[::2]: {every_other}")
print(f"   Step parameter: select every 2nd element")
print(f"   Index sequence: {list(range(0, len(v), 2))}")
print()

# ============================================================================
# SECTION 2: ADVANCED INDEXING PATTERNS
# ============================================================================
"""
Advanced patterns leverage Python's flexible slicing semantics:
- Omitted parameters use sensible defaults
- Negative indices count from array end
- Negative step enables reverse traversal
"""

print("=" * 60)
print("SECTION 2: Advanced Indexing Techniques")
print("=" * 60)

# Omission pattern: default to boundaries
from_index_3 = v[3:]
print(f"v[3:]  → Elements from index 3: {from_index_3}")

# End-relative indexing
last_three = v[-3:]
print(f"v[-3:] → Terminal elements: {last_three}")

# Reverse traversal
reversed_v = v[::-1]
print(f"v[::-1] → Reversed sequence: {reversed_v}")

# Offset strided sampling
every_other_from_1 = v[1::2]
print(f"v[1::2] → Alternate elements from index 1: {every_other_from_1}")
print()

# ============================================================================
# SECTION 3: MEMORY ARCHITECTURE - VIEWS VS. COPIES
# ============================================================================
"""
CRITICAL DISTINCTION:
- View: Memory-efficient reference to original data (O(1) space)
- Copy: Independent memory allocation (O(n) space)

Professional applications require conscious choice between these paradigms
based on mutability requirements and performance constraints.
"""

print("=" * 60)
print("SECTION 3: Memory Management - Views and Copies")
print("=" * 60)

# View demonstration: shared memory reference
print("\nVIEW BEHAVIOR (default slicing):")
slice_view = v[2:5]  # Creates view, not copy
reference_value = v[2]  # Preserve for comparison

print(f"Initial state:")
print(f"  slice_view = {slice_view}")
print(f"  v = {v}")

slice_view[0] = 999  # Modification propagates to original

print(f"\nAfter slice_view[0] = 999:")
print(f"  slice_view = {slice_view}")
print(f"  v = {v}")
print(f"  Observation: View modification alters original array")
print(f"  Memory efficiency: Shared buffer, no duplication")

# Restore original state
v[2] = reference_value

# Copy demonstration: independent memory allocation
print("\nCOPY BEHAVIOR (explicit duplication):")
slice_copy = v[2:5].copy()  # Allocate separate memory

print(f"Initial state:")
print(f"  slice_copy = {slice_copy}")
print(f"  v = {v}")

slice_copy[0] = 888  # Modification isolated to copy

print(f"\nAfter slice_copy[0] = 888:")
print(f"  slice_copy = {slice_copy}")
print(f"  v = {v}")
print(f"  Observation: Copy modification isolated")
print(f"  Memory cost: Full duplication of selected elements")
print()

# ============================================================================
# SECTION 4: ENGINEERING CONSIDERATIONS
# ============================================================================
"""
PROFESSIONAL GUIDELINES:

View usage recommended when:
- Operations are read-only
- Memory constraints exist
- Real-time data streaming

Copy usage required when:
- Data integrity must be preserved
- Concurrent modifications occur
- Function returns modified subsets
"""

print("=" * 60)
print("SECTION 4: Engineering Best Practices")
print("=" * 60)

print("\nView-Copy Decision Matrix:")
print("┌─────────────────┬──────────────────────────────────────┐")
print("│ Criterion       │ View               │ Copy            │")
print("├─────────────────┼────────────────────┼─────────────────┤")
print("│ Memory          │ O(1)               │ O(n)            │")
print("│ Speed           │ Instant            │ Allocation time │")
print("│ Mutability      │ Affects original   │ Isolated        │")
print("│ Use Case        │ Inspection         │ Transformation  │")
print("└─────────────────┴────────────────────┴─────────────────┘")

print("\nPerformance Characteristics:")
print("  • Views: Constant time/space complexity")
print("  • Copies: Linear space complexity")

print("\nImplementation Notes:")
print("  1. Default slicing creates views (memory efficient)")
print("  2. Use .copy() method for data isolation")
print("  3. Large-scale operations require careful memory planning")
print()

# ============================================================================
# ADDITIONAL TECHNICAL NOTES
# ============================================================================
"""
NUMERICAL COMPUTING CONTEXT:

NumPy's view-copy distinction stems from its C-based memory model,
optimized for numerical operations on contiguous memory blocks.

For educational audiences:
- Views demonstrate pointer semantics in high-level abstraction
- Copies illustrate defensive programming patterns
- The trade-off exemplifies computer science's space-time duality
"""