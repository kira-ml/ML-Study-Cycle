import numpy as np

"""
NumPy Fundamentals: Understanding Scalars and Vectors
This tutorial demonstrates the distinction between scalar values and vector
arrays in numerical computing, highlighting their characteristics and use cases.
"""

# PART 1: Scalar Values - Single Data Points
# Scalars represent single numerical values without dimensionality.
# Common examples: temperature readings, prices, measurements, or constants.
temperature = 98.6  # A scalar representing a single temperature value

print("Part 1: Scalar Value Demonstration")
print(f"Temperature: {temperature}")
print(f"Data type: {type(temperature)}")
print(f"Is this a scalar? Yes - it represents a single numerical value: {temperature}")
print("Characteristics: No dimensional attributes, operates as individual data point")
print()

# PART 2: Vector Arrays - Ordered Collections
# Vectors are 1-dimensional arrays containing multiple ordered elements.
# Applications: feature sets, coordinates, time series data, parameter vectors.
feature_vector = np.array([2.5, -1.0, 4.7, 0.3])  # Four-element feature vector

print("Part 2: Vector Array Demonstration")
print(f"Feature vector: {feature_vector}")
print(f"Data type: {type(feature_vector)} (NumPy ndarray)")
print(f"Shape attribute: {feature_vector.shape}")
print("Note: (4,) indicates a 1-dimensional array with 4 elements")
print(f"Element count: {len(feature_vector)} features")
print("Application: Ordered numerical collections for mathematical operations")
print()

# PART 3: Scalar vs Vector Comparison
# This section contrasts their properties and operational capabilities.
print("Part 3: Scalar vs Vector Comparison")
print("=" * 60)

print("Scalar Properties (temperature):")
print(f"  Value: {temperature} (single numerical element)")
print(f"  Dimensionality: 0-dimensional (point data)")
print(f"  Mathematical operation: temperature + 5 = {temperature + 5}")
print("  Limitation: Cannot access individual elements or perform vector operations")

print("\nVector Properties (feature_vector):")
print(f"  Value: {feature_vector} (ordered element collection)")
print(f"  Dimensionality: 1-dimensional (linear array)")
print(f"  Mathematical operation: feature_vector + 5 = {feature_vector + 5}")
print("  Note: Broadcasting applies scalar operation to all elements")
print(f"  Element access: feature_vector[0] = {feature_vector[0]} (index-based retrieval)")
print(f"  Element access: feature_vector[1] = {feature_vector[1]} (second position)")
print("  Capability: Supports indexing, slicing, and vectorized operations")

print("\n" + "=" * 60)
print("Summary:")
print("• Scalar: Individual numerical value with no dimensional structure")
print("• Vector: Ordered 1-dimensional array enabling element-wise operations")
print("• Both essential for numerical computing with distinct mathematical properties")
print()