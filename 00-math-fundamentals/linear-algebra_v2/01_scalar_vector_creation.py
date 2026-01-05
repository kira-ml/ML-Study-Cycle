import numpy as np

temperature = 98.6


print("Part 1: Scalar example")
print(f"Temperature: {temperature}")
print(f"Type of temperature: {type(temperature)}")
print(f"Is this a scalar? Yes, because it's a single value: {temperature}\n")


feature_vector = np.array([2.5, -1.0, 4.7, 0.3])

print("Part 2: Vector Example")
print(f"Feature Vector: {feature_vector}")
print(f"Type of feature_vector: {type(feature_vector)}")
print(f"Shape of feature_vector: {feature_vector.shape}")  # Shows dimensions
print(f"Number of elements: {len(feature_vector)}")
print(f"Is this a vector? Yes, because it's an ordered collection of values: {feature_vector}\n")



print("Part 3: Comparison")
print("Scalar (temperature):")
print(f"  Value: {temperature}")
print(f"  Dimension: 0 (just a point)")
print(f"  Can do: temperature + 5 = {temperature + 5}")

print("\nVector (feature_vector):")
print(f"  Value: {feature_vector}")
print(f"  Dimension: 1 (a line of points)")
print(f"  Can do: feature_vector + 5 = {feature_vector + 5}")
print(f"  Can access individual elements: feature_vector[0] = {feature_vector[0]} (first element)")
print(f"  Can access individual elements: feature_vector[1] = {feature_vector[1]} (second element)")
