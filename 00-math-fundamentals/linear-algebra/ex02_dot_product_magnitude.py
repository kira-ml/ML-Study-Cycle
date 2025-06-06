import numpy as np

# In this implementation, I demonstrate how to compute the angle between two vectors 
# using the dot product and vector magnitudes. This is a fundamental operation 
# in linear algebra with applications in machine learning, physics, and geometry.

# Define Vector A and Vector B as NumPy arrays.
# These vectors are intentionally chosen with positive integer elements 
# to keep calculations interpretable while demonstrating general principles.
vector_a = np.array([2, 4, 6])
vector_b = np.array([8, 10, 12])

# Compute the dot product of the two vectors.
# The dot product is a scalar value representing the projection of one vector onto another.
# It is central to calculating the angle between two vectors.
dot_product = np.dot(vector_a, vector_b)

# Calculate the magnitudes (Euclidean norms) of each vector.
# These values represent the lengths of the vectors in Euclidean space.
# np.linalg.norm computes the square root of the sum of squared elements.
magnitude_a = np.linalg.norm(vector_a)
magnitude_b = np.linalg.norm(vector_b)

# Compute the cosine of the angle between the two vectors.
# This is derived from the dot product formula: 
#     dot(A, B) = ||A|| * ||B|| * cos(theta)
cos_theta = dot_product / (magnitude_a * magnitude_b)

# Calculate the actual angle in radians using the arccosine of the cosine value.
angle_radians = np.arccos(cos_theta)

# Convert the angle from radians to degrees for more intuitive interpretation.
angle_degrees = np.degrees(angle_radians)

# Validate the result using the Cauchy–Schwarz inequality:
#     |dot(A, B)| ≤ ||A|| * ||B||
# This inequality always holds in real inner product spaces, and this check helps
# confirm numerical stability and correctness of our calculations.
valid = abs(dot_product) <= (magnitude_a * magnitude_b)
assert valid, "Cauchy-Schwarz inequality violated!"

# Display the computed results in a clean, formatted output.
# This includes both vector definitions and all derived quantities.
print(f"""
Vector A:           {vector_a}
Vector B:           {vector_b}
Dot Product:        {dot_product}
Magnitude of A:     {magnitude_a}
Magnitude of B:     {magnitude_b}
Cosine of angle:    {cos_theta}
Angle (radians):    {angle_radians}
Angle (degrees):    {angle_degrees}
""")
