import numpy as np


vector_a = np.array([2, 4, 6])
vector_b = np.array([8, 10, 12])


dot_product = np.dot(vector_a, vector_b)


magnitude_a = np.linalg.norm(vector_a)
magnitude_b = np.linalg.norm(vector_b)


cos_theta = dot_product / (magnitude_a * magnitude_b)
angle_radians = np.arccos(cos_theta)
angle_degrees = np.degrees(angle_radians)


valid = abs(dot_product) <= (magnitude_a * magnitude_b)

assert valid, "Cauchy-Schwarz inequality violated!"


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
