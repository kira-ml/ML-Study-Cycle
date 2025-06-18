import numpy as np

A = np.array([[4, 1], [2, 3]])


A = np.random.rand(5, 5)


b = np.random.rand(A.shape[1])


num_iterations = 100


tolerance = 1e-10




for i in range(num_iterations):
    b = np.dot(A, b)
    b_norm = np.linalg.norm(b)

    b = b / b_norm


eigenvalue = np.dot(b.T, np.dot(A, b)) / np.dot(b.T, b)


print("Final estimated eigenvectors:")
print(b)

print("\nEstimated dominant eigenvalue:", eigenvalue)