import numpy as np

A = np.array([
    [2, 7, 8],
    [4, 5, 9],
    [3, 8, 2],
    
], dtype=float)


m, n = A.shape

Q = np.zeros((m, n))
R = np.zeros((n, n))

for i in range(n):
    v = A[:, i]. copy()

    for j in range(i):
        R[j, i] = np.dot(Q[:, j], A[:, i])

        v -= R[j, i] * Q[:, j]


    R[i, i] = np.linalg.norm(v)

    Q[:, i] = v / R[i, i]

print("Reconstructed A:", Q @ R)
print("Original A:", A)
print("Difference:" A - Q @ R)

print("Q.T @ Q:", Q.T @ Q)