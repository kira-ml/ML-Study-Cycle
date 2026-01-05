import numpy as np
import time

def vector_dot(v1: np.ndarray, v2: np.ndarray) -> float:



    result = 0.0
    for i in range(len(v1)):
        result += v1[i] * v2[i]

    return result



def vector_outer(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:


    m, n = len(v1), len(v2)

    result = np.zeros((m, n))


    for i in range(m):
        for j in range(n):
            result[i, j] = v1[i] * v2[j]

            