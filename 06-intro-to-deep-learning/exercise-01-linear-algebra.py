import numpy as np


def vector_dot(a: np.ndarray, b: np.ndarray) -> float:




    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Input vector must be 1D vectors")
    
    if a.shape[0] != b.shape[0]:
        raise ValueError("Input vectors must have the same length")
    
    return float(np.dot(a, b))



def vector_outer(a: np.ndarray, b: np.ndarray) -> np.ndarray:


    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Input vector must be 1D")
    return np.outer(a, b)



def batch_dot(A: np.ndarray, B: np.ndarray) -> np.ndarray:

    if A.ndim != 3 or B.ndim != 3:

        raise ValueError("Input matrices must be 3D")
    

    if A.shape[0] != B.shape[0] or A.shape[2] != B.shape[1]:

        raise ValueError("Shapes must be compatible for batch product")
    

    return np.matmul(A, B)

if __name__ == "__main__":


    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    A = np.random.randn(2, 3, 4)
    B = np.random.randn(2, 4, 5)
    result = batch_dot(A, B)


    print("Batch product of A and B", result)
    print("Outer product of v1 and v2", vector_outer(v1, v2))
    print("Dot product of v1 and v2", vector_dot(v1, v2))