import numpy as np


def set_random_seed(seed: int = 42) -> None:

    np.random.seed(seed)
    

def make_linear_system(m: int, n: int, noise_std: float = 0.0, seed: int = 42):



    set_random_seed(seed)
    A = np.random.randn(m, n)
    x_true = np.random.randn(n)
    b = A @ x_true
    if noise_std > 0:
        b = b + np.random.randn(m) * noise_std

    return A, x_true, b



def pseudoinverse_svd(A: np.ndarray, rcond: float | None = None) -> np.ndarray:

    U, s, Vt = np.linalg.svd(A, full_matrices=False)


    if  rcond is None:
        eps = np.finfo(float).eps
        rcond = max(A.shape) * eps * (s[0] if s.size else 1.0)


    s_inv = np.array([1.0/si if si > rcond else 0.0 for si in s])

if __name__ == "__main__":
    A, x_true, b = make_linear_system(m=8, n=5, noise_std=0.1, seed=0)
    
    print("A shape" A.shape, "b shape", b.shape, "x true shape", x_true.shape)