import numpy as np


def affine_forward(x: np.ndarray, W: np.ndarray, b: np.ndarray):

    out = x @ W + b
    cache = (x, W, b)

    return out, cache






def affine_backward(dout: np.ndarray, cache: tuple):

    x, W, b = cache

    dx = dout @ W.T
    dW = x.T @ dout
    db = np.sum(dout, axis=0)


    return dx, dW, db






np.random.seed(42)
x = np.random.randn(2, 3)
W = np.random.randn(3, 4)
b = np.random.randn(4)


out, cache = affine_forward(x, W, b)


dout = np.random.randn(2, 4)

dx, dW, db = affine_backward(dout, cache)
print("Affine forward output\n", out)


print("dx shape:", dx.shape)
print("dW shape:", dW.shape)
print("db shape:", db.shape)



