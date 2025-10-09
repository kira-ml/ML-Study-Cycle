import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Dict



def rosenbrock(x: np.ndarray, a: float = 1.0, b: float = 100.0) -> Tuple[float, np.ndarray]:


    x1, x2 = x[0], x[1]

    loss = (a - x1)**2 + b * (x2 - x1**2)**2

    grad = np.zeros(2)
    grad[0] = -2 * (a - x1) - 4 * b * x1 * (x2 - x1**2)
    grad[1] = 2 * b * (x2 - x1**2)

    return loss, grad


x_test = np.array([0.0, 0.0])
loss, grad = rosenbrock(x_test)
print(f"Rosenbrock at {x_test}: loss = {loss:.2f}, grad = {grad}")