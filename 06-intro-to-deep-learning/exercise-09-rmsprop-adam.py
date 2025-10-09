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



def gradient_descent(
        grad_fn: Callable,
        x0: np.ndarray,
        learning_rate: float = 0.001,
        max_iters: int = 1000,
        tol: float = 1e-6

) -> Dict:
    
    x = x0.copy()
    trajectory = [x.copy()]
    losses = []


    for i in range(max_iters):
        loss, grad = grad_fn(x)
        losses.append(loss)

        x_new = x - learning_rate * grad



        trajectory.append(x_new.copy())


        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new

    return {

        'x_opt': x,
        'trajectory': np.array(trajectory),
        'losses': losses,
        'iterations': i + 1
    }

x0 = np.array([-1.0, 2.0])
gd_result = gradient_descent(rosenbrock, x0, learning_rate=0.001)

print(f"Vanilla GD: found minimum at {gd_resulto['x_opt']} in {gd_result['iterations']} iterations")

print(f"Final loss: {rosenbrock(gd_result['x_opt'])[0]:.6f} ")