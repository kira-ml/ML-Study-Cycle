import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Dict


def rosenbrock(x: np.ndarray, a: float = 1.0, b: float = 100.0) -> Tuple[float, np.ndarray]:
    """
    Compute the Rosenbrock function value and gradient.
    
    The Rosenbrock function is a non-convex function used as a performance test problem for optimization algorithms.
    It has a global minimum at (a, a^2) where the function value is 0.
    The classic form uses a=1, b=100: f(x,y) = (1-x)^2 + 100(y-x^2)^2
    
    Args:
        x: Input array of shape (2,) containing [x1, x2]
        a: First parameter controlling the location of the minimum (default 1.0)
        b: Second parameter controlling the steepness of the valley (default 100.0)
    
    Returns:
        Tuple containing:
        - function value at x
        - gradient array of shape (2,) with partial derivatives
    """
    if x.shape[0] < 2:
        raise ValueError("Input must have at least 2 dimensions")
    
    # Extract coordinates
    x1, x2 = x[0], x[1]
    
    # Compute function value: f(x1, x2) = (a - x1)^2 + b * (x2 - x1^2)^2
    loss = (a - x1)**2 + b * (x2 - x1**2)**2
    
    # Compute partial derivatives
    # df/dx1 = -2(a - x1) - 4bx1(x2 - x1^2)
    # df/dx2 = 2b(x2 - x1^2)
    grad = np.zeros(2)
    grad[0] = -2 * (a - x1) - 4 * b * x1 * (x2 - x1**2)  # Partial derivative w.r.t. x1
    grad[1] = 2 * b * (x2 - x1**2)                        # Partial derivative w.r.t. x2
    
    return loss, grad


def gradient_descent(
        grad_fn: Callable,
        x0: np.ndarray,
        learning_rate: float = 0.001,
        max_iters: int = 1000,
        tol: float = 1e-6
) -> Dict:
    """
    Perform gradient descent optimization to minimize a function.
    
    Gradient descent is a first-order iterative optimization algorithm that moves in the direction
    opposite to the gradient to find local minima. Each step is proportional to the negative of
    the gradient of the function at the current point.
    
    Args:
        grad_fn: Function that takes x and returns (loss_value, gradient_array)
        x0: Initial parameter vector
        learning_rate: Step size for each iteration (controls the size of parameter updates)
        max_iters: Maximum number of iterations to perform
        tol: Convergence tolerance - algorithm stops when parameter change is below this value
    
    Returns:
        Dictionary containing:
        - 'x_opt': Optimal parameter vector found
        - 'trajectory': Array of parameter vectors at each iteration
        - 'losses': List of loss values at each iteration
        - 'iterations': Number of iterations performed
    """
    x = x0.copy()  # Work with a copy to avoid modifying the original
    trajectory = [x.copy()]  # Track parameter evolution
    losses = []  # Track loss evolution

    for i in range(max_iters):
        # Get current function value and gradient
        loss, grad = grad_fn(x)
        losses.append(loss)

        # Update parameters: x_new = x_old - learning_rate * gradient
        # This moves in the direction of steepest descent
        x_new = x - learning_rate * grad
        trajectory.append(x_new.copy())

        # Check for convergence based on parameter change magnitude
        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new

    return {
        'x_opt': x,  # Final optimized parameters
        'trajectory': np.array(trajectory),  # History of parameter values
        'losses': losses,  # History of loss values
        'iterations': i + 1  # Total iterations performed
    }


def rmsprop(
        grad_fn: Callable,
        x0: np.ndarray,
        learning_rate: float = 0.01,
        beta: float = 0.9,
        epsilon: float = 1e-8,
        max_iters: int = 1000,
        tol: float = 1e-6
) -> Dict:
    """
    Perform RMSProp optimization, an adaptive learning rate method.
    
    RMSProp (Root Mean Square Propagation) is an adaptive learning rate optimization algorithm
    designed to work well in online settings. It maintains a moving average of squared gradients
    and normalizes the gradient by the square root of this average, helping to adapt the learning
    rate for each parameter based on recent gradient magnitudes.
    
    Args:
        grad_fn: Function that takes x and returns (loss_value, gradient_array)
        x0: Initial parameter vector
        learning_rate: Base learning rate (will be adaptively scaled)
        beta: Decay rate for moving average of squared gradients (typically 0.9)
        epsilon: Small constant to prevent division by zero (numerical stability)
        max_iters: Maximum number of iterations to perform
        tol: Convergence tolerance - algorithm stops when parameter change is below this value
    
    Returns:
        Dictionary containing:
        - 'x_opt': Optimal parameter vector found
        - 'trajectory': Array of parameter vectors at each iteration
        - 'losses': List of loss values at each iteration
        - 'iterations': Number of iterations performed
        - 'final_s': Final moving average of squared gradients
    """
    x = x0.copy()  # Work with a copy to avoid modifying the original
    trajectory = [x.copy()]  # Track parameter evolution
    losses = []  # Track loss evolution
    s = np.zeros_like(x)  # Initialize moving average of squared gradients

    for i in range(max_iters):
        # Get current function value and gradient
        loss, grad = grad_fn(x)
        losses.append(loss)

        # Update moving average of squared gradients: s = β*s + (1-β)*g²
        # This exponentially decays past squared gradients
        s = beta * s + (1 - beta) * (grad ** 2)

        # Update parameters with adaptive learning rate: each parameter's step size
        # is inversely proportional to square root of its recent squared gradients
        x_new = x - learning_rate * grad / (np.sqrt(s) + epsilon)
        trajectory.append(x_new.copy())

        # Check for convergence based on parameter change magnitude
        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new

    return {
        'x_opt': x,  # Final optimized parameters
        'trajectory': np.array(trajectory),  # History of parameter values
        'losses': losses,  # History of loss values
        'iterations': i + 1,  # Total iterations performed
        'final_s': s  # Final moving average of squared gradients
    }


# Test the functions with sample inputs
x_test = np.array([0.0, 0.0])  # Test point at origin
loss, grad = rosenbrock(x_test)
print(f"Rosenbrock at {x_test}: loss = {loss:.2f}, grad = {grad}")

# Initialize optimization from point [-1, 2]
x0 = np.array([-1.0, 2.0])

# Perform gradient descent optimization
gd_result = gradient_descent(rosenbrock, x0, learning_rate=0.001)
print(f"Vanilla GD: found minimum at {gd_result['x_opt']} in {gd_result['iterations']} iterations")
print(f"Final loss: {rosenbrock(gd_result['x_opt'])[0]:.6f}")

# Perform RMSProp optimization
rms_result = rmsprop(rosenbrock, x0, learning_rate=0.01, beta=0.9)
print(f"RMSProp: found minimum at {rms_result['x_opt']} in {rms_result['iterations']} iterations")
print(f"Final loss: {rosenbrock(rms_result['x_opt'])[0]:.6f}")