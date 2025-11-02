import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Dict


def rosenbrock(x: np.ndarray, a: float = 1.0, b: float = 100.0) -> Tuple[float, np.ndarray]:
    """
    Compute the Rosenbrock function value and gradient.

    The Rosenbrock function is a classic non-convex optimization test problem. Its global minimum
    lies in a narrow, curved valley, making it challenging for optimization algorithms to converge.
    The global minimum is at (a, a^2) where the function value is 0. The classic form uses a=1, b=100:
    f(x,y) = (1-x)^2 + 100(y-x^2)^2. This function is particularly useful for testing optimization
    algorithms because the valley is steep in some directions and flat in others, requiring algorithms
    to balance exploration and exploitation.

    Args:
        x: Input array of shape (2,) containing [x1, x2]. The function is typically evaluated in
           2D space, though higher dimensions are possible with modifications.
        a: First parameter controlling the location of the minimum (default 1.0). The global
           minimum is located at (a, a^2).
        b: Second parameter controlling the steepness of the valley (default 100.0). A higher
           value of b makes the valley steeper and the optimization problem more challenging.

    Returns:
        Tuple containing:
        - function value at x: Scalar representing the Rosenbrock function evaluation.
        - gradient array of shape (2,): Numpy array with partial derivatives [df/dx1, df/dx2].
          The gradient points in the direction of steepest ascent.
    """
    if x.shape[0] < 2:
        raise ValueError("Input must have at least 2 dimensions")

    # Extract coordinates
    x1, x2 = x[0], x[1]

    # Compute function value: f(x1, x2) = (a - x1)^2 + b * (x2 - x1^2)^2
    # This formulation creates a parabolic valley along x2 = x1^2.
    loss = (a - x1)**2 + b * (x2 - x1**2)**2

    # Compute partial derivatives analytically:
    # df/dx1 = -2(a - x1) - 4bx1(x2 - x1^2)
    # The first term is the gradient of (a - x1)^2.
    # The second term is the gradient of b(x2 - x1^2)^2, using the chain rule.
    # df/dx2 = 2b(x2 - x1^2)
    # This is the gradient of b(x2 - x1^2)^2 with respect to x2.
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

    Gradient descent is a fundamental first-order iterative optimization algorithm. It works by
    iteratively moving in the direction opposite to the gradient of the objective function.
    The gradient points in the direction of steepest ascent, so moving against it leads to
    steepest descent. The learning rate determines the step size for each update. While simple,
    vanilla gradient descent can be sensitive to the learning rate and may converge slowly
    or oscillate if not carefully tuned.

    Args:
        grad_fn: A function that takes a parameter vector `x` and returns a tuple:
                 (loss_value, gradient_array). This function must be differentiable.
        x0: Initial parameter vector (starting point for optimization). The choice of x0
            can significantly impact the convergence path and final result.
        learning_rate: Step size for each iteration (controls the magnitude of parameter updates).
                       Too large a value can cause oscillation or divergence; too small a value
                       can lead to slow convergence.
        max_iters: Maximum number of iterations to perform. Acts as a safety net to prevent
                   infinite loops if convergence is not met.
        tol: Convergence tolerance. The algorithm stops when the L2 norm of the parameter
             change (||x_new - x_old||) falls below this value, indicating minimal progress.

    Returns:
        Dictionary containing:
        - 'x_opt': The optimized parameter vector found after termination.
        - 'trajectory': Numpy array of shape (n_iterations+1, len(x0)) containing the
                        parameter values at each step, including the initial point.
                        Useful for visualizing the optimization path.
        - 'losses': List of loss values recorded at each iteration. Can be used to plot
                    the convergence curve.
        - 'iterations': The total number of iterations performed before termination.
    """
    x = x0.copy()  # Work with a copy to avoid modifying the original input array
    trajectory = [x.copy()]  # Track parameter evolution across iterations
    losses = []  # Track loss evolution across iterations

    for i in range(max_iters):
        # Get current function value and gradient from the provided function
        loss, grad = grad_fn(x)
        losses.append(loss)

        # Update parameters: x_new = x_old - learning_rate * gradient
        # This moves in the direction of steepest descent, scaled by the learning rate.
        x_new = x - learning_rate * grad
        trajectory.append(x_new.copy())

        # Check for convergence based on parameter change magnitude
        # If the change is very small, the algorithm is likely near a minimum.
        if np.linalg.norm(x_new - x) < tol:
            print(f"Converged after {i+1} iterations.")
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
    designed to improve upon basic gradient descent, especially in settings with sparse gradients
    or non-stationary objectives. It maintains a moving average of squared gradients for each
    parameter. This average is used to normalize the current gradient, effectively providing
    a parameter-specific learning rate. This helps mitigate the problem of oscillations in
    steep directions and allows for more aggressive updates in gentle directions, leading to
    faster and more stable convergence compared to standard gradient descent.

    Args:
        grad_fn: A function that takes a parameter vector `x` and returns a tuple:
                 (loss_value, gradient_array).
        x0: Initial parameter vector (starting point for optimization).
        learning_rate: Base learning rate (will be adaptively scaled for each parameter).
                       This serves as the maximum possible step size.
        beta: Decay rate for the moving average of squared gradients (typically 0.9).
              A value close to 1.0 means past gradients have a longer-lasting influence.
        epsilon: Small constant added for numerical stability during division to prevent
                 division by zero when the moving average of squared gradients is very small.
        max_iters: Maximum number of iterations to perform.
        tol: Convergence tolerance based on parameter change magnitude.

    Returns:
        Dictionary containing:
        - 'x_opt': The optimized parameter vector found after termination.
        - 'trajectory': Numpy array of parameter vectors at each iteration.
        - 'losses': List of loss values at each iteration.
        - 'iterations': The total number of iterations performed.
        - 'final_s': The final moving average of squared gradients, which can provide
                     insight into the behavior of the optimizer near the minimum.
    """
    x = x0.copy()
    trajectory = [x.copy()]
    losses = []
    # Initialize the moving average of squared gradients to zero for each parameter
    s = np.zeros_like(x)

    for i in range(max_iters):
        # Get current function value and gradient
        loss, grad = grad_fn(x)
        losses.append(loss)

        # Update moving average of squared gradients: s = β*s + (1-β)*g²
        # This exponentially decays past squared gradients, focusing on recent history.
        s = beta * s + (1 - beta) * (grad ** 2)

        # Update parameters with adaptive learning rate.
        # The denominator sqrt(s) + epsilon acts as a per-parameter normalization factor.
        # Parameters with historically large gradients get smaller effective learning rates.
        # Parameters with historically small gradients get larger effective learning rates.
        x_new = x - learning_rate * grad / (np.sqrt(s) + epsilon)
        trajectory.append(x_new.copy())

        # Check for convergence based on parameter change magnitude
        if np.linalg.norm(x_new - x) < tol:
            print(f"RMSProp converged after {i+1} iterations.")
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
print("\n--- Gradient Descent ---")
gd_result = gradient_descent(rosenbrock, x0, learning_rate=0.001)
print(f"GD Result: x_opt = {gd_result['x_opt']}, Iterations = {gd_result['iterations']}")
print(f"Final loss: {rosenbrock(gd_result['x_opt'])[0]:.6f}")

# Perform RMSProp optimization
print("\n--- RMSProp ---")
rms_result = rmsprop(rosenbrock, x0, learning_rate=0.01, beta=0.9)
print(f"RMSProp Result: x_opt = {rms_result['x_opt']}, Iterations = {rms_result['iterations']}")
print(f"Final loss: {rosenbrock(rms_result['x_opt'])[0]:.6f}")