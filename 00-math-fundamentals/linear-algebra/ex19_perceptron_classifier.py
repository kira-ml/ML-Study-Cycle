import numpy as np
import matplotlib.pyplot as plt
import random

# Set a global seed for reproducibility across the entire script
np.random.seed(42)


class Perceptron:
    """
    A simple Perceptron binary classifier.

    This class implements the original Perceptron learning algorithm, which
    learns a linear decision boundary for binary classification using a step
    function as its activation.

    Attributes:
        lr (float): The learning rate controlling the size of weight updates.
        n_iters (int): Number of iterations over the training dataset.
        weights (ndarray): Weight vector (initialized during training).
        bias (float): Bias term (initialized during training).
        random_state (int): Seed for reproducibility of weight initialization.
    """

    def __init__(self, learning_rate=0.01, n_iters=1000, random_state=42):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.random_state = random_state

    def _init_weights(self, n_features):
        """
        Initializes weights and bias with small random values.

        We use a small random initialization for weights to break symmetry,
        which is necessary to ensure that different weights are updated differently.

        Args:
            n_features (int): The number of input features.
        """
        np.random.seed(self.random_state)  # Ensures consistent initialization
        self.weights = np.random.randn(n_features) * 0.01  # Small Gaussian noise
        self.bias = 0.0

    def _activation(self, x):
        """
        Applies the step function (sign) activation.

        The Perceptron uses this non-differentiable threshold function to make binary predictions.
        """
        return np.where(x >= 0, 1, -1)

    def predict(self, X):
        """
        Predicts binary labels for given input samples.

        Args:
            X (ndarray): Input feature matrix of shape (n_samples, n_features) or a single sample.

        Returns:
            ndarray: Predicted labels (1 or -1).
        """
        if self.weights is None:
            raise RuntimeError("Model is untrained. Please call 'fit()' before prediction.")

        # Ensure input is 2D (even if a single sample is passed)
        X = np.atleast_2d(X)
        linear_output = np.dot(X, self.weights) + self.bias
        return self._activation(linear_output)

    def fit(self, X, y):
        """
        Trains the Perceptron using the input features and binary labels.

        This method updates weights iteratively using the Perceptron rule:
        w = w + lr * (y_true - y_pred) * x

        Args:
            X (ndarray): Input feature matrix (n_samples, n_features).
            y (ndarray): Target labels (can be any numeric binary values).

        Raises:
            ValueError: If inputs are incorrectly shaped or labels are not binary.
        """
        if X.ndim != 2:
            raise ValueError("X must be 2D array (n_samples, n_features)")
        if len(np.unique(y)) > 2:
            raise ValueError("Perceptron supports binary classification only")

        n_samples, n_features = X.shape
        self._init_weights(n_features)
        y_ = self._binarize_labels(y)  # Normalize labels to -1 and 1

        for _ in range(self.n_iters):
            for xi, target in zip(X, y_):
                # Compute model output and update rule
                linear_output = np.dot(xi, self.weights) + self.bias
                y_pred = self._activation(linear_output)
                update = self.lr * (target - y_pred)

                # Apply Perceptron weight update rule
                self.weights += update * xi
                self.bias += update

    def _binarize_labels(self, y):
        """
        Converts arbitrary binary labels to standard -1 and 1 encoding.

        This ensures compatibility with the Perceptron update rule.

        Args:
            y (ndarray): Original label vector

        Returns:
            ndarray: Labels converted to -1 or 1
        """
        return np.where(y <= 0, -1, 1)

    def plot_decision_boundary(self, X, y):
        """
        Visualizes the learned decision boundary in 2D.

        Only works when the input features are 2-dimensional.

        Args:
            X (ndarray): Input features
            y (ndarray): Binary labels
        """
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        # Create a grid of points across the input space
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        grid = np.c_[xx.ravel(), yy.ravel()]

        # Predict labels across the grid
        Z = self.predict(grid).reshape(xx.shape)

        # Plot decision surface and training data
        plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.coolwarm)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, edgecolor='k', cmap=plt.cm.coolwarm)
        plt.title("Perceptron Decision Boundary")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()


if __name__ == "__main__":
    # Define a simple 2D dataset for demonstration
    X = np.array([[1, 1], [2, 2], [-1, -2], [-2, -1]])
    y = np.array([1, 1, -1, -1])

    # Instantiate and train the Perceptron model
    p = Perceptron(learning_rate=0.1, n_iters=100)
    p.fit(X, y)
    p.plot_decision_boundary(X, y)
