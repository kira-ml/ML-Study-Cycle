import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)


class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None


    def _init_weights(self, n_features):
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
    

    def _activation(self, x):
        return np.where(x >= 0, 1, -1)
    
    def predict(self, X):
        if self.weights is None or self.bias is None:
            raise ValueError("Model weights are uninitialized. Call 'fit' before 'predict'.")
        X = np.atleast_2d(X)
        linear_output = np.dot(X, self.weights) + self.bias
        return self._activation(linear_output)
    

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._init_weights(n_features)
        y_ = np.where(y <= 0, -1, 1)  # Ensure labels are -1 or 1
        for _ in range(self.n_iters):
            for xi, target in zip(X, y_):
                linear_output = np.dot(xi, self.weights) + self.bias
                y_pred = self._activation(linear_output)
                update = self.lr * (target - y_pred)
                self.weights += update * xi
                self.bias += update

    def plot_decision_boundary(self, X, y):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(grid)
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.coolwarm)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, edgecolor='k', cmap=plt.cm.coolwarm)
        plt.title("Perceptron Decision Boundary")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()


if __name__ == "__main__":
    X = np.array([[1, 1], [2, 2], [-1, -2], [-2, -1]])
    y = np.array([1, 1, -1, -1])

    p = Perceptron(learning_rate=0.1, n_iters=100)
    p.fit(X, y)
    p.plot_decision_boundary(X, y)