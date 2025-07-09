import numpy as np
import matplotlib.pyplot as plt



class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None


    def _init_weights(self, n_features):
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0
    

    def _activation(self, x):
        return np.where(x >= 0, 1, -1)
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self._activation(linear_output)
    

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._init_weights(n_features)


        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                y_pred = self.predict(x_i)
                update = self.lr * (y[idx] - y_pred)
                self.weights += update * x_i
                self.bias += update    
        
    def plot_decision_boundary(self, X, y):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)


        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
        plt.title("Perceptron Decision Boundary")
        plt.show()

    
    if __name__ == "__main__":
        X = np.array([[1, 1], [2, 2], [-1, -2], [-2, -1 ]])
        y = np.array([1, 1, -1, 1])

        p = Perceptron(learning_rate=0.1, n_iters=100)
        p.fit(X, y)
        p.plot_decision_boundary(X, y)