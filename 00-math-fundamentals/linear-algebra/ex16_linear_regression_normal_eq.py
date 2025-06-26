import numpy as np

X = np.array([[500], [1200], [1500], [1800], [2100]])

y = np.array([150000, 220000, 350000, 420000, 500000])

print("Feature Matrix X: \n", X)
print("Target vector y: \n", y)

X_b = np.c_[np.ones((X.shape[0], 1)), X]

print("X with bias term: \n", X_b)

theta = np.linalg.solve(X_b.T.dot(X_b), X_b.T.dot(y))
print("Computed Parameters (θ):\n", theta)


def predict(area, theta):
    x_with_bias = np.c_[np.ones((1, 1)), np.array([[area]])]
    return x_with_bias.dot(theta)

predicted_price = predict(1000, theta)