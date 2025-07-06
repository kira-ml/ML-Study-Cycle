import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.rand(100, 1)


X_b = np.c_[np.ones((100, 1)), X]



def ridge_regression(X, y, alpha=1.0):
    m = X.shape[1]
    I = np.eye(m)
    I[0, 0] = 0
    theta = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y
    return theta

alphas = [0, 0.1, 1, 10, 100]
coefs = []
for alpha in alphas:
    theta = ridge_regression(X_b, y, alpha)
    coefs.append(theta.ravel())

plt.figure(figsize=(10, 6))
plt.plot(alphas, coefs)
plt.xscale('log')
plt.xlabel('Alpha (log scale)')
plt.ylabel('Coefficient value')
plt.title('Ridge coefficients as a function of regulation')
plt.legend(['intercept', 'Slope'])
plt.show()


X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
theta = ridge_regression(X_b, y, alpha=1)
y_predict = X_new_b @ theta


theta_linear = ridge_regression(X_b, y, alpha=0)
theta_ridge = ridge_regression(X_b, y, alpha=1)

print("Linear regression coefficients:", theta_linear.ravel())
print("Ridge regression coefficients:", theta_ridge.ravel())