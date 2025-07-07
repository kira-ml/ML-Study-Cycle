import numpy as np
import matplotlib.pyplot as plt

# Set a fixed random seed to ensure reproducibility of the generated dataset.
np.random.seed(42)

# Generate 100 random samples of X from a uniform distribution in [0, 2).
# This simulates one feature input for regression.
X = 2 * np.random.rand(100, 1)

# Construct the target variable y based on a linear relationship:
# y = 4 + 3*X + Gaussian noise. This creates a synthetic dataset with known ground truth.
y = 4 + 3 * X + np.random.rand(100, 1)

# Add a bias term (column of 1s) to X to account for the intercept in the linear model.
# This is standard practice when fitting models with a bias term using matrix operations.
X_b = np.c_[np.ones((100, 1)), X]


# Define Ridge Regression using closed-form solution (also known as Tikhonov regularization).
# alpha: Regularization strength. Higher alpha means more shrinkage of coefficients.
# I: Identity matrix used for regularization, excluding the bias term from penalization.
# Return: The parameter vector theta minimizing the penalized least squares cost.
def ridge_regression(X, y, alpha=1.0):
    m = X.shape[1]  # Number of features including bias
    I = np.eye(m)   # Identity matrix for L2 regularization
    I[0, 0] = 0     # Do not regularize the intercept (bias term)
    theta = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y
    return theta


# Explore how varying alpha affects model coefficients.
alphas = [0, 0.1, 1, 10, 100]  # Regularization strengths
coefs = []

# For each alpha, compute the Ridge Regression coefficients and store them.
for alpha in alphas:
    theta = ridge_regression(X_b, y, alpha)
    coefs.append(theta.ravel())  # Flatten for easier plotting

# Plot coefficient shrinkage behavior as regularization strength increases.
plt.figure(figsize=(10, 6))
plt.plot(alphas, coefs)
plt.xscale('log')  # Use log scale for alpha to better visualize changes
plt.xlabel('Alpha (log scale)')
plt.ylabel('Coefficient value')
plt.title('Ridge coefficients as a function of regulation')
plt.legend(['Intercept', 'Slope'])  # Label each line
plt.show()


# Predict using Ridge Regression for two new data points: X = 0 and X = 2.
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]  # Add bias term

# Fit the model using alpha = 1 and predict the target values
theta = ridge_regression(X_b, y, alpha=1)
y_predict = X_new_b @ theta


# Compare the coefficients from standard Linear Regression (alpha=0)
# and Ridge Regression (alpha=1) to highlight the effect of regularization.
theta_linear = ridge_regression(X_b, y, alpha=0)
theta_ridge = ridge_regression(X_b, y, alpha=1)

print("Linear regression coefficients:", theta_linear.ravel())
print("Ridge regression coefficients:", theta_ridge.ravel())
