import numpy as np

# In this example, I demonstrate how to perform linear regression using the Normal Equation.
# The goal is to predict house prices based on square footage (a single input feature).

# Define the feature matrix `X` representing the area (in square feet) of five properties.
X = np.array([[500], [1200], [1500], [1800], [2100]])

# Define the target vector `y` representing the corresponding house prices.
y = np.array([150000, 220000, 350000, 420000, 500000])

# Display the raw input features and target values.
print("Feature Matrix X: \n", X)
print("Target vector y: \n", y)

# To include the intercept (bias) term in the regression model, I prepend a column of ones to X.
# This creates the augmented feature matrix `X_b` of shape (m, 2), where m is the number of samples.
X_b = np.c_[np.ones((X.shape[0], 1)), X]

print("X with bias term: \n", X_b)

# Compute the model parameters θ (theta) using the Normal Equation:
# θ = (XᵀX)^(-1) Xᵀy
# This closed-form solution minimizes the mean squared error between predictions and actual values.
theta = np.linalg.solve(X_b.T.dot(X_b), X_b.T.dot(y))

# Display the learned parameters.
# The first value represents the intercept; the second represents the coefficient for square footage.
print("Computed Parameters (θ):\n", theta)

# Define a function to make predictions using the learned model.
# It accepts a new area value and the parameter vector θ, and returns the predicted price.
def predict(area, theta):
    # Augment the new input value with a bias term before applying the linear model.
    x_with_bias = np.c_[np.ones((1, 1)), np.array([[area]])]
    return x_with_bias.dot(theta)

# Predict the price of a house with 1000 square feet using the trained model.
predicted_price = predict(1000, theta)
