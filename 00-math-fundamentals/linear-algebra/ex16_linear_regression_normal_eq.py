import numpy as np

X = np.array([[500], [1200], [1500], [1800]])

y = np.array([150000, 220000, 350000, 420000, 500000])

print("Feature Matrix X: \n", X)
print("Target vector y: \n", y)

X_b = np.c_[np.ones((X.shape[0], 1)), X]

print("X with bias term: \n", X_b)

theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print("Computed Parameters (θ):\n", theta)


new_house_area = np.array([[1000]])
new_house_with_bias = np.c_[np.ones((1, 1)), new_house_area]
predicted_price = new_house_with_bias.dot(theta)

print("Predicted price for 1000 sqft house: $%.2f" % predicted_price[0])

print("Computed price for 1000 sqft house: $%.2f" % predicted_price[0])

