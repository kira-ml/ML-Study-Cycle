import numpy as np
import matplotlib.pyplot as plt

# --- Data ---
X = np.array([[500], [1200], [1500], [1800], [2100]])
y = np.array([150000, 220000, 350000, 420000, 500000])

print("Feature Matrix X: \n", X)
print("Target vector y: \n", y)

# Add bias term (intercept)
X_b = np.c_[np.ones((X.shape[0], 1)), X]
print("X with bias term: \n", X_b)

# Normal Equation solution
theta = np.linalg.solve(X_b.T.dot(X_b), X_b.T.dot(y))
print("Computed Parameters (θ):\n", theta)

# Prediction function
def predict(area, theta):
    x_with_bias = np.c_[np.ones((1, 1)), np.array([[area]])]
    return x_with_bias.dot(theta)

# Example prediction
predicted_price = predict(1000, theta)
print(f"Predicted price for 1000 sq. ft: {predicted_price[0]:,.0f}")

# --- Visualization ---
plt.figure(figsize=(8, 6))

# Scatter plot of actual data
plt.scatter(X, y, color="blue", label="Actual Data", s=80)

# Regression line
x_line = np.linspace(X.min() - 200, X.max() + 200, 100).reshape(-1, 1)
y_line = np.c_[np.ones((x_line.shape[0], 1)), x_line].dot(theta)
plt.plot(x_line, y_line, color="red", linewidth=2, label="Regression Line")

# Predicted point
plt.scatter(1000, predicted_price, color="green", s=120, marker="X", label=f"Prediction (1000 sq.ft)")

# Annotate theta values
plt.text(600, 480000, f"Intercept θ₀ = {theta[0]:,.0f}\nSlope θ₁ = {theta[1]:,.2f}", 
         fontsize=10, bbox=dict(facecolor="white", alpha=0.6))

# Labels & legend
plt.xlabel("Square Footage")
plt.ylabel("House Price ($)")
plt.title("Linear Regression using Normal Equation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
