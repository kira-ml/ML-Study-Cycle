import numpy as np
import matplotlib.pyplot as plt

# --- Data Preparation ---
# Feature matrix: Square footage of 5 houses (1 feature per house)
X = np.array([[500], [1200], [1500], [1800], [2100]])
# Target vector: Corresponding house prices (regression targets)
y = np.array([150000, 220000, 350000, 420000, 500000])

print("📊 Feature Matrix X (sq.ft): \n", X)
print("💰 Target vector y (prices): \n", y)

# Add bias term (intercept column of 1's) to X
# This enables the model equation: price = θ₀*1 + θ₁*sq_ft
X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Shape: (5, 2)
print("➕ X with bias term (column of 1's added): \n", X_b)

# --- Normal Equation Solution ---
# Closed-form solution: θ = (XᵀX)⁻¹ Xᵀy
# No iterative training needed - directly computes optimal parameters
# Note: In practice, use np.linalg.pinv() for better numerical stability
theta = np.linalg.solve(X_b.T.dot(X_b), X_b.T.dot(y))  # θ = [intercept, slope]
print("\n✅ Computed Parameters (θ):")
print(f"   θ₀ (intercept): {theta[0]:,.0f}")
print(f"   θ₁ (slope): {theta[1]:,.2f}")
print(f"   Model: Price = {theta[0]:,.0f} + {theta[1]:,.2f} * sq_ft")

# --- Prediction Function ---
def predict(area, theta):
    """
    Predict house price for given square footage using linear model.
    
    Args:
        area (scalar or array): Square footage to predict price for
        theta (array): Model parameters [intercept, slope]
        
    Returns:
        Predicted price(s) based on linear equation
    """
    # Format input, add bias term, compute dot product with parameters
    x_with_bias = np.c_[np.ones((1, 1)), np.array([[area]])]
    return x_with_bias.dot(theta)  # = θ₀ + θ₁*area

# Example prediction
area_to_predict = 1000
predicted_price = predict(area_to_predict, theta)
print(f"\n🎯 Predicted price for {area_to_predict} sq.ft: ${predicted_price[0]:,.0f}")

# --- Visualization ---
plt.figure(figsize=(9, 6))

# Scatter plot: Actual data points
plt.scatter(X, y, color="navy", label="Actual Data", s=80, alpha=0.8, edgecolors='black')

# Regression line: Generate predictions across x-range
# Create 100 evenly spaced points between min and max X
x_line = np.linspace(X.min() - 200, X.max() + 200, 100).reshape(-1, 1)
# Add bias column and compute y = Xθ
x_line_with_bias = np.c_[np.ones((x_line.shape[0], 1)), x_line]
y_line = x_line_with_bias.dot(theta)
plt.plot(x_line, y_line, color="crimson", linewidth=2.5, 
         label=f"Regression Line (slope={theta[1]:,.2f})")

# Highlight prediction point
plt.scatter(area_to_predict, predicted_price, color="limegreen", 
            s=150, marker="X", linewidths=2, edgecolors='darkgreen',
            label=f"Prediction ({area_to_predict} sq.ft)")
plt.annotate(f"${predicted_price[0]:,.0f}", 
             (area_to_predict, predicted_price + 10000),
             ha='center', fontsize=10)

# Display model equation
equation_text = f"Price = {theta[0]:,.0f} + {theta[1]:,.2f}·sq_ft"
plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes,
         fontsize=11, bbox=dict(boxstyle="round,pad=0.5", 
         facecolor="lightyellow", alpha=0.9))

# Labels and formatting
plt.xlabel("Square Footage", fontsize=12)
plt.ylabel("House Price ($)", fontsize=12)
plt.title("Linear Regression: House Price vs. Square Footage\n(Normal Equation Method)", 
          fontsize=14, weight='bold')
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

# Show plot
plt.show()

# --- Model Evaluation (Bonus) ---
# Calculate predictions for training data
y_pred = X_b.dot(theta)
# Mean Squared Error (MSE) - lower is better
mse = np.mean((y - y_pred) ** 2)
print(f"\n📈 Model Evaluation:")
print(f"   Mean Squared Error (MSE): {mse:,.2f}")
print(f"   Root Mean Squared Error (RMSE): ${np.sqrt(mse):,.0f}")