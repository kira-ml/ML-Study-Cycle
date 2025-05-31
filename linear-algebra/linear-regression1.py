import numpy as np


X = np.array ([1000, 1500, 2000, 2500, 3000])
y = np.array ([200000, 300000, 400000, 500000, 600000])


w = 0
b = 0

def predict(X, y, b):
    return w * X + b



def compute_cost(y_true, y_pred):
    m = len(y_true)
    return (1 / (2 * m)) * np.sum((y_pred - y_true) ** 2)



def gradient_descent(X, y, y_pred, w, b, learning_rate):
    m = len(y)
    dw = (1 / m) * np.dot(X, (y_pred -y))
    db = (1 / m) * np.sum(y_pred -y)



    w -= learning_rate * dw
    b -= learning_rate * db

    return w, b



learning_rate = 0.000001
epochs = 1000



for epoch in range(epochs):
    y_pred = predict(X, w, b)

    cost = compute_cost(y, y_pred)

    w, b = gradient_descent(X, y, y_pred, w, b, learning_rate)


    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Cost = {cost:.4f}, w = {w:.4f}, b = {b:.4f}")


final_prediction = predict(X, w, b)


print("Final Parameter:")
print(f"Slope (w): {w:.4f}")
print(f"Intercept (b): {b:.4f}")

print("\nPrediction vs Actual:")

for pred, actual in zip(final_prediction, y):
    print(f"Predicted: {pred:.2f}, Actual {actual}")