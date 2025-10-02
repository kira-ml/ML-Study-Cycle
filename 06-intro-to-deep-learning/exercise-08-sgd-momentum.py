import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)


def generate_separable_data(n_samples=1000):
    """Generate linearly separable data."""
    class_0 = np.random.multivariate_normal(
        mean=[-1, 1], cov=[[1, 0], [0, 1]], size=n_samples//2
    )
    class_1 = np.random.multivariate_normal(
        mean=[1, -1], cov=[[1, 0], [0, 1]], size=n_samples//2
    )
    
    X = np.vstack([class_0, class_1])
    y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
    
    # Add bias term
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    
    return X, y


def generate_near_separable_data(n_samples=1000, noise_level=0.3):
    """Generate nearly linearly separable data."""
    class_0 = np.random.multivariate_normal(
        mean=[-0.5, -0.5], cov=[[1.5, 0.2], [0.2, 1.5]], size=n_samples//2
    )
    class_1 = np.random.multivariate_normal(
        mean=[0.5, 0.5], cov=[[1.5, 0.2], [0.2, 1.5]], size=n_samples//2
    )
    
    X = np.vstack([class_0, class_1])
    y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
    
    # Add bias term
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    
    return X, y


def sigmoid(z):
    """Compute sigmoid function with numerical stability."""
    z_clipped = np.clip(z, -250, 250)
    return 1 / (1 + np.exp(-z_clipped))


def predict_proba(X, weights):
    """Predict probabilities using sigmoid."""
    return sigmoid(X @ weights)


def predict(X, weights, threshold=0.5):
    """Make binary predictions."""
    return (predict_proba(X, weights) >= threshold).astype(int)


def compute_loss(y_true, y_pred_proba):
    """Compute binary cross-entropy loss."""
    epsilon = 1e-15
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))


def compute_gradient(X, y_true, weights):
    """Compute gradient of the loss function."""
    y_pred = predict_proba(X, weights)
    error = y_pred - y_true
    gradient = X.T @ error / len(y_true)
    return gradient


def initialize_weights(n_features):
    """Initialize weights to zeros."""
    return np.zeros(n_features)


def sgd_vanilla(X, y, learning_rate=0.1, n_epochs=100, batch_size=32):
    """Vanilla SGD implementation."""
    n_samples, n_features = X.shape
    weights = initialize_weights(n_features)
    losses = []
    
    for epoch in range(n_epochs):
        indices = np.random.permutation(n_samples)
        X_shuffled, y_shuffled = X[indices], y[indices]
        
        epoch_loss = 0
        batches = 0
        
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            gradient = compute_gradient(X_batch, y_batch, weights)
            weights = weights - learning_rate * gradient
            batch_loss = compute_loss(y_batch, predict_proba(X_batch, weights))
            epoch_loss += batch_loss
            batches += 1
        
        avg_epoch_loss = epoch_loss / batches
        losses.append(avg_epoch_loss)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {avg_epoch_loss:.4f}")
    
    return weights, losses


def sgd_momentum(X, y, learning_rate=0.1, momentum=0.9, n_epochs=100, batch_size=32):
    """SGD with momentum implementation."""
    n_samples, n_features = X.shape
    weights = initialize_weights(n_features)
    velocity = np.zeros_like(weights)
    losses = []
    
    for epoch in range(n_epochs):
        indices = np.random.permutation(n_samples)
        X_shuffled, y_shuffled = X[indices], y[indices]
        
        epoch_loss = 0
        batches = 0
        
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            gradient = compute_gradient(X_batch, y_batch, weights)
            velocity = momentum * velocity - learning_rate * gradient
            weights = weights + velocity
            
            batch_loss = compute_loss(y_batch, predict_proba(X_batch, weights))
            epoch_loss += batch_loss
            batches += 1
        
        avg_epoch_loss = epoch_loss / batches
        losses.append(avg_epoch_loss)
        
        if epoch % 20 == 0:
            print(f"Momentum epoch {epoch}: Loss = {avg_epoch_loss:.4f}")
    
    return weights, losses


def sgd_nesterov(X, y, learning_rate=0.1, momentum=0.9, n_epochs=100, batch_size=32):
    """Nesterov accelerated gradient implementation."""
    n_samples, n_features = X.shape
    weights = initialize_weights(n_features)
    velocity = np.zeros_like(weights)
    losses = []
    
    for epoch in range(n_epochs):
        indices = np.random.permutation(n_samples)
        X_shuffled, y_shuffled = X[indices], y[indices]
        
        epoch_loss = 0
        batches = 0
        
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            lookahead_weights = weights + momentum * velocity
            gradient = compute_gradient(X_batch, y_batch, lookahead_weights)
            
            velocity = momentum * velocity - learning_rate * gradient
            weights = weights + velocity
            
            batch_loss = compute_loss(y_batch, predict_proba(X_batch, weights))
            epoch_loss += batch_loss
            batches += 1
        
        avg_epoch_loss = epoch_loss / batches
        losses.append(avg_epoch_loss)
        
        if epoch % 20 == 0:
            print(f"Nesterov epoch {epoch}: Loss = {avg_epoch_loss:.4f}")
    
    return weights, losses


def compare_optimizers(X, y, dataset_name):
    print(f"\n=== Training on {dataset_name} Data ===")

    print(f"\n1. Vanilla SGD:")
    weights_vanilla, losses_vanilla = sgd_vanilla(X, y, learning_rate=0.1, n_epochs=100)

    print(f"\n2. SGD with momentum:")
    weights_momentum, losses_momentum = sgd_momentum(X, y, learning_rate=0.1, momentum=0.9, n_epochs=100)

    print(f"\n3. Nesterov Momentum:")
    weights_nesterov, losses_nesterov = sgd_nesterov(X, y, learning_rate=0.1, momentum=0.9, n_epochs=100)

    return (losses_vanilla, losses_momentum, losses_nesterov, weights_vanilla, weights_momentum, weights_nesterov)


def plot_results(losses_vanilla, losses_momentum, losses_nesterov, dataset_name):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(losses_vanilla, label='Vanilla SGD', alpha=0.7)
    plt.plot(losses_momentum, label='SGD + Momentum', alpha=0.7)
    plt.plot(losses_nesterov, label='Nesterov', alpha=0.7)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title(f'Loss curves - {dataset_name} Data')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    start_idx = max(0, len(losses_vanilla) - 50)
    plt.plot(range(start_idx, len(losses_vanilla)), losses_vanilla[start_idx:], label='Vanilla SGD', alpha=0.7)
    plt.plot(range(start_idx, len(losses_momentum)), losses_momentum[start_idx:], label='SGD + Momentum', alpha=0.7)
    plt.plot(range(start_idx, len(losses_nesterov)), losses_nesterov[start_idx:], label='Nesterov', alpha=0.7)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Final convergence (Last 50 epochs)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\nFinal losses - {dataset_name}:")
    print(f"Vanilla SGD: {losses_vanilla[-1]:.6f}")
    print(f"SGD + Momentum: {losses_momentum[-1]:.6f}")
    print(f"Nesterov: {losses_nesterov[-1]:.6f}")


def evaluate_accuracy(X, y, weights_vanilla, weights_momentum, weights_nesterov, dataset_name):
    y_pred_vanilla = predict(X, weights_vanilla)
    y_pred_momentum = predict(X, weights_momentum)
    y_pred_nesterov = predict(X, weights_nesterov)

    acc_vanilla = np.mean(y_pred_vanilla == y)
    acc_momentum = np.mean(y_pred_momentum == y)
    acc_nesterov = np.mean(y_pred_nesterov == y)

    print(f"\n=== Final accuracy - {dataset_name} Data ===")
    print(f"Vanilla SGD: {acc_vanilla:.3f}")
    print(f"SGD + Momentum: {acc_momentum:.3f}")
    print(f"Nesterov: {acc_nesterov:.3f}")

    return acc_vanilla, acc_momentum, acc_nesterov


# Generate datasets first
X_sep, y_sep = generate_separable_data(1000)
X_near_sep, y_near_sep = generate_near_separable_data(1000)

print(f"Separable data shape: X{X_sep.shape}, y{y_sep.shape}")
print(f"Near-separable data shape: X{X_near_sep.shape}, y{y_near_sep.shape}")
print("Both datasets generated successfully")

# Run comparisons
results_sep = compare_optimizers(X_sep, y_sep, "Separable")
results_near_sep = compare_optimizers(X_near_sep, y_near_sep, "Near-Separable")

# Plot results
plot_results(*results_sep[:3], "Separable")
plot_results(*results_near_sep[:3], "Near-Separable")

# Evaluate accuracy
print("\n" + "="*60)
evaluate_accuracy(X_sep, y_sep, *results_sep[3:], "Separable")
evaluate_accuracy(X_near_sep, y_near_sep, *results_near_sep[3:], "Near-separable")