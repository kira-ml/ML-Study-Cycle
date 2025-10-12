import numpy as np
import matplotlib.pyplot as plt


# Set random seed for reproducible results across runs
# This ensures that random operations (like data generation and shuffling) 
# produce the same sequence of numbers each time the script is executed
np.random.seed(42)


def generate_separable_data(n_samples=1000):
    """
    Generate linearly separable data for binary classification.
    
    This function creates two classes of data points that can be perfectly
    separated by a straight line, making them ideal for testing optimization algorithms
    on problems where a clear decision boundary exists.
    
    Args:
        n_samples (int): Total number of samples to generate (divided equally between classes)
        
    Returns:
        X (numpy.ndarray): Feature matrix with shape (n_samples, 3) - includes bias term
        y (numpy.ndarray): Binary labels with shape (n_samples,) - values are 0 or 1
    """
    # Generate class 0: samples centered at [-1, 1] with identity covariance
    # Identity covariance means features are uncorrelated with unit variance
    class_0 = np.random.multivariate_normal(
        mean=[-1, 1], cov=[[1, 0], [0, 1]], size=n_samples//2
    )
    
    # Generate class 1: samples centered at [1, -1] with identity covariance
    # These centers create a diagonal decision boundary in 2D space
    class_1 = np.random.multivariate_normal(
        mean=[1, -1], cov=[[1, 0], [0, 1]], size=n_samples//2
    )
    
    # Combine both classes into single feature matrix
    X = np.vstack([class_0, class_1])
    
    # Create corresponding binary labels (0 for class_0, 1 for class_1)
    y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
    
    # Add bias term (column of ones) to enable intercept in linear model
    # This allows the decision boundary to not pass through origin
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    
    return X, y


def generate_near_separable_data(n_samples=1000, noise_level=0.3):
    """
    Generate nearly linearly separable data with overlapping regions.
    
    This function creates data where classes are not perfectly separable,
    simulating real-world scenarios where some misclassification is expected.
    The overlapping regions make optimization more challenging.
    
    Args:
        n_samples (int): Total number of samples to generate (divided equally between classes)
        noise_level (float): Controls the degree of overlap between classes
        
    Returns:
        X (numpy.ndarray): Feature matrix with shape (n_samples, 3) - includes bias term
        y (numpy.ndarray): Binary labels with shape (n_samples,) - values are 0 or 1
    """
    # Generate class 0: samples centered at [-0.5, -0.5] with correlated features
    # Off-diagonal covariance elements (0.2) introduce correlation between features
    class_0 = np.random.multivariate_normal(
        mean=[-0.5, -0.5], cov=[[1.5, 0.2], [0.2, 1.5]], size=n_samples//2
    )
    
    # Generate class 1: samples centered at [0.5, 0.5] with correlated features
    # Centers are closer together than in separable case, creating overlap
    class_1 = np.random.multivariate_normal(
        mean=[0.5, 0.5], cov=[[1.5, 0.2], [0.2, 1.5]], size=n_samples//2
    )
    
    # Combine both classes into single feature matrix
    X = np.vstack([class_0, class_1])
    
    # Create corresponding binary labels (0 for class_0, 1 for class_1)
    y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
    
    # Add bias term (column of ones) to enable intercept in linear model
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    
    return X, y


def sigmoid(z):
    """
    Compute sigmoid function with numerical stability.
    
    The sigmoid function maps any real number to the range (0, 1), making it
    ideal for binary classification probability outputs. Numerical clipping
    prevents overflow in the exponential function for extreme input values.
    
    Args:
        z (numpy.ndarray): Input values to apply sigmoid to
        
    Returns:
        numpy.ndarray: Sigmoid-transformed values in range (0, 1)
    """
    # Clip input values to prevent numerical overflow in exp(-z)
    # Values beyond [-250, 250] would cause overflow in exp() function
    z_clipped = np.clip(z, -250, 250)
    
    # Compute sigmoid: 1 / (1 + e^(-z))
    # This squashes values to (0, 1) range for probability interpretation
    return 1 / (1 + np.exp(-z_clipped))


def predict_proba(X, weights):
    """
    Predict class probabilities using the sigmoid function.
    
    This function computes the probability that each sample belongs to class 1
    using the logistic regression model: P(y=1|x) = sigmoid(x^T * weights)
    
    Args:
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features)
        weights (numpy.ndarray): Model weights of shape (n_features,)
        
    Returns:
        numpy.ndarray: Predicted probabilities for class 1, shape (n_samples,)
    """
    # Compute linear combination: X @ weights (dot product for each sample)
    # Then apply sigmoid to get probabilities in range (0, 1)
    return sigmoid(X @ weights)


def predict(X, weights, threshold=0.5):
    """
    Make binary predictions based on probability threshold.
    
    Converts probability outputs to binary class predictions using a threshold.
    Default threshold of 0.5 means class 1 is predicted when P(y=1|x) > 0.5
    
    Args:
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features)
        weights (numpy.ndarray): Model weights of shape (n_features,)
        threshold (float): Probability threshold for classification (default 0.5)
        
    Returns:
        numpy.ndarray: Binary predictions (0 or 1) of shape (n_samples,)
    """
    # First compute probabilities using sigmoid
    probabilities = predict_proba(X, weights)
    
    # Convert to binary predictions: 1 if prob >= threshold, else 0
    # astype(int) converts boolean array to integer array (True->1, False->0)
    return (probabilities >= threshold).astype(int)


def compute_loss(y_true, y_pred_proba):
    """
    Compute binary cross-entropy loss for logistic regression.
    
    Binary cross-entropy measures the difference between true labels and predicted
    probabilities. It's the appropriate loss function for binary classification
    with sigmoid outputs, providing convex optimization landscape.
    
    Args:
        y_true (numpy.ndarray): True binary labels (0 or 1)
        y_pred_proba (numpy.ndarray): Predicted probabilities for class 1
        
    Returns:
        float: Average cross-entropy loss across all samples
    """
    # Add small epsilon to prevent log(0) which would cause infinity
    # This ensures numerical stability when predictions are exactly 0 or 1
    epsilon = 1e-15
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
    
    # Compute binary cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
    # This penalizes confident wrong predictions more heavily than uncertain ones
    return -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))


def compute_gradient(X, y_true, weights):
    """
    Compute gradient of the loss function with respect to weights.
    
    The gradient indicates the direction of steepest increase in loss.
    For minimization, we move in the opposite direction (-gradient).
    This is the derivative of binary cross-entropy loss with respect to weights.
    
    Args:
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features)
        y_true (numpy.ndarray): True binary labels (0 or 1)
        weights (numpy.ndarray): Current model weights
        
    Returns:
        numpy.ndarray: Gradient vector of shape (n_features,)
    """
    # Compute predicted probabilities using current weights
    y_pred = predict_proba(X, weights)
    
    # Calculate prediction errors (predicted - true)
    # Positive error means we predicted too high, negative means too low
    error = y_pred - y_true
    
    # Compute gradient: X^T @ error / n_samples
    # This is the derivative of cross-entropy loss w.r.t. weights
    # Division by n_samples gives average gradient across batch
    gradient = X.T @ error / len(y_true)
    
    return gradient


def initialize_weights(n_features):
    """
    Initialize model weights to zeros.
    
    Starting with zeros is a common initialization strategy that provides
    a neutral starting point for optimization. For logistic regression,
    this means starting with no preference for either class.
    
    Args:
        n_features (int): Number of features including bias term
        
    Returns:
        numpy.ndarray: Zero-initialized weight vector of shape (n_features,)
    """
    # Initialize all weights to zero for symmetric starting point
    # This ensures no initial bias toward either class before training
    return np.zeros(n_features)


def sgd_vanilla(X, y, learning_rate=0.1, n_epochs=100, batch_size=32):
    """
    Vanilla Stochastic Gradient Descent implementation for logistic regression.
    
    This basic SGD algorithm updates weights using gradient computed on small
    batches of data, providing efficient optimization for large datasets.
    No momentum or adaptive learning rate adjustments are included.
    
    Args:
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features)
        y (numpy.ndarray): Binary labels of shape (n_samples,)
        learning_rate (float): Step size for weight updates (default 0.1)
        n_epochs (int): Number of complete passes through the dataset (default 100)
        batch_size (int): Number of samples per mini-batch (default 32)
        
    Returns:
        tuple: (final_weights, loss_history)
            - final_weights: Optimized weight vector
            - loss_history: List of average loss per epoch
    """
    # Extract dataset dimensions
    n_samples, n_features = X.shape
    
    # Initialize weights to zeros
    weights = initialize_weights(n_features)
    
    # Track loss values for monitoring convergence
    losses = []
    
    # Main training loop - iterate through multiple epochs
    for epoch in range(n_epochs):
        # Shuffle data indices to randomize batch order
        # This prevents learning artifacts from fixed data ordering
        indices = np.random.permutation(n_samples)
        X_shuffled, y_shuffled = X[indices], y[indices]
        
        # Track epoch-level statistics
        epoch_loss = 0
        batches = 0
        
        # Process data in mini-batches
        for i in range(0, n_samples, batch_size):
            # Extract current batch of features and labels
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Compute gradient on current batch
            gradient = compute_gradient(X_batch, y_batch, weights)
            
            # Update weights: move in opposite direction of gradient
            # learning_rate controls step size of updates
            weights = weights - learning_rate * gradient
            
            # Track batch loss for epoch averaging
            batch_loss = compute_loss(y_batch, predict_proba(X_batch, weights))
            epoch_loss += batch_loss
            batches += 1
        
        # Calculate average loss across all batches in this epoch
        avg_epoch_loss = epoch_loss / batches
        losses.append(avg_epoch_loss)
        
        # Print progress every 20 epochs to monitor training
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {avg_epoch_loss:.4f}")
    
    return weights, losses


def sgd_momentum(X, y, learning_rate=0.1, momentum=0.9, n_epochs=100, batch_size=32):
    """
    SGD with momentum implementation for faster convergence.
    
    Momentum accumulates past gradients to maintain direction and speed up
    optimization in consistent directions while dampening oscillations.
    This helps escape local minima and accelerates convergence.
    
    Args:
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features)
        y (numpy.ndarray): Binary labels of shape (n_samples,)
        learning_rate (float): Step size for updates (default 0.1)
        momentum (float): Momentum coefficient (0-1, default 0.9)
        n_epochs (int): Number of training epochs (default 100)
        batch_size (int): Size of mini-batches (default 32)
        
    Returns:
        tuple: (final_weights, loss_history)
            - final_weights: Optimized weight vector
            - loss_history: List of average loss per epoch
    """
    # Extract dataset dimensions
    n_samples, n_features = X.shape
    
    # Initialize weights to zeros
    weights = initialize_weights(n_features)
    
    # Initialize velocity (momentum accumulator) to zeros
    # This stores exponentially weighted average of past gradients
    velocity = np.zeros_like(weights)
    
    # Track loss values for monitoring convergence
    losses = []
    
    # Main training loop with momentum
    for epoch in range(n_epochs):
        # Shuffle data for each epoch to prevent ordering artifacts
        indices = np.random.permutation(n_samples)
        X_shuffled, y_shuffled = X[indices], y[indices]
        
        # Track epoch statistics
        epoch_loss = 0
        batches = 0
        
        # Process data in mini-batches
        for i in range(0, n_samples, batch_size):
            # Extract current batch
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Compute gradient on current batch
            gradient = compute_gradient(X_batch, y_batch, weights)
            
            # Update velocity: momentum * old_velocity - learning_rate * gradient
            # This accumulates past gradient information with exponential decay
            velocity = momentum * velocity - learning_rate * gradient
            
            # Update weights using velocity (momentum-accelerated updates)
            weights = weights + velocity
            
            # Track loss for monitoring
            batch_loss = compute_loss(y_batch, predict_proba(X_batch, weights))
            epoch_loss += batch_loss
            batches += 1
        
        # Calculate and store average epoch loss
        avg_epoch_loss = epoch_loss / batches
        losses.append(avg_epoch_loss)
        
        # Print progress every 20 epochs
        if epoch % 20 == 0:
            print(f"Momentum epoch {epoch}: Loss = {avg_epoch_loss:.4f}")
    
    return weights, losses


def sgd_nesterov(X, y, learning_rate=0.1, momentum=0.9, n_epochs=100, batch_size=32):
    """
    Nesterov Accelerated Gradient (NAG) implementation.
    
    Nesterov momentum looks ahead by computing the gradient at the
    momentum-adjusted position, providing better convergence properties
    than standard momentum by anticipating the next position.
    
    Args:
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features)
        y (numpy.ndarray): Binary labels of shape (n_samples,)
        learning_rate (float): Step size for updates (default 0.1)
        momentum (float): Momentum coefficient (0-1, default 0.9)
        n_epochs (int): Number of training epochs (default 100)
        batch_size (int): Size of mini-batches (default 32)
        
    Returns:
        tuple: (final_weights, loss_history)
            - final_weights: Optimized weight vector
            - loss_history: List of average loss per epoch
    """
    # Extract dataset dimensions
    n_samples, n_features = X.shape
    
    # Initialize weights to zeros
    weights = initialize_weights(n_features)
    
    # Initialize velocity accumulator to zeros
    velocity = np.zeros_like(weights)
    
    # Track loss values for monitoring convergence
    losses = []
    
    # Main training loop with Nesterov momentum
    for epoch in range(n_epochs):
        # Shuffle data for each epoch
        indices = np.random.permutation(n_samples)
        X_shuffled, y_shuffled = X[indices], y[indices]
        
        # Track epoch statistics
        epoch_loss = 0
        batches = 0
        
        # Process data in mini-batches
        for i in range(0, n_samples, batch_size):
            # Extract current batch
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Compute "look-ahead" weights: current + momentum * velocity
            # This anticipates where we'll be after momentum update
            lookahead_weights = weights + momentum * velocity
            
            # Compute gradient at look-ahead position (Nesterov insight)
            # This provides better gradient information than current position
            gradient = compute_gradient(X_batch, y_batch, lookahead_weights)
            
            # Update velocity: momentum * old_velocity - learning_rate * new_gradient
            velocity = momentum * velocity - learning_rate * gradient
            
            # Update weights using updated velocity
            weights = weights + velocity
            
            # Track loss for monitoring
            batch_loss = compute_loss(y_batch, predict_proba(X_batch, weights))
            epoch_loss += batch_loss
            batches += 1
        
        # Calculate and store average epoch loss
        avg_epoch_loss = epoch_loss / batches
        losses.append(avg_epoch_loss)
        
        # Print progress every 20 epochs
        if epoch % 20 == 0:
            print(f"Nesterov epoch {epoch}: Loss = {avg_epoch_loss:.4f}")
    
    return weights, losses


def compare_optimizers(X, y, dataset_name):
    """
    Compare performance of different optimization algorithms on the same dataset.
    
    This function trains three different optimizers (vanilla SGD, momentum SGD,
    and Nesterov) on the same data and returns their performance metrics.
    
    Args:
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features)
        y (numpy.ndarray): Binary labels of shape (n_samples,)
        dataset_name (str): Name of dataset for logging purposes
        
    Returns:
        tuple: All loss histories and final weights from the three optimizers
    """
    print(f"\n=== Training on {dataset_name} Data ===")

    print(f"\n1. Vanilla SGD:")
    # Train with basic SGD optimizer
    weights_vanilla, losses_vanilla = sgd_vanilla(X, y, learning_rate=0.1, n_epochs=100)

    print(f"\n2. SGD with momentum:")
    # Train with momentum-enhanced SGD optimizer
    weights_momentum, losses_momentum = sgd_momentum(X, y, learning_rate=0.1, momentum=0.9, n_epochs=100)

    print(f"\n3. Nesterov Momentum:")
    # Train with Nesterov accelerated gradient optimizer
    weights_nesterov, losses_nesterov = sgd_nesterov(X, y, learning_rate=0.1, momentum=0.9, n_epochs=100)

    return (losses_vanilla, losses_momentum, losses_nesterov, 
            weights_vanilla, weights_momentum, weights_nesterov)


def plot_results(losses_vanilla, losses_momentum, losses_nesterov, dataset_name):
    """
    Plot and compare loss curves for different optimization algorithms.
    
    Creates two subplots: one showing full training curves and another
    focusing on final convergence behavior to compare algorithm performance.
    
    Args:
        losses_vanilla (list): Loss history from vanilla SGD
        losses_momentum (list): Loss history from momentum SGD
        losses_nesterov (list): Loss history from Nesterov optimizer
        dataset_name (str): Name of dataset for plot titles
    """
    # Create figure with two subplots side by side
    plt.figure(figsize=(12, 4))

    # Left subplot: Full training curves
    plt.subplot(1, 2, 1)
    plt.plot(losses_vanilla, label='Vanilla SGD', alpha=0.7)
    plt.plot(losses_momentum, label='SGD + Momentum', alpha=0.7)
    plt.plot(losses_nesterov, label='Nesterov', alpha=0.7)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title(f'Loss curves - {dataset_name} Data')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Right subplot: Focus on final 50 epochs for convergence analysis
    start_idx = max(0, len(losses_vanilla) - 50)
    plt.subplot(1, 2, 2)
    plt.plot(range(start_idx, len(losses_vanilla)), losses_vanilla[start_idx:], label='Vanilla SGD', alpha=0.7)
    plt.plot(range(start_idx, len(losses_momentum)), losses_momentum[start_idx:], label='SGD + Momentum', alpha=0.7)
    plt.plot(range(start_idx, len(losses_nesterov)), losses_nesterov[start_idx:], label='Nesterov', alpha=0.7)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Final convergence (Last 50 epochs)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Adjust layout and display plots
    plt.tight_layout()
    plt.show()

    # Print final loss values for quantitative comparison
    print(f"\nFinal losses - {dataset_name}:")
    print(f"Vanilla SGD: {losses_vanilla[-1]:.6f}")
    print(f"SGD + Momentum: {losses_momentum[-1]:.6f}")
    print(f"Nesterov: {losses_nesterov[-1]:.6f}")


def evaluate_accuracy(X, y, weights_vanilla, weights_momentum, weights_nesterov, dataset_name):
    """
    Evaluate and compare classification accuracy of different trained models.
    
    Computes predictions using final weights from each optimizer and compares
    their classification performance on the given dataset.
    
    Args:
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features)
        y (numpy.ndarray): True binary labels of shape (n_samples,)
        weights_vanilla (numpy.ndarray): Trained weights from vanilla SGD
        weights_momentum (numpy.ndarray): Trained weights from momentum SGD
        weights_nesterov (numpy.ndarray): Trained weights from Nesterov optimizer
        dataset_name (str): Name of dataset for logging
        
    Returns:
        tuple: Accuracy values for each optimizer (vanilla, momentum, nesterov)
    """
    # Make binary predictions using each set of trained weights
    y_pred_vanilla = predict(X, weights_vanilla)
    y_pred_momentum = predict(X, weights_momentum)
    y_pred_nesterov = predict(X, weights_nesterov)

    # Calculate accuracy as percentage of correct predictions
    acc_vanilla = np.mean(y_pred_vanilla == y)
    acc_momentum = np.mean(y_pred_momentum == y)
    acc_nesterov = np.mean(y_pred_nesterov == y)

    # Print accuracy comparison results
    print(f"\n=== Final accuracy - {dataset_name} Data ===")
    print(f"Vanilla SGD: {acc_vanilla:.3f}")
    print(f"SGD + Momentum: {acc_momentum:.3f}")
    print(f"Nesterov: {acc_nesterov:.3f}")

    return acc_vanilla, acc_momentum, acc_nesterov


# Generate two different types of datasets for comprehensive testing
# Separable data: classes can be perfectly separated by a linear boundary
X_sep, y_sep = generate_separable_data(1000)

# Near-separable data: classes have overlapping regions, making classification harder
X_near_sep, y_near_sep = generate_near_separable_data(1000)

# Print dataset information to confirm successful generation
print(f"Separable data shape: X{X_sep.shape}, y{y_sep.shape}")
print(f"Near-separable data shape: X{X_near_sep.shape}, y{y_near_sep.shape}")
print("Both datasets generated successfully")

# Run optimization comparisons on both dataset types
# This allows us to see how different optimizers perform on both easy and hard problems
results_sep = compare_optimizers(X_sep, y_sep, "Separable")
results_near_sep = compare_optimizers(X_near_sep, y_near_sep, "Near-Separable")

# Plot loss curves for both datasets to visualize convergence patterns
plot_results(*results_sep[:3], "Separable")
plot_results(*results_near_sep[:3], "Near-Separable")

# Evaluate and compare final classification accuracy on both datasets
print("\n" + "="*60)
evaluate_accuracy(X_sep, y_sep, *results_sep[3:], "Separable")
evaluate_accuracy(X_near_sep, y_near_sep, *results_near_sep[3:], "Near-separable")