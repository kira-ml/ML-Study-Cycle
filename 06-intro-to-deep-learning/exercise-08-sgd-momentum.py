"""
Gradient Descent Algorithms: A Visual Journey from Vanilla to Nesterov
========================================================================
Author: kira-ml (Machine Learning Student)
Date: 2024
Purpose: Understanding optimization through implementation

Welcome to my learning journey! As a fellow ML student, I created this
to understand gradient descent variants through hands-on coding. Let's
learn together how optimizers work under the hood!
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# SECTION 1: Understanding Random Seeds - The "Pause Button" for Randomness
# ---------------------------------------------------------------------

# Why set random seed? Think of it as a "pause button" for randomness!
# Without it, every time you run the code, you get different results.
# With it, you get the SAME random numbers, making debugging possible!
np.random.seed(42)  # 42 is the Answer to the Ultimate Question of Life


# ---------------------------------------------------------------------
# SECTION 2: Creating Data - Why Synthetics First?
# ---------------------------------------------------------------------

def generate_separable_data(n_samples=1000):
    """
    Create perfectly separable data - The "Easy Test" for our models.
    
    Student Analogy: Imagine teaching a child to separate red and blue marbles.
    This data is like red marbles on the left, blue marbles on the right - easy!
    
    Visualize: Two clear blobs far apart - a straight line can perfectly separate them.
    
    Key Concepts:
    - Multivariate normal: Fancy way of saying "data blob with Gaussian shape"
    - Identity covariance: Features are independent (no correlation)
    - Bias term: The +c in y = mx + c - allows line to not pass through origin
    """
    # Class 0: Red marbles centered at (-1, 1)
    class_0 = np.random.multivariate_normal(
        mean=[-1, 1],  # Center point of the blob
        cov=[[1, 0], [0, 1]],  # Spread: 1 unit in all directions, no tilt
        size=n_samples//2  # Half the marbles
    )
    
    # Class 1: Blue marbles centered at (1, -1) - far from red ones
    class_1 = np.random.multivariate_normal(
        mean=[1, -1],  # Opposite quadrant from class 0
        cov=[[1, 0], [0, 1]],  # Same spread
        size=n_samples//2
    )
    
    # Stack them: Put all marbles on the table
    X = np.vstack([class_0, class_1])
    
    # Labels: Red=0, Blue=1
    y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
    
    # Add bias: The "starting point" for our line
    # Without this, line MUST pass through origin (0,0) - limiting!
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    
    return X, y


def generate_near_separable_data(n_samples=1000, noise_level=0.3):
    """
    Create messy, overlapping data - The "Real World Test".
    
    Student Analogy: Now the red and blue marbles are mixed together!
    Some reds are in blue territory and vice versa - just like real data.
    
    Visualize: Two blobs that overlap in the middle - no perfect line exists.
    
    Why this matters: Real-world data is NEVER perfectly separable.
    Our models need to handle ambiguity and make the "best guess".
    """
    # Notice the means are closer together now! Overlap guaranteed.
    class_0 = np.random.multivariate_normal(
        mean=[-0.5, -0.5],  # Closer to center
        cov=[[1.5, 0.2], [0.2, 1.5]],  # Larger spread + correlation (tilted blob)
        size=n_samples//2
    )
    
    class_1 = np.random.multivariate_normal(
        mean=[0.5, 0.5],  # Also closer to center
        cov=[[1.5, 0.2], [0.2, 1.5]],  # Same tilted, spread-out shape
        size=n_samples//2
    )
    
    # Stack and label as before
    X = np.vstack([class_0, class_1])
    y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
    
    # Don't forget the bias term!
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    
    return X, y


# ---------------------------------------------------------------------
# SECTION 3: The Magic Function - Sigmoid
# ---------------------------------------------------------------------

def sigmoid(z):
    """
    The "Squishifier" - Turns any number into a probability between 0 and 1.
    
    Student Analogy: Imagine a thermometer that shows probability instead of temperature.
    -∞ (very cold) → 0% chance
    0 (room temp) → 50% chance  
    +∞ (very hot) → 100% chance
    
    Why clip? Computers can't handle ∞! exp(1000) crashes.
    Clipping at ±250 keeps things safe.
    """
    # Safety first! Don't let z get too extreme
    z_clipped = np.clip(z, -250, 250)
    
    # The magic formula: 1 / (1 + e^(-z))
    # For large positive z: e^(-big) ≈ 0 → 1/(1+0) ≈ 1
    # For large negative z: e^(-negative) = e^big ≈ ∞ → 1/∞ ≈ 0
    return 1 / (1 + np.exp(-z_clipped))


# ---------------------------------------------------------------------
# SECTION 4: Making Predictions - From Weights to Guesses
# ---------------------------------------------------------------------

def predict_proba(X, weights):
    """
    Turn feature math into probabilities.
    
    Math: probability = sigmoid(dot_product(features, weights))
    
    Student Thinking: Each feature "votes" on the outcome.
    Weights decide how much each vote counts.
    Sigmoid converts the total vote count into a probability.
    """
    return sigmoid(X @ weights)  # @ means matrix multiplication


def predict(X, weights, threshold=0.5):
    """
    Make a yes/no decision from probabilities.
    
    Student Decision Making:
    - If model says >50% chance of rain → take umbrella
    - If model says ≤50% chance → leave umbrella
    
    Threshold tuning: For cancer detection, we might use 0.1 (be more cautious!)
    """
    probabilities = predict_proba(X, weights)
    return (probabilities >= threshold).astype(int)


# ---------------------------------------------------------------------
# SECTION 5: The Report Card - Measuring Mistakes (Loss Function)
# ---------------------------------------------------------------------

def compute_loss(y_true, y_pred_proba):
    """
    Cross-Entropy Loss: How "surprised" we are by wrong predictions.
    
    Student Analogy: Grading your predictions:
    - Predict 90% rain and it rains: Good job! (-log(0.9) = 0.1 small penalty)
    - Predict 90% rain and it's sunny: Bad! (-log(0.1) = 2.3 big penalty)
    - Predict 50% either way: Meh (-log(0.5) = 0.69 medium penalty)
    
    Why clip? log(0) = -∞ (computer hates this!)
    """
    epsilon = 1e-15  # Tiny number, but not zero!
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
    
    # The loss formula: average surprise across all predictions
    loss = -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
    return loss


# ---------------------------------------------------------------------
# SECTION 6: Finding the Right Direction - The Gradient
# ---------------------------------------------------------------------

def compute_gradient(X, y_true, weights):
    """
    Gradient: Which way to move to improve the most?
    
    Physical Analogy: You're on a foggy hill (loss landscape).
    Gradient tells you the steepest downhill direction.
    
    Math: Gradient = average(error × features)
    - Error = prediction - truth (how wrong we were)
    - Multiply by features: which features contributed to error?
    """
    y_pred = predict_proba(X, weights)
    error = y_pred - y_true  # Positive = predicted too high
    gradient = X.T @ error / len(y_true)  # Average across all samples
    return gradient


def initialize_weights(n_features):
    """
    Start from zero - a blank slate.
    
    Student Thinking: Before any learning, we have no preference.
    All features start with zero importance.
    The bias (last weight) starts at zero too.
    """
    return np.zeros(n_features)


# ---------------------------------------------------------------------
# SECTION 7: The Optimizers - Different Ways to Walk Downhill
# ---------------------------------------------------------------------

def sgd_vanilla(X, y, learning_rate=0.1, n_epochs=100, batch_size=32):
    """
    Vanilla SGD: Take small steps in the steepest downhill direction.
    
    Analogy: Walking down a hill while constantly checking your feet.
    - Look at current slope (gradient)
    - Take step opposite to slope
    - Repeat
    
    Pros: Simple, works
    Cons: Can zig-zag in valleys, slow on flat areas
    """
    n_samples, n_features = X.shape
    weights = initialize_weights(n_features)
    losses = []
    
    for epoch in range(n_epochs):
        # Shuffle: Like shuffling flashcards - prevents memorizing order
        indices = np.random.permutation(n_samples)
        X_shuffled, y_shuffled = X[indices], y[indices]
        
        epoch_loss = 0
        batches = 0
        
        # Mini-batches: Study in chunks, not all at once!
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Find downhill direction
            gradient = compute_gradient(X_batch, y_batch, weights)
            
            # Take step: weights = weights - learning_rate × gradient
            weights = weights - learning_rate * gradient
            
            # Check how well we're doing
            batch_loss = compute_loss(y_batch, predict_proba(X_batch, weights))
            epoch_loss += batch_loss
            batches += 1
        
        avg_loss = epoch_loss / batches
        losses.append(avg_loss)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    return weights, losses


def sgd_momentum(X, y, learning_rate=0.1, momentum=0.9, n_epochs=100, batch_size=32):
    """
    SGD with Momentum: Like a ball rolling downhill.
    
    Analogy: Skiing down a slope - you gain speed in consistent directions!
    - Velocity remembers past gradients
    - Builds speed in good directions
    - Dampens oscillations
    
    Physics: velocity = momentum × old_velocity - learning_rate × gradient
             weights = weights + velocity
    """
    n_samples, n_features = X.shape
    weights = initialize_weights(n_features)
    velocity = np.zeros_like(weights)  # Start with zero velocity
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
            
            # Update velocity (accumulate past gradients)
            velocity = momentum * velocity - learning_rate * gradient
            
            # Update weights using velocity (not just gradient!)
            weights = weights + velocity
            
            batch_loss = compute_loss(y_batch, predict_proba(X_batch, weights))
            epoch_loss += batch_loss
            batches += 1
        
        avg_loss = epoch_loss / batches
        losses.append(avg_loss)
        
        if epoch % 20 == 0:
            print(f"Momentum epoch {epoch}: Loss = {avg_loss:.4f}")
    
    return weights, losses


def sgd_nesterov(X, y, learning_rate=0.1, momentum=0.9, n_epochs=100, batch_size=32):
    """
    Nesterov Momentum: The "Look Ahead" Ball.
    
    Analogy: A smart ball that looks WHERE IT WILL BE before deciding.
    - First: Estimate future position (look ahead)
    - Then: Calculate gradient THERE
    - Finally: Adjust course
    
    Why better? More accurate gradient estimate at the destination.
    """
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
            
            # Nesterov's key insight: look ahead!
            lookahead_weights = weights + momentum * velocity
            
            # Calculate gradient at the lookahead position
            gradient = compute_gradient(X_batch, y_batch, lookahead_weights)
            
            # Update velocity and weights
            velocity = momentum * velocity - learning_rate * gradient
            weights = weights + velocity
            
            batch_loss = compute_loss(y_batch, predict_proba(X_batch, weights))
            epoch_loss += batch_loss
            batches += 1
        
        avg_loss = epoch_loss / batches
        losses.append(avg_loss)
        
        if epoch % 20 == 0:
            print(f"Nesterov epoch {epoch}: Loss = {avg_loss:.4f}")
    
    return weights, losses


# ---------------------------------------------------------------------
# SECTION 8: The Experiment - Let's Compare!
# ---------------------------------------------------------------------

def compare_optimizers(X, y, dataset_name):
    """
    Run all three optimizers and see who wins!
    
    Student Experiment: Like racing three different downhill methods:
    1. Careful walker (Vanilla)
    2. Skier with momentum
    3. Smart skier who looks ahead (Nesterov)
    """
    print(f"\n{'='*60}")
    print(f"Testing on {dataset_name} Data")
    print('='*60)
    
    print("\n1. Vanilla SGD (The Careful Walker):")
    weights_vanilla, losses_vanilla = sgd_vanilla(X, y, learning_rate=0.1, n_epochs=100)
    
    print("\n2. SGD with Momentum (The Skier):")
    weights_momentum, losses_momentum = sgd_momentum(X, y, learning_rate=0.1, momentum=0.9, n_epochs=100)
    
    print("\n3. Nesterov Momentum (The Smart Skier):")
    weights_nesterov, losses_nesterov = sgd_nesterov(X, y, learning_rate=0.1, momentum=0.9, n_epochs=100)
    
    return (losses_vanilla, losses_momentum, losses_nesterov, 
            weights_vanilla, weights_momentum, weights_nesterov)


def plot_results(losses_vanilla, losses_momentum, losses_nesterov, dataset_name):
    """
    Visualize the race results!
    
    Two plots:
    1. Full race: See who started fast, who finished well
    2. Final sprint: Zoom in on the last 50 epochs
    """
    plt.figure(figsize=(12, 4))
    
    # Plot 1: The full race
    plt.subplot(1, 2, 1)
    plt.plot(losses_vanilla, label='Vanilla SGD', alpha=0.7, linewidth=2)
    plt.plot(losses_momentum, label='SGD + Momentum', alpha=0.7, linewidth=2)
    plt.plot(losses_nesterov, label='Nesterov', alpha=0.7, linewidth=2)
    plt.xlabel('Epochs (Training Time)')
    plt.ylabel('Loss (How Wrong We Are)')
    plt.title(f'Optimizer Race: {dataset_name} Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: The final sprint (last 50 epochs)
    plt.subplot(1, 2, 2)
    start_idx = max(0, len(losses_vanilla) - 50)
    
    # Create x-axis labels for the zoomed plot
    epochs_range = range(start_idx, len(losses_vanilla))
    
    plt.plot(epochs_range, losses_vanilla[start_idx:], label='Vanilla SGD', alpha=0.7, linewidth=2)
    plt.plot(epochs_range, losses_momentum[start_idx:], label='SGD + Momentum', alpha=0.7, linewidth=2)
    plt.plot(epochs_range, losses_nesterov[start_idx:], label='Nesterov', alpha=0.7, linewidth=2)
    plt.xlabel('Epochs (Last 50)')
    plt.ylabel('Loss')
    plt.title('Final Convergence - Who Got Lowest?')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print final scores
    print(f"\nFinal Loss Scores - {dataset_name}:")
    print(f"🏃 Vanilla SGD: {losses_vanilla[-1]:.6f}")
    print(f"⛷️  SGD + Momentum: {losses_momentum[-1]:.6f}")
    print(f"🔮 Nesterov: {losses_nesterov[-1]:.6f}")


def evaluate_accuracy(X, y, weights_vanilla, weights_momentum, weights_nesterov, dataset_name):
    """
    The Final Exam: How well did each optimizer actually learn?
    
    Accuracy = Percentage of correct predictions
    """
    # Make predictions with each set of learned weights
    y_pred_vanilla = predict(X, weights_vanilla)
    y_pred_momentum = predict(X, weights_momentum)
    y_pred_nesterov = predict(X, weights_nesterov)
    
    # Calculate accuracy
    acc_vanilla = np.mean(y_pred_vanilla == y)
    acc_momentum = np.mean(y_pred_momentum == y)
    acc_nesterov = np.mean(y_pred_nesterov == y)
    
    print(f"\n📊 Final Accuracy - {dataset_name} Data:")
    print(f"🏃 Vanilla SGD: {acc_vanilla:.3f} ({acc_vanilla*100:.1f}% correct)")
    print(f"⛷️  SGD + Momentum: {acc_momentum:.3f} ({acc_momentum*100:.1f}% correct)")
    print(f"🔮 Nesterov: {acc_nesterov:.3f} ({acc_nesterov*100:.1f}% correct)")
    
    return acc_vanilla, acc_momentum, acc_nesterov


# ---------------------------------------------------------------------
# SECTION 9: Let's Run the Experiment!
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("🎓 Gradient Descent Learning Lab - by kira-ml")
    print("="*60)
    
    # Generate our test datasets
    print("\n📊 Generating datasets...")
    X_sep, y_sep = generate_separable_data(1000)  # Easy test
    X_near_sep, y_near_sep = generate_near_separable_data(1000)  # Hard test
    
    print(f"✅ Easy data (separable): {X_sep.shape[0]} samples, {X_sep.shape[1]-1} features + bias")
    print(f"✅ Hard data (near-separable): {X_near_sep.shape[0]} samples, {X_near_sep.shape[1]-1} features + bias")
    
    # Run experiments on both datasets
    print("\n" + "="*60)
    print("🚀 STARTING OPTIMIZATION EXPERIMENTS")
    print("="*60)
    
    # Test 1: Easy data (should all do well)
    print("\n🧪 TEST 1: Easy Separable Data")
    print("(Classes are far apart - any optimizer should work)")
    results_sep = compare_optimizers(X_sep, y_sep, "Separable")
    plot_results(*results_sep[:3], "Separable")
    
    # Test 2: Hard data (here's where optimizers differ!)
    print("\n" + "="*60)
    print("\n🧪 TEST 2: Hard Near-Separable Data")
    print("(Classes overlap - this tests optimizer quality!)")
    results_near_sep = compare_optimizers(X_near_sep, y_near_sep, "Near-Separable")
    plot_results(*results_near_sep[:3], "Near-Separable")
    
    # Final accuracy comparison
    print("\n" + "="*60)
    print("🎯 FINAL ACCURACY COMPARISON")
    print("="*60)
    
    print("\n📈 On Easy (Separable) Data:")
    evaluate_accuracy(X_sep, y_sep, *results_sep[3:], "Separable")
    
    print("\n📉 On Hard (Near-Separable) Data:")
    evaluate_accuracy(X_near_sep, y_near_sep, *results_near_sep[3:], "Near-Separable")
    
    print("\n" + "="*60)
    print("✨ EXPERIMENT COMPLETE! ✨")
    print("\nKey Takeaways:")
    print("1. All optimizers work on easy data")
    print("2. Momentum helps convergence speed")
    print("3. Nesterov often gives slight edge")
    print("4. Try changing learning rates and momentum values!")
    print("\nHappy Learning! - kira-ml 👩‍💻")
    print("="*60)