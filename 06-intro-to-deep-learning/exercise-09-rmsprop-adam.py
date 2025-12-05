"""
Optimization Algorithms Playground: Understanding Gradient Descent vs RMSProp
==============================================================================
Welcome to the Optimization Playground! This tutorial helps you understand
how different optimization algorithms navigate the "Banana Valley" (Rosenbrock function).

As a beginner, you'll learn:
1. What makes an optimization problem "hard" vs "easy"
2. How gradient descent works (and where it struggles)
3. How RMSProp adapts learning rates to navigate tricky terrain
4. How to visualize optimization paths to understand algorithm behavior

Think of this as learning to hike:
- Gradient Descent: Always takes the same size steps
- RMSProp: Adjusts step size based on slope steepness
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Dict


# -----------------------------------------------------------------------------
# PART 1: The "Banana Valley" - Our Test Function
# -----------------------------------------------------------------------------

def rosenbrock(x: np.ndarray, a: float = 1.0, b: float = 100.0) -> Tuple[float, np.ndarray]:
    """
    The Rosenbrock "Banana Valley" Function - A Classic Optimization Challenge
    
    Beginner's Analogy: Imagine hiking in a curved banana-shaped valley!
    - The valley bottom is flat but winding (slow progress)
    - Valley walls are steep (hard to climb)
    - Minimum is at (1, 1) - our goal destination
    
    Why it's challenging for algorithms:
    1. Narrow valley: Easy to overshoot and "bounce" between walls
    2. Steep walls: Gradients are large on sides
    3. Flat bottom: Gradients are tiny near minimum (hard to know when to stop)
    
    Mathematical form: f(x,y) = (a - x)² + b(y - x²)²
    For a=1, b=100: Minimum at (1, 1) with f(1,1) = 0
    
    Visualize: A long, curved, banana-shaped valley with steep sides!
    """
    if x.shape[0] < 2:
        raise ValueError("Rosenbrock needs at least 2 dimensions (x and y coordinates)")
    
    # Unpack our coordinates
    x_coord, y_coord = x[0], x[1]
    
    # Calculate the "hiking difficulty" at this point
    # Part 1: Distance from x = a (horizontal distance from goal)
    # Part 2: Distance from y = x² (how far from valley bottom)
    valley_penalty = (a - x_coord) ** 2
    banana_penalty = b * (y_coord - x_coord ** 2) ** 2
    
    total_difficulty = valley_penalty + banana_penalty
    
    # Calculate slope in each direction (gradient)
    slope = np.zeros(2)
    
    # Slope in x-direction: How steep if we move east/west
    # Chain rule: -2(a - x) comes from valley_penalty
    #           -4b*x(y - x²) comes from banana_penalty
    slope[0] = -2 * (a - x_coord) - 4 * b * x_coord * (y_coord - x_coord ** 2)
    
    # Slope in y-direction: How steep if we move north/south
    # Much simpler: Only depends on banana_penalty
    slope[1] = 2 * b * (y_coord - x_coord ** 2)
    
    return total_difficulty, slope


# -----------------------------------------------------------------------------
# PART 2: The "Fixed-Step Hiker" - Gradient Descent
# -----------------------------------------------------------------------------

def gradient_descent(
    grad_fn: Callable,
    x0: np.ndarray,
    learning_rate: float = 0.001,
    max_iters: int = 1000,
    tol: float = 1e-6
) -> Dict:
    """
    Vanilla Gradient Descent - The Fixed-Step Hiker
    
    Beginner's Analogy: Imagine hiking with a pedometer that always says "take 1 meter steps"
    - Looks at current slope (gradient)
    - Takes step downhill (opposite to gradient)
    - Step size = learning_rate × slope_strength
    - Same step size everywhere!
    
    Pros: Simple, easy to understand
    Cons: 
    - Steep slopes: Steps too small (slow progress)
    - Gentle slopes: Steps too big (overshoot, oscillation)
    - Flat areas: Takes tiny steps (very slow)
    
    Real-world analogy: Using cruise control on a winding mountain road
    """
    x = x0.copy()  # Start at initial position
    hiking_path = [x.copy()]  # Record every step we take
    difficulties = []  # Record "hiking difficulty" at each step
    
    print(f"🚶 Starting Gradient Descent hike from {x0}")
    print(f"   Step size: {learning_rate}")
    
    for step in range(max_iters):
        # Check current position: How difficult? What's the slope?
        difficulty, slope = grad_fn(x)
        difficulties.append(difficulty)
        
        # Take a step downhill (opposite to slope direction)
        # Formula: new_position = current_position - step_size × slope_direction
        x_new = x - learning_rate * slope
        hiking_path.append(x_new.copy())
        
        # Did we move significantly?
        movement = np.linalg.norm(x_new - x)
        
        # Print progress occasionally
        if step % 100 == 0:
            print(f"   Step {step}: Difficulty = {difficulty:.4f}, Position = [{x[0]:.3f}, {x[1]:.3f}]")
        
        # Stop if we're barely moving (found flat spot)
        if movement < tol:
            print(f"✅ Gradient Descent converged after {step+1} steps!")
            print(f"   Final position: [{x_new[0]:.6f}, {x_new[1]:.6f}]")
            print(f"   Final difficulty: {difficulty:.6f}")
            break
        
        # Update position for next step
        x = x_new
    
    # If we never converged
    if step == max_iters - 1:
        print(f"⚠️  Gradient Descent reached max steps ({max_iters}) without converging")
    
    return {
        'x_opt': x,  # Final position
        'trajectory': np.array(hiking_path),  # Our hiking path
        'losses': difficulties,  # Difficulty log
        'iterations': step + 1  # Total steps taken
    }


# -----------------------------------------------------------------------------
# PART 3: The "Smart Hiker" - RMSProp
# -----------------------------------------------------------------------------

def rmsprop(
    grad_fn: Callable,
    x0: np.ndarray,
    learning_rate: float = 0.01,
    beta: float = 0.9,
    epsilon: float = 1e-8,
    max_iters: int = 1000,
    tol: float = 1e-6
) -> Dict:
    """
    RMSProp - The Smart Adaptive Hiker
    
    Beginner's Analogy: Imagine a hiker with a smart pedometer that:
    - Remembers past slopes in each direction
    - Takes smaller steps in consistently steep directions
    - Takes larger steps in consistently gentle directions
    - Adapts step size based on terrain history!
    
    How it works:
    1. Track "slope memory" for each direction (exponentially weighted average)
    2. Steep direction history → smaller steps (be careful!)
    3. Gentle direction history → larger steps (move faster!)
    
    Real-world analogy: A driver who slows down on curvy roads, speeds up on straights
    
    Mathematical insight:
    - s = β*s_old + (1-β)*gradient² (slope memory)
    - step = learning_rate × gradient / sqrt(s + ε) (adaptive step)
    """
    x = x0.copy()  # Start position
    hiking_path = [x.copy()]
    difficulties = []
    
    # "Slope memory" - remembers how steep each direction has been
    # Starts at 0 (no memory yet)
    slope_memory = np.zeros_like(x)
    
    print(f"🧠 Starting RMSProp hike from {x0}")
    print(f"   Base step size: {learning_rate}, Memory decay: {beta}")
    
    for step in range(max_iters):
        # Check current position
        difficulty, slope = grad_fn(x)
        difficulties.append(difficulty)
        
        # Update slope memory (weighted average of past squared slopes)
        # β controls how much we remember old slopes vs new slopes
        # High β (e.g., 0.9): Long memory, slow to adapt
        # Low β (e.g., 0.5): Short memory, quick to adapt
        slope_memory = beta * slope_memory + (1 - beta) * (slope ** 2)
        
        # Take adaptive step:
        # - Divide by sqrt(slope_memory): Normalizes by historical steepness
        # - ε prevents division by zero (tiny number for safety)
        adaptive_step = learning_rate * slope / (np.sqrt(slope_memory) + epsilon)
        x_new = x - adaptive_step
        hiking_path.append(x_new.copy())
        
        # Check progress
        movement = np.linalg.norm(x_new - x)
        
        if step % 100 == 0:
            print(f"   Step {step}: Difficulty = {difficulty:.4f}, Memory = [{slope_memory[0]:.3f}, {slope_memory[1]:.3f}]")
        
        if movement < tol:
            print(f"✅ RMSProp converged after {step+1} steps!")
            print(f"   Final position: [{x_new[0]:.6f}, {x_new[1]:.6f}]")
            print(f"   Final difficulty: {difficulty:.6f}")
            break
        
        x = x_new
    
    if step == max_iters - 1:
        print(f"⚠️  RMSProp reached max steps ({max_iters}) without converging")
    
    return {
        'x_opt': x,
        'trajectory': np.array(hiking_path),
        'losses': difficulties,
        'iterations': step + 1,
        'final_s': slope_memory  # Return final slope memory for insight
    }


# -----------------------------------------------------------------------------
# PART 4: Visualizing the Hike - See the Algorithms in Action!
# -----------------------------------------------------------------------------

def create_optimization_visualization(gd_result, rms_result, x0):
    """
    Create a beautiful visualization comparing the two hiking strategies
    
    We'll create:
    1. Contour plot of the Banana Valley
    2. Both hiking paths overlaid
    3. Convergence speed comparison
    4. Step size analysis
    
    This helps you SEE why RMSProp is often better!
    """
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('🚀 Optimization Algorithms: Gradient Descent vs RMSProp\n'
                'Navigating the Rosenbrock "Banana Valley"', 
                fontsize=16, fontweight='bold')
    
    # ------------------------------------------------------------
    # Subplot 1: The Valley Terrain with Hiking Paths
    # ------------------------------------------------------------
    ax = axes[0, 0]
    
    # Create grid for contour plot
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculate Rosenbrock values on grid
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j], _ = rosenbrock(np.array([X[i, j], Y[i, j]]))
    
    # Plot contour (valley lines)
    levels = np.logspace(-1, 4, 10)  # Logarithmic levels for better visualization
    contour = ax.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
    ax.clabel(contour, inline=True, fontsize=8)
    
    # Plot hiking paths
    ax.plot(gd_result['trajectory'][:, 0], gd_result['trajectory'][:, 1], 
            'ro-', linewidth=2, markersize=4, label='Gradient Descent', alpha=0.7)
    ax.plot(rms_result['trajectory'][:, 0], rms_result['trajectory'][:, 1], 
            'bo-', linewidth=2, markersize=4, label='RMSProp', alpha=0.7)
    
    # Mark start and end points
    ax.plot(x0[0], x0[1], 'g*', markersize=15, label='Start', markeredgecolor='black')
    ax.plot(1, 1, 'y*', markersize=15, label='Goal (1,1)', markeredgecolor='black')
    
    ax.set_xlabel('x coordinate')
    ax.set_ylabel('y coordinate')
    ax.set_title('Hiking Paths Through Banana Valley')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # ------------------------------------------------------------
    # Subplot 2: Difficulty Over Time (Convergence Plot)
    # ------------------------------------------------------------
    ax = axes[0, 1]
    
    # Plot loss curves (log scale shows relative improvement better)
    gd_steps = range(len(gd_result['losses']))
    rms_steps = range(len(rms_result['losses']))
    
    ax.semilogy(gd_steps, gd_result['losses'], 'r-', linewidth=2, label='Gradient Descent')
    ax.semilogy(rms_steps, rms_result['losses'], 'b-', linewidth=2, label='RMSProp')
    
    ax.set_xlabel('Step Number')
    ax.set_ylabel('Difficulty (Loss) - Log Scale')
    ax.set_title('Convergence Speed: Who Gets "Unstuck" Faster?')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    ax.annotate(f"GD: {gd_result['iterations']} steps\nFinal: {gd_result['losses'][-1]:.2e}",
                xy=(gd_result['iterations'], gd_result['losses'][-1]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.2))
    
    ax.annotate(f"RMS: {rms_result['iterations']} steps\nFinal: {rms_result['losses'][-1]:.2e}",
                xy=(rms_result['iterations'], rms_result['losses'][-1]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round', facecolor='blue', alpha=0.2))
    
    # ------------------------------------------------------------
    # Subplot 3: Step-by-Step Movement Analysis
    # ------------------------------------------------------------
    ax = axes[1, 0]
    
    # Calculate step sizes for each algorithm
    gd_traj = gd_result['trajectory']
    rms_traj = rms_result['trajectory']
    
    gd_step_sizes = np.linalg.norm(gd_traj[1:] - gd_traj[:-1], axis=1)
    rms_step_sizes = np.linalg.norm(rms_traj[1:] - rms_traj[:-1], axis=1)
    
    ax.plot(gd_step_sizes, 'r-', alpha=0.7, label='Gradient Descent Steps')
    ax.plot(rms_step_sizes, 'b-', alpha=0.7, label='RMSProp Steps')
    
    ax.set_xlabel('Step Number')
    ax.set_ylabel('Step Size (Distance Moved)')
    ax.set_title('How Step Sizes Change During Optimization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add horizontal lines for average step sizes
    ax.axhline(y=np.mean(gd_step_sizes), color='red', linestyle='--', alpha=0.5, 
               label=f'GD Avg: {np.mean(gd_step_sizes):.3f}')
    ax.axhline(y=np.mean(rms_step_sizes), color='blue', linestyle='--', alpha=0.5,
               label=f'RMS Avg: {np.mean(rms_step_sizes):.3f}')
    ax.legend(loc='upper right')
    
    # ------------------------------------------------------------
    # Subplot 4: Final Performance Comparison
    # ------------------------------------------------------------
    ax = axes[1, 1]
    
    # Prepare data for bar chart
    algorithms = ['Gradient Descent', 'RMSProp']
    final_losses = [gd_result['losses'][-1], rms_result['losses'][-1]]
    iterations = [gd_result['iterations'], rms_result['iterations']]
    
    x_pos = np.arange(len(algorithms))
    
    # Create bars
    bars1 = ax.bar(x_pos - 0.2, final_losses, 0.4, label='Final Loss', color=['red', 'blue'], alpha=0.7)
    bars2 = ax.bar(x_pos + 0.2, iterations, 0.4, label='Steps Taken', color=['pink', 'lightblue'], alpha=0.7)
    
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Value')
    ax.set_title('Final Performance Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algorithms)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}' if bars == bars2 else f'{height:.2e}',
                   ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# PART 5: Let's Hike! Running the Experiment
# -----------------------------------------------------------------------------

def run_optimization_experiment():
    """
    Main function to run the complete optimization comparison experiment
    
    We'll:
    1. Start from a challenging point in the valley
    2. Run both optimizers
    3. Visualize their journeys
    4. Compare their performance
    """
    print("=" * 70)
    print("🚀 OPTIMIZATION PLAYGROUND: Banana Valley Challenge")
    print("=" * 70)
    
    # Test Rosenbrock function at origin
    print("\n📍 Testing Rosenbrock function at origin (0, 0):")
    loss, grad = rosenbrock(np.array([0.0, 0.0]))
    print(f"   Difficulty: {loss:.2f}")
    print(f"   Slope direction: [{grad[0]:.2f}, {grad[1]:.2f}]")
    print("   Interpretation: Starting uphill climb!")
    
    # Choose a challenging starting point
    # This point is in the valley but far from minimum
    x0 = np.array([-1.0, 2.0])
    print(f"\n🗺️  Starting our hike from: {x0}")
    print("   (This is in the banana valley but far from the minimum at [1, 1])")
    
    # -----------------------------------------------------------------
    # Run Gradient Descent (Fixed-Step Hiker)
    # -----------------------------------------------------------------
    print("\n" + "-" * 70)
    print("🧍 GRADIENT DESCENT: The Fixed-Step Hiker")
    print("-" * 70)
    
    gd_result = gradient_descent(
        rosenbrock, 
        x0, 
        learning_rate=0.001,  # Small steps to avoid overshooting
        max_iters=5000,  # Might need many steps
        tol=1e-6
    )
    
    # -----------------------------------------------------------------
    # Run RMSProp (Smart Adaptive Hiker)
    # -----------------------------------------------------------------
    print("\n" + "-" * 70)
    print("🧠 RMSPROP: The Smart Adaptive Hiker")
    print("-" * 70)
    
    rms_result = rmsprop(
        rosenbrock,
        x0,
        learning_rate=0.01,  # Can use larger base learning rate
        beta=0.9,  # Remember 90% of past, 10% of new
        max_iters=2000,  # Usually needs fewer steps
        tol=1e-6
    )
    
    # -----------------------------------------------------------------
    # Create Visualization
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("📊 CREATING VISUALIZATION...")
    print("=" * 70)
    
    create_optimization_visualization(gd_result, rms_result, x0)
    
    # -----------------------------------------------------------------
    # Summary Insights
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("💡 KEY INSIGHTS FOR BEGINNERS")
    print("=" * 70)
    
    print("\n1. Gradient Descent (Fixed Steps):")
    print("   - Takes same size steps everywhere")
    print("   - Slow in flat valley bottom")
    print("   - Can oscillate on valley walls")
    print(f"   - Took {gd_result['iterations']} steps to get to {gd_result['losses'][-1]:.2e}")
    
    print("\n2. RMSProp (Adaptive Steps):")
    print("   - Remembers past slope steepness")
    print("   - Takes smaller steps in steep directions")
    print("   - Takes larger steps in gentle directions")
    print(f"   - Took {rms_result['iterations']} steps to get to {rms_result['losses'][-1]:.2e}")
    
    print("\n3. When to use each:")
    print("   ✓ Gradient Descent: Simple problems, convex functions")
    print("   ✓ RMSProp: Complex landscapes, varying curvatures")
    print("   ✓ RMSProp is usually better for deep learning!")
    
    print("\n4. Try changing these parameters to learn more:")
    print("   - learning_rate: How big are steps? (Try 0.0001 vs 0.01)")
    print("   - beta: How much memory? (Try 0.5 vs 0.99)")
    print("   - x0: Start from different points!")
    
    print("\n" + "=" * 70)
    print("🎉 EXPERIMENT COMPLETE! Happy optimizing! 🎉")
    print("=" * 70)


# Run the experiment if this script is executed directly
if __name__ == "__main__":
    run_optimization_experiment()