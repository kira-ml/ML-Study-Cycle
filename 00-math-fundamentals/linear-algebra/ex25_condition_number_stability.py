"""
🎮 CONDITION NUMBER SIMULATOR: ML's "Butterfly Effect" Detector

Think of condition number as the "drama amplifier" of your ML models!
High condition number = Small data changes cause BIG prediction drama. 😱

Created by: @kira-ml (GitHub ML Student)
#MachineLearning #DataScience #NumericalStability #MathForML
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hilbert

def compute_condition_number_demo():
    """
    🎭 THE CONDITION NUMBER SHOWDOWN:
    Good Matrix vs. Drama Queen Matrix 🎯
    
    CONDITION NUMBER TL;DR:
    • Low (< 1000) = Chill matrix 😎
    • High (> 10^10) = Drama queen matrix 😱
    • Measures "how much small changes in input cause big changes in output"
    
    REAL-WORLD ANALOGY:
    • Low condition number = Stable friendship (small fights don't break it)
    • High condition number = High school drama (one rumor ruins everything)
    
    Created by: @kira-ml (GitHub ML Student)
    """
    print("\n" + "🎬" * 30)
    print("EPISODE 1: STABLE VS. DRAMATIC MATRICES")
    print("🎬" * 30)
    
    # Scene 1: The Chill Matrix 😎
    print("\n🎯 SCENE 1: THE CHILL MATRIX (Well-conditioned)")
    print("   Like a stable relationship - small fights don't break it!")
    
    A1 = np.array([[2, 1], [1, 2]])  # Nice, stable matrix
    b1 = np.array([3, 3])
    
    cond_A1 = np.linalg.cond(A1)
    x1 = np.linalg.solve(A1, b1)
    
    print(f"\n📊 Matrix A (Chill):")
    print(A1)
    print(f"🧮 Condition number: {cond_A1:.2f} (Low drama! 😎)")
    print(f"🎯 True solution x: {x1}")
    
    # Add some drama (small perturbation)
    print("\n🎭 ADDING A LITTLE DRAMA...")
    b1_perturbed = b1 + np.array([0.01, -0.01])  # Tiny change
    x1_perturbed = np.linalg.solve(A1, b1_perturbed)
    error1 = np.linalg.norm(x1_perturbed - x1) / np.linalg.norm(x1)
    
    print(f"📈 New b (with drama): {b1_perturbed}")
    print(f"🎯 New solution: {x1_perturbed}")
    print(f"⚠️  Relative error: {error1:.2%}")
    print("💡 Insight: Small input change → Small output change (Good!)")
    
    # Scene 2: The Drama Queen Matrix 👑
    print("\n\n🎯 SCENE 2: THE DRAMA QUEEN MATRIX (Ill-conditioned)")
    print("   Like high school drama - one rumor ruins everything! 😱")
    
    n = 5
    A2 = hilbert(n)  # Famous for being dramatic!
    x_true = np.ones(n)
    b2 = A2 @ x_true
    
    cond_A2 = np.linalg.cond(A2)
    x2 = np.linalg.solve(A2, b2)
    
    print(f"\n📊 Matrix A (Hilbert Matrix - Professional Drama Queen):")
    print(A2)
    print(f"🧮 Condition number: {cond_A2:.2e} (OMG SO DRAMATIC! 😱)")
    print(f"🎯 True solution x: {x_true}")
    print(f"🎯 Computed solution: {x2}")
    print(f"⚠️  Relative error: {np.linalg.norm(x2 - x_true) / np.linalg.norm(x_true):.2%}")
    print("💡 Insight: Perfect input → Still gets wrong answer!")
    
    # Scene 3: The "Almost Twins" Matrix 👯
    print("\n\n🎯 SCENE 3: THE 'ALMOST TWINS' MATRIX (Nearly Singular)")
    print("   Like two nearly identical people - hard to tell apart!")
    
    A3 = np.array([[1, 1], [1, 1.0001]])  # Almost identical rows
    b3 = np.array([2, 2.0001])
    
    cond_A3 = np.linalg.cond(A3)
    x3 = np.linalg.solve(A3, b3)
    
    print(f"\n📊 Matrix A (Almost Twins):")
    print(A3)
    print(f"🧮 Condition number: {cond_A3:.2f} (High drama alert!)")
    print(f"🎯 Solution x: {x3}")
    
    # Add microscopic drama
    print("\n🎭 ADDING MICROSCOPIC DRAMA...")
    b3_perturbed = b3 + np.array([0.001, 0])  # SUPER tiny change
    x3_perturbed = np.linalg.solve(A3, b3_perturbed)
    error3 = np.linalg.norm(x3_perturbed - x3) / np.linalg.norm(x3)
    
    print(f"📈 New b (micro-drama): {b3_perturbed}")
    print(f"🎯 New solution: {x3_perturbed}")
    print(f"⚠️  Relative error: {error3:.2%}")
    print("💡 Insight: Microscopic input change → MACROSCOPIC output change!")

def condition_number_vs_error():
    """
    📈 THE DRAMA GRAPH: How Condition Number Creates Chaos
    
    This plot shows why ML engineers fear high condition numbers!
    It's the "butterfly effect" visualization for matrices.
    
    Created by: @kira-ml (GitHub ML Student)
    """
    print("\n\n" + "📊" * 30)
    print("EPISODE 2: THE DRAMA-ERROR CONNECTION")
    print("📊" * 30)
    
    sizes = range(3, 12)
    condition_numbers = []
    relative_errors = []
    
    print("\n🔬 EXPERIMENT: Growing Hilbert Matrices")
    print("   Hilbert matrices get MORE dramatic as they grow! 📈")
    
    for n in sizes:
        A = hilbert(n)  # Professional drama queen matrix
        x_true = np.ones(n)
        b = A @ x_true
        
        cond_num = np.linalg.cond(A)
        condition_numbers.append(cond_num)
        
        x_computed = np.linalg.solve(A, b)
        error = np.linalg.norm(x_computed - x_true) / np.linalg.norm(x_true)
        relative_errors.append(error)
        
        print(f"\n🎭 n={n}x{n} Hilbert Matrix:")
        print(f"   Condition number: {cond_num:.2e}")
        print(f"   Relative error: {error:.2e}")
        
        if cond_num > 1e10:
            print("   ⚠️  WARNING: Condition number > 10^10! Epic drama levels!")
        elif cond_num > 1e6:
            print("   ⚠️  WARNING: Condition number > 10^6! High drama!")
    
    # Create dramatic visualization
    plt.figure(figsize=(14, 6))
    plt.suptitle('🎭 The Condition Number Drama Effect 🎭\nCreated by: @kira-ml', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Condition Number Growth
    plt.subplot(1, 2, 1)
    plt.semilogy(sizes, condition_numbers, 'r^-', linewidth=3, markersize=10, 
                 label='Drama Level')
    plt.fill_between(sizes, condition_numbers, alpha=0.2, color='red')
    plt.xlabel('Matrix Size (n x n)', fontsize=12)
    plt.ylabel('Condition Number (log scale)', fontsize=12)
    plt.title('📈 How Drama Grows with Size', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add drama zones
    plt.axhline(y=1e10, color='red', linestyle='--', alpha=0.5, 
                label='Epic Drama Zone')
    plt.axhline(y=1e6, color='orange', linestyle='--', alpha=0.5,
                label='High Drama Zone')
    
    # Plot 2: Error Explosion
    plt.subplot(1, 2, 2)
    plt.semilogy(sizes, relative_errors, 'bs-', linewidth=3, markersize=10,
                 label='Error Level')
    plt.fill_between(sizes, relative_errors, alpha=0.2, color='blue')
    plt.xlabel('Matrix Size (n x n)', fontsize=12)
    plt.ylabel('Relative Error (log scale)', fontsize=12)
    plt.title('💥 Error Explosion from Drama', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("\n💡 KEY TAKEAWAY:")
    print("   Bigger Hilbert matrices = MORE drama = MORE error!")
    print("   This is why ML models can fail with correlated features!")

def stability_analysis():
    """
    🛡️ ML SUPERWEAPON: Regularization (The Drama Reducer)
    
    Regularization is like giving your ML model chill pills! 💊
    It reduces condition number and prevents overfitting drama.
    
    REAL ML EXAMPLE: Ridge Regression adds λI to XᵀX
    
    Created by: @kira-ml (GitHub ML Student)
    """
    print("\n\n" + "🛡️" * 30)
    print("EPISODE 3: THE DRAMA REDUCER (Regularization)")
    print("🛡️" * 30)
    
    # Simulate a typical ML dataset with drama (correlated features)
    np.random.seed(42)
    n_samples, n_features = 100, 10
    
    print("\n🎯 SETUP: Simulating ML Features with Drama")
    print("   Creating features that are highly correlated...")
    print("   (Common in real datasets like housing prices)")
    
    X = np.random.randn(n_samples, n_features)
    # Make features 0, 1, 2 highly correlated (DRAMA SOURCE!)
    X[:, 1] = X[:, 0] + 0.01 * np.random.randn(n_samples)
    X[:, 2] = X[:, 0] + 0.02 * np.random.randn(n_samples)
    
    # Normal equation for linear regression: XᵀX w = Xᵀy
    XTX = X.T @ X
    cond_original = np.linalg.cond(XTX)
    
    print(f"\n📊 ORIGINAL XᵀX Matrix:")
    print(f"   Condition number: {cond_original:.2e}")
    
    if cond_original > 1e10:
        print("   🔴 CRITICAL DRAMA: Condition > 10^10!")
        print("   Model predictions will be UNSTABLE!")
    elif cond_original > 1e6:
        print("   🟡 HIGH DRAMA: Condition > 10^6!")
        print("   Model might overfit to noise!")
    else:
        print("   🟢 LOW DRAMA: Model should be stable!")
    
    # The superhero: REGULARIZATION!
    print("\n\n🦸 SUPERHERO ENTERS: REGULARIZATION!")
    print("   Adding λI to XᵀX (Ridge Regression trick)")
    print("   λ = regularization strength (drama reducer power)")
    
    lambda_vals = [0, 1e-8, 1e-6, 1e-4, 1e-2]
    
    print("\n" + "🧪" * 50)
    print("EXPERIMENT: How λ Reduces Drama")
    print("λ (lambda)\tCondition Number\tDrama Reduction\tEffect")
    print("-" * 60)
    
    for lambda_val in lambda_vals:
        XTX_regularized = XTX + lambda_val * np.eye(n_features)
        cond_regularized = np.linalg.cond(XTX_regularized)
        improvement = cond_original / cond_regularized
        
        # Fun drama rating
        if improvement > 1000:
            drama_effect = "🎭 EPIC CALMING! 🎭"
        elif improvement > 100:
            drama_effect = "😌 Super chill"
        elif improvement > 10:
            drama_effect = "😊 Much calmer"
        elif improvement > 2:
            drama_effect = "🙂 A bit calmer"
        else:
            drama_effect = "😐 Still dramatic"
        
        print(f"{lambda_val:.0e}\t\t{cond_regularized:.2e}\t\t{improvement:.1f}x\t\t{drama_effect}")
    
    print("\n💡 REGULARIZATION TRADE-OFF:")
    print("   More λ = Less drama (better stability) = Less fitting power")
    print("   Less λ = More drama (more unstable) = More fitting power")
    print("   Sweet spot usually λ = 1e-4 to 1e-2 for many ML problems")

def main():
    """
    🚀 MAIN COURSE: Your Journey from Drama to Stability
    
    Welcome to the Condition Number Bootcamp! You'll learn:
    1. What condition number REALLY means for ML
    2. How to spot drama queen matrices
    3. How to fix them with regularization
    
    Created by: @kira-ml (GitHub ML Student)
    Follow my ML journey on GitHub! 👩💻
    """
    print("\n" + "🌟" * 50)
    print("WELCOME TO: CONDITION NUMBER BOOTCAMP!")
    print("Learn ML's Most Important Stability Concept")
    print("🌟" * 50)
    
    print("\n👋 Hey! I'm Kira, an ML student on GitHub (@kira-ml)")
    print("   I created this tutorial to make numerical stability FUN!")
    
    print("\n🎯 TODAY'S MISSION:")
    print("   • Episode 1: Spot drama queen matrices 🎭")
    print("   • Episode 2: See the drama-error connection 📈")
    print("   • Episode 3: Learn to reduce drama with regularization 🛡️")
    
    input("\n🎬 Press Enter to start Episode 1...")
    compute_condition_number_demo()
    
    input("\n📊 Press Enter for Episode 2 (with plots!)...")
    condition_number_vs_error()
    
    input("\n🛡️ Press Enter for Episode 3 (ML applications!)...")
    stability_analysis()
    
    # Grand Finale!
    print("\n\n" + "🎓" * 50)
    print("CONGRATULATIONS! YOU'VE MASTERED CONDITION NUMBERS!")
    print("🎓" * 50)
    
    print("\n🔥 YOUR NEW ML SUPERPOWERS:")
    print("   1. 🎭 Spot drama queen matrices before they ruin your models")
    print("   2. 📈 Understand why errors explode with high condition numbers")
    print("   3. 🛡️ Use regularization to stabilize ANY ML model")
    print("   4. 🔍 Debug why your model gives weird predictions")
    
    print("\n📚 REAL-WORLD APPLICATIONS:")
    print("   • Linear/Logistic Regression → Check XᵀX condition number")
    print("   • Neural Networks → High condition numbers cause vanishing/exploding gradients")
    print("   • Recommendation Systems → Matrix factorization stability")
    print("   • Computer Vision → Numerical stability in transformations")
    
    print("\n👩💻 NEXT STEPS FOR @kira-ml FRIENDS:")
    print("   1. Try sklearn's Ridge() and Lasso() with different λ values")
    print("   2. Check condition numbers of your own datasets with np.linalg.cond()")
    print("   3. Follow me on GitHub for more beginner-friendly ML tutorials!")
    
    print("\n" + "💖" * 50)
    print("Remember: Good ML engineers don't just build models,")
    print("they build STABLE models. You've got this! 💪")
    print("💖" * 50)
    
    print("\n#MLNewbie #DataScience #Python #NumericalStability")
    print("Created with ❤️ by @kira-ml (GitHub ML Student)")

if __name__ == "__main__":
    # Clear screen for fresh start
    print("\033c", end="")
    main()