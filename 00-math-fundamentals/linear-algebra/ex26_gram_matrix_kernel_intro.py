"""
🔥 GRAM MATRIX & KERNEL TRICK EXPLAINED 🔥

Think of this as your cheat sheet for understanding how ML models 
"see" patterns in data! Gram matrices are like friendship networks 
for data points - they show who's similar to whom.

TL;DR for TikTok attention spans:
• Gram matrix = Tinder for data points (who matches with whom)
• Kernel trick = Magic that finds patterns without doing hard math
• SVMs use this to draw the best boundary between different groups

#MachineLearning #AI #DataScience #CodingForGenZ
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel

def compute_gram_matrix_linear(X):
    """
    🎯 WHAT THIS DOES:
    Calculates the "friendship matrix" for your data points using linear similarity.
    
    Think of it like: "How similar is each student's Spotify playlist to every other student's?"
    
    FORMULA (the math behind the magic):
    G[i, j] = dot_product(X[i], X[j]) = sum of (feature_i * feature_j)
    
    EXAMPLE:
    If X = [[1, 2],  # Student A: likes pop(1) and rock(2)
            [3, 4]]  # Student B: likes pop(3) and rock(4)
    Then G = [[1*1+2*2, 1*3+2*4],  = [[5, 11],
              [3*1+4*2, 3*3+4*4]]     [11, 25]]
    
    Parameters:
    X : Your data, shaped like (n_samples, n_features)
        Example: 100 students × 10 music genres they like
    
    Returns:
    G : The friendship matrix (Gram matrix)
        Example: 100 × 100 matrix showing playlist similarity scores
    """
    # @ symbol = matrix multiplication (like Excel's SUMPRODUCT on steroids)
    return X @ X.T

def verify_gram_properties(G, tolerance=1e-10):
    """
    🕵️♂️ GRAM MATRIX POLICE - Checking if it's valid:
    
    Gram matrices must follow 2 golden rules:
    1. SYMMETRIC: If A is friends with B, then B is friends with A
       (Makes sense: similarity goes both ways!)
    2. POSITIVE SEMI-DEFINITE: All "vibes" (eigenvalues) are non-negative
       (No negative friendships allowed!)
    
    🎮 ANALOGY:
    Imagine a social network where:
    • Every friendship is mutual (symmetric)
    • No person has overall negative "social energy" (positive semi-definite)
    
    Parameters:
    G : Your Gram matrix to check
    tolerance : Small number for computer math (computers aren't perfect!)
    
    Returns:
    dict : Report card for your Gram matrix
    """
    results = {}
    
    # Rule 1: Check if symmetric (friendships are mutual)
    is_symmetric = np.allclose(G, G.T, atol=tolerance)
    results['symmetric'] = is_symmetric
    print("🔍 Symmetry check: " + ("✅ Pass! Friendships are mutual" 
          if is_symmetric else "❌ Fail! Some friendships aren't mutual"))
    
    # Rule 2: Check positive semi-definite (all vibes ≥ 0)
    eigenvalues = np.linalg.eigvalsh(G)  # These are the "vibe scores"
    all_non_negative = np.all(eigenvalues >= -tolerance)
    results['positive_semi_definite'] = all_non_negative
    results['eigenvalues'] = eigenvalues
    
    vibe_status = "✅ All good vibes" if all_non_negative else "❌ Negative vibes detected"
    print(f"🔍 PSD check: {vibe_status}")
    print(f"   Eigenvalues (vibe scores): {eigenvalues}")
    
    # Cool extra metrics (like social media analytics for your data!)
    results['rank'] = np.linalg.matrix_rank(G)
    results['trace'] = np.trace(G)
    results['frobenius_norm'] = np.linalg.norm(G, 'fro')
    
    print(f"📊 Matrix Stats:")
    print(f"   Rank (unique friend groups): {results['rank']}")
    print(f"   Trace (total friendship energy): {results['trace']:.2f}")
    print(f"   Frobenius Norm (overall friendship strength): {results['frobenius_norm']:.2f}")
    
    return results

def demonstrate_linear_gram():
    """
    🎮 HANDS-ON EXAMPLE: Your first Gram matrix!
    
    We'll create a tiny dataset and watch the magic happen.
    Perfect for understanding before scaling to big data.
    """
    print("\n" + "✨" * 35)
    print("LEVEL 1: YOUR FIRST GRAM MATRIX")
    print("✨" * 35)
    
    # Let's make a tiny "music taste" dataset
    # 3 students, 2 genres (pop and rock scores 1-10)
    print("\n🎵 Our Tiny Music Taste Dataset:")
    print("Student 1: Pop=1, Rock=2")
    print("Student 2: Pop=3, Rock=4") 
    print("Student 3: Pop=5, Rock=6")
    
    X_simple = np.array([[1, 2],  # Student 1
                         [3, 4],  # Student 2
                         [5, 6]]) # Student 3
    
    print(f"\n📋 Data matrix X (3 students × 2 genres):")
    print(X_simple)
    
    # Method 1: Manual calculation (showing the math)
    print("\n🧮 CALCULATION TIME:")
    print("G[0,0] = Student1·Student1 = 1*1 + 2*2 = 5")
    print("G[0,1] = Student1·Student2 = 1*3 + 2*4 = 11")
    print("G[1,2] = Student2·Student3 = 3*5 + 4*6 = 39")
    
    G_manual = compute_gram_matrix_linear(X_simple)
    G_sklearn = linear_kernel(X_simple)
    
    print(f"\n🎯 Gram matrix G (manual):")
    print(G_manual)
    print(f"\n🎯 Gram matrix G (using sklearn's linear_kernel):")
    print(G_sklearn)
    
    # Verify it's a proper Gram matrix
    print("\n" + "🔬" * 35)
    print("GRAM MATRIX VALIDATION REPORT:")
    properties = verify_gram_properties(G_manual)
    
    # Bonus: What does this mean for ML?
    print("\n💡 WHAT THIS MEANS FOR ML:")
    print(f"• Highest similarity: Students {np.unravel_index(np.argmax(G_manual), G_manual.shape)}")
    print(f"• Most similar pair has score: {np.max(G_manual - np.diag(np.diag(G_manual)))}")
    print("• Diagonal = self-similarity (always highest!)")

def demonstrate_kernel_trick():
    """
    🎩 THE KERNEL TRICK - ML's best magic trick!
    
    Instead of transforming data to a new space (expensive!),
    we compute similarities directly. It's like judging a cooking
    competition by tasting, not by listing every ingredient.
    """
    print("\n\n" + "🎩" * 35)
    print("LEVEL 2: KERNEL TRICK UNLOCKED!")
    print("🎩" * 35)
    
    print("\n🤔 PROBLEM: Some data isn't linearly separable")
    print("   Like separating 🍩 from 🥯 - need curves, not straight lines!")
    
    # Create some random 2D points
    np.random.seed(42)  # Same random numbers every time (for teaching!)
    n_samples = 4
    X = np.random.randn(n_samples, 2)
    
    print(f"\n📊 Our data (4 random points in 2D space):")
    print(X)
    
    print("\n🎭 DIFFERENT KERNELS = DIFFERENT SIMILARITY RULES:")
    
    # Kernel 1: Linear (basic similarity)
    print("\n1️⃣ LINEAR KERNEL - 'How aligned are the arrows?'")
    G_linear = linear_kernel(X)
    print(f"   Matrix shape: {G_linear.shape}")
    print(f"   Sample value: G[0,1] = {G_linear[0,1]:.3f}")
    
    # Kernel 2: RBF/Gaussian (exponential similarity)
    print("\n2️⃣ RBF KERNEL - 'How close are the points?'")
    print("   Like: 'I only befriend people within 100m'")
    G_rbf = rbf_kernel(X, gamma=0.1)
    print(f"   Matrix shape: {G_rbf.shape}")
    print(f"   Sample value: G[0,1] = {G_rbf[0,1]:.3f}")
    print(f"   Closest pair: {np.max(G_rbf - np.eye(len(G_rbf))):.3f}")
    
    # Kernel 3: Polynomial (pattern matching)
    print("\n3️⃣ POLYNOMIAL KERNEL - 'Do they have similar patterns?'")
    print("   Like: 'Do both playlists have metal AND classical?'")
    G_poly = polynomial_kernel(X, degree=2)
    print(f"   Matrix shape: {G_poly.shape}")
    print(f"   Sample value: G[0,1] = {G_poly[0,1]:.3f}")
    
    # The magic: All give valid Gram matrices!
    print("\n" + "🎯" * 35)
    print("THE MAGIC: All kernels create valid Gram matrices!")
    for name, G in [("Linear", G_linear), ("RBF", G_rbf), ("Polynomial", G_poly)]:
        props = verify_gram_properties(G)
        print(f"\n{name}: Symmetric={props['symmetric']}, PSD={props['positive_semi_definite']}")

def feature_space_visualization():
    """
    🚀 SPACE ELEVATOR: Lifting data to higher dimensions!
    
    Sometimes data needs to be "lifted" to see patterns clearly.
    Like needing 3D glasses to see a movie's effects.
    """
    print("\n\n" + "🚀" * 35)
    print("LEVEL 3: FEATURE SPACE - THE DATA ELEVATOR")
    print("🚀" * 35)
    
    print("\n🎯 GOAL: Separate circular data (like separating a 🍩 from its hole)")
    
    # Create points on a circle (classic non-linear problem)
    theta = np.linspace(0, 2*np.pi, 8, endpoint=False)
    X_original = np.column_stack([np.cos(theta), np.sin(theta)])
    
    print("\n📍 Original 2D points (on a circle):")
    print("   Imagine trying to separate inner vs outer ring with a straight line...")
    print("   Impossible in 2D! Need to add dimensions.")
    
    # Polynomial feature mapping: 2D → 6D!
    def polynomial_feature_map(X, degree=2):
        """
        ✨ MANIFESTING HIGHER DIMENSIONS ✨
        
        Original point: [x, y]
        Mapped to: [x², y², √2xy, √2x, √2y, 1]
        
        WHY? x² + y² could be the radius squared!
        Now we can separate by radius (inner vs outer circle).
        """
        n_samples = X.shape[0]
        x1, x2 = X[:, 0], X[:, 1]
        
        if degree == 2:
            phi = np.column_stack([
                x1**2,  # x-squared dimension
                x2**2,  # y-squared dimension  
                np.sqrt(2)*x1*x2,  # interaction dimension
                np.sqrt(2)*x1,  # scaled x dimension
                np.sqrt(2)*x2,  # scaled y dimension
                np.ones(n_samples)  # bias dimension
            ])
        return phi
    
    # Show the transformation
    X_mapped = polynomial_feature_map(X_original)
    
    print("\n🔮 Transformation: 2D → 6D")
    print(f"   Before: [x, y] = {X_original[0]}")
    print(f"   After : [x², y², √2xy, √2x, √2y, 1] = {X_mapped[0]}")
    
    # The kernel trick: Skip the 6D computation!
    print("\n" + "🎩" * 35)
    print("KERNEL TRICK IN ACTION:")
    print("Instead of computing in 6D (expensive!),")
    print("we compute K(x,y) = (x·y + 1)² directly in 2D!")
    
    G_direct = X_mapped @ X_mapped.T
    G_kernel = polynomial_kernel(X_original, degree=2)
    
    print("\n📊 Direct 6D computation vs Kernel trick:")
    print("✅ Same results!" if np.allclose(G_direct, G_kernel) else "❌ Different!")
    print(f"\nDirect (6D math):\n{G_direct}")
    print(f"\nKernel (2D shortcut):\n{G_kernel}")
    
    print("\n💡 REAL-WORLD IMPACT:")
    print("• 2D → 6D: 3× more computation")
    print("• 2D → 1000D: Kernel trick saves 99.8% computation!")
    print("• This is why SVMs with kernels work on huge datasets")

def svm_connection_demo():
    """
    ⚔️ SVM SHOWDOWN: How Gram matrices power Support Vector Machines
    
    SVMs use Gram matrices to find the optimal boundary.
    Think of it as finding the fairest line between two friend groups.
    """
    print("\n\n" + "⚔️" * 35)
    print("LEVEL 4: SVM - THE ULTIMATE BOUNDARY FINDER")
    print("⚔️" * 35)
    
    print("\n🎯 MISSION: Separate cat people 🐱 from dog people 🐶")
    
    # Create two groups of people
    np.random.seed(42)
    n_per_class = 3
    
    # Group 1: Cat people (prefer cats > dogs)
    X_cats = np.random.randn(n_per_class, 2) + [2, 2]
    # Group 2: Dog people (prefer dogs > cats)  
    X_dogs = np.random.randn(n_per_class, 2) + [-2, -2]
    
    X = np.vstack([X_cats, X_dogs])
    y = np.array([1, 1, 1, -1, -1, -1])  # 1=🐱, -1=🐶
    
    print("\n📊 The Data:")
    print(f"Cat people positions (first 3):\n{X[:3]}")
    print(f"Dog people positions (last 3):\n{X[3:]}")
    print(f"Labels: {y} (1=🐱, -1=🐶)")
    
    # RBF kernel: "How similar are their pet preferences?"
    G_rbf = rbf_kernel(X, gamma=0.1)
    
    print("\n🎭 Similarity Matrix (RBF Kernel):")
    print("High values = similar preferences")
    print(G_rbf)
    
    # SVM's secret sauce: y_i * y_j * K(x_i, x_j)
    yyT = np.outer(y, y)  # Agreement matrix: 1=both same, -1=different
    SVM_matrix = yyT * G_rbf  # The actual matrix SVM optimizes
    
    print("\n" + "🧠" * 35)
    print("SVM'S BRAIN: The Optimization Matrix")
    print("Value = (agreement) × (similarity)")
    print("Positive = Helpful (same class & similar)")
    print("Negative = Problematic (different class & similar)")
    print(SVM_matrix)
    
    print("\n🔍 What SVM sees:")
    print("• High positive values = Easy decisions (clear friends)")
    print("• High negative values = Hard decisions (enemies who look like friends)")
    print("• SVM tries to maximize positive, minimize negative")

def practical_insights():
    """
    💎 REAL-WORLD GEMS: What you actually need to know
    """
    print("\n\n" + "💎" * 35)
    print("BONUS LEVEL: PRACTICAL TIPS & TRICKS")
    print("💎" * 35)
    
    insights = [
        "\n🎯 TIP 1: Gram Matrix = Data's Social Network",
        "   • Rows/columns = data points",
        "   • Values = similarity scores",
        "   • Diagonal = self-love (always highest)",
        
        "\n🎯 TIP 2: Kernel Choice = Similarity Definition",  
        "   • Linear: 'Are vectors pointing same way?'",
        "   • RBF: 'Are points close together?'",
        "   • Polynomial: 'Do they have similar patterns?'",
        
        "\n🎯 TIP 3: PSD Property = Mathematical Sanity Check",
        "   • If not PSD = negative 'friendship energy'",
        "   • Computers hate negative eigenvalues",
        "   • Always verify for custom kernels!",
        
        "\n🎯 TIP 4: Kernel Trick = Computational Cheat Code",
        "   • Skip expensive high-D computations",
        "   • Work with infinite dimensions (RBF kernel)",
        "   • Key to SVM's superpowers",
        
        "\n🎯 TIP 5: Regularization = Social Boundaries",
        "   • Prevents overfitting to noise",
        "   • Like not judging friendships on one interaction",
        "   • Add small value to diagonal of Gram matrix",
    ]
    
    for line in insights:
        print(line)

def main():
    """
    🚀 MAIN SHOW: Your journey from zero to kernel hero!
    
    Follow along with these levels:
    1. Basics: What's a Gram matrix?
    2. Magic: The kernel trick  
    3. Elevator: Feature spaces
    4. Battle: SVM application
    5. Wisdom: Practical tips
    """
    print("\n" + "🌟" * 60)
    print("WELCOME TO KERNEL KINGDOM!")
    print("Your journey from 'WTF is a kernel?' to 'I got this!'")
    print("🌟" * 60)
    
    print("\n📚 COURSE MAP:")
    print("1️⃣ Level 1: Gram Matrix Basics")
    print("2️⃣ Level 2: Kernel Trick Magic")
    print("3️⃣ Level 3: Feature Space Elevator")  
    print("4️⃣ Level 4: SVM Showdown")
    print("5️⃣ Level 5: Pro Tips")
    
    input("\nPress Enter to begin your adventure...")
    
    demonstrate_linear_gram()
    input("\n⏭️ Press Enter for Level 2...")
    
    demonstrate_kernel_trick()
    input("\n⏭️ Press Enter for Level 3...")
    
    feature_space_visualization()
    input("\n⏭️ Press Enter for Level 4...")
    
    svm_connection_demo()
    input("\n⏭️ Press Enter for Level 5...")
    
    practical_insights()
    
    print("\n" + "🎓" * 60)
    print("CONGRATULATIONS! YOU'VE MASTERED:")
    print("✓ Gram matrices = Data friendship networks")
    print("✓ Kernel trick = Computational shortcut magic")
    print("✓ Feature spaces = Pattern-revealing dimensions")
    print("✓ SVM connection = Real ML application")
    print("\nNext: Try different kernels on real datasets!")
    print("Check out sklearn's SVC class to use this in practice.")
    print("🎓" * 60)
    
    print("\n🔗 RESOURCES TO CONTINUE LEARNING:")
    print("• sklearn kernels: https://scikit-learn.org/stable/modules/metrics.html")
    print("• Interactive visualizer: https://www.youtube.com/watch?v=3liCbRZPrZA")
    print("• SVM tutorial: https://www.youtube.com/watch?v=efR1C6CvhmE")

if __name__ == "__main__":
    # Clear console and start fresh
    print("\033c", end="")  # Clears terminal (works on most systems)
    main()