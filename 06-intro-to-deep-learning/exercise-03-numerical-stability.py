import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

# Set up plotting style for better educational visuals
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 10


def show_float_info():
    """Display floating-point precision information for float32 and float64."""
    for dtype in (np.float32, np.float64):
        info = np.finfo(dtype)
        print(f"{dtype.__name__}: eps={info.eps:.3e}, max={info.max:.3e}, tiny={info.tiny:.3e}")


def visualize_float_limits():
    """Create educational visualization of floating point limits."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Float32 vs Float64 range comparison
    dtypes = ['float32', 'float64']
    max_values = [np.finfo(np.float32).max, np.finfo(np.float64).max]
    min_values = [np.finfo(np.float32).tiny, np.finfo(np.float64).tiny]
    
    x_pos = np.arange(len(dtypes))
    ax1.bar(x_pos - 0.2, max_values, width=0.4, label='Max Value', alpha=0.8)
    ax1.bar(x_pos + 0.2, min_values, width=0.4, label='Min Positive Value', alpha=0.8)
    ax1.set_yscale('log')
    ax1.set_ylabel('Value (log scale)')
    ax1.set_title('Floating Point Range Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(dtypes)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Epsilon comparison
    eps_values = [np.finfo(np.float32).eps, np.finfo(np.float64).eps]
    ax2.bar(dtypes, eps_values, alpha=0.8, color='orange')
    ax2.set_yscale('log')
    ax2.set_ylabel('Machine Epsilon (log scale)')
    ax2.set_title('Precision Comparison (Machine Epsilon)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('float_limits_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_overflow_underflow_demo():
    """Visualize where overflow and underflow occur for exponential function."""
    x = np.linspace(-1000, 1000, 1000)
    exp_x = np.exp(x)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Linear scale plot
    ax1.plot(x, exp_x, 'b-', linewidth=2)
    ax1.axhline(y=np.finfo(np.float32).max, color='r', linestyle='--', 
                label=f'float32 max ({np.finfo(np.float32).max:.1e})')
    ax1.axhline(y=np.finfo(np.float64).max, color='g', linestyle='--', 
                label=f'float64 max ({np.finfo(np.float64).max:.1e})')
    ax1.set_xlabel('x')
    ax1.set_ylabel('exp(x)')
    ax1.set_title('Exponential Function - Linear Scale')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Log scale plot
    ax2.plot(x, exp_x, 'b-', linewidth=2)
    ax2.axhline(y=np.finfo(np.float32).max, color='r', linestyle='--')
    ax2.axhline(y=np.finfo(np.float64).max, color='g', linestyle='--')
    ax2.set_yscale('log')
    ax2.set_xlabel('x')
    ax2.set_ylabel('exp(x) - log scale')
    ax2.set_title('Exponential Function - Log Scale')
    ax2.grid(True, alpha=0.3)
    
    # Add annotations
    ax1.annotate('OVERFLOW REGION\n(float32 breaks here)', 
                xy=(700, 1e200), xytext=(500, 1e100),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    ax1.annotate('UNDERFLOW REGION\n(values become 0)', 
                xy=(-700, 1e-300), xytext=(-400, 1e-200),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10, color='blue')
    
    plt.tight_layout()
    plt.savefig('overflow_underflow_demo.png', dpi=150, bbox_inches='tight')
    plt.show()


def naive_logsumexp(x, axis=-1):
    """Compute log-sum-exp in a numerically unstable way."""
    x = np.asarray(x)
    return np.log(np.sum(np.exp(x), axis=axis))


def stable_logsumexp(x, axis=-1):
    """Compute log-sum-exp in a numerically stable way."""
    x = np.asarray(x)
    x_max = np.max(x, axis=axis, keepdims=True)
    shifted = x - x_max
    s = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return np.squeeze(x_max + s, axis=axis)


def compare_logsumexp_implementations():
    """Compare naive vs stable logsumexp across different input ranges."""
    # Test across a range of values
    test_ranges = [np.array([-1000, -999, -998]),  # Very negative
                  np.array([-10, -9, -8]),        # Negative
                  np.array([0, 1, 2]),            # Around zero
                  np.array([10, 11, 12]),         # Positive
                  np.array([100, 101, 102])]      # Large positive
    
    results = {'naive': [], 'stable': [], 'range_label': []}
    
    for i, test_input in enumerate(test_ranges):
        range_label = f"Range {i+1}"
        results['range_label'].append(range_label)
        
        try:
            naive_result = naive_logsumexp(test_input)
            results['naive'].append(naive_result)
        except:
            results['naive'].append(np.nan)
        
        stable_result = stable_logsumexp(test_input)
        results['stable'].append(stable_result)
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(results['range_label']))
    
    width = 0.35
    ax.bar(x_pos - width/2, results['naive'], width, label='Naive', alpha=0.7, color='red')
    ax.bar(x_pos + width/2, results['stable'], width, label='Stable', alpha=0.7, color='green')
    
    ax.set_xlabel('Input Range (increasing values)')
    ax.set_ylabel('logsumexp Result')
    ax.set_title('Comparison of Naive vs Stable logsumexp')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(results['range_label'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(results['naive']):
        if not np.isnan(v):
            ax.text(i - width/2, v, f'{v:.2f}', ha='center', va='bottom')
    
    for i, v in enumerate(results['stable']):
        ax.text(i + width/2, v, f'{v:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('logsumexp_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def naive_softmax(x, axis=-1):
    """Compute softmax in a numerically unstable way."""
    x = np.asarray(x)
    exps = np.exp(x)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def stable_softmax(x, axis=-1):
    """Compute softmax in a numerically stable way."""
    x = np.asarray(x)
    x_max = np.max(x, axis=axis, keepdims=True)
    shifted = x - x_max
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def visualize_softmax_stability():
    """Visualize softmax stability across different input magnitudes."""
    magnitudes = np.linspace(0, 1000, 50)
    naive_results = []
    stable_results = []
    
    for mag in magnitudes:
        logits = np.array([[mag, mag+1, mag+2]])
        
        try:
            naive_probs = naive_softmax(logits)
            naive_results.append(naive_probs[0])
        except:
            naive_results.append([np.nan, np.nan, np.nan])
        
        stable_probs = stable_softmax(logits)
        stable_results.append(stable_probs[0])
    
    naive_results = np.array(naive_results)
    stable_results = np.array(stable_results)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot probabilities
    for i in range(3):
        ax1.plot(magnitudes, stable_results[:, i], label=f'Class {i+1} (stable)', linewidth=2)
        ax1.plot(magnitudes, naive_results[:, i], '--', label=f'Class {i+1} (naive)', alpha=0.7)
    
    ax1.set_xlabel('Input Magnitude')
    ax1.set_ylabel('Probability')
    ax1.set_title('Softmax Stability Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot errors
    errors = np.abs(naive_results - stable_results)
    for i in range(3):
        ax2.plot(magnitudes, errors[:, i], label=f'Class {i+1} error', linewidth=2)
    
    ax2.set_xlabel('Input Magnitude')
    ax2.set_ylabel('Absolute Error')
    ax2.set_yscale('log')
    ax2.set_title('Numerical Errors in Naive Softmax')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('softmax_stability.png', dpi=150, bbox_inches='tight')
    plt.show()


def naive_cross_entropy_from_probs(probs, labels):
    """Compute cross-entropy from probabilities in a numerically unstable way."""
    return -np.sum(labels * np.log(probs), axis=-1)


def stable_cross_entropy_from_probs(probs, labels, epsilon=1e-12):
    """Compute cross-entropy from probabilities in a numerically stable way."""
    # Clip probabilities to avoid log(0)
    probs = np.clip(probs, epsilon, 1.0 - epsilon)
    return -np.sum(labels * np.log(probs), axis=-1)


def naive_cross_entropy_from_logits(logits, labels):
    """Compute cross-entropy from logits in a numerically unstable way."""
    probs = naive_softmax(logits)
    return naive_cross_entropy_from_probs(probs, labels)


def stable_cross_entropy_from_logits(logits, labels):
    """Compute cross-entropy from logits in a numerically stable way."""
    # Use log-sum-exp trick: -sum(labels * logits) + log(sum(exp(logits)))
    log_sum_exp = stable_logsumexp(logits, axis=-1)
    weighted_logits = np.sum(labels * logits, axis=-1)
    return log_sum_exp - weighted_logits


def create_test_cases():
    """Create various test cases to demonstrate numerical stability issues."""
    test_cases = []
    
    # Normal case
    test_cases.append({
        'name': 'Normal values',
        'logits': np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        'labels': np.array([[0.1, 0.3, 0.6]], dtype=np.float32)
    })
    
    # Large positive values (causes overflow in naive implementations)
    test_cases.append({
        'name': 'Large positive values',
        'logits': np.array([[100.0, 101.0, 102.0]], dtype=np.float32),
        'labels': np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    })
    
    # Very large positive values (extreme overflow case)
    test_cases.append({
        'name': 'Very large positive values',
        'logits': np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32),
        'labels': np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    })
    
    # Large negative values (causes underflow in naive implementations)
    test_cases.append({
        'name': 'Large negative values',
        'logits': np.array([[-1000.0, -1001.0, -1002.0]], dtype=np.float32),
        'labels': np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    })
    
    # Mixed large and small values
    test_cases.append({
        'name': 'Mixed large and small values',
        'logits': np.array([[-1000.0, 0.0, 1000.0]], dtype=np.float32),
        'labels': np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    })
    
    return test_cases


def run_comprehensive_tests():
    """Run comprehensive tests comparing stable and unstable implementations."""
    test_cases = create_test_cases()
    
    for i, test_case in enumerate(test_cases):
        print(f"\n=== Test Case {i+1}: {test_case['name']} ===")
        logits = test_case['logits']
        labels = test_case['labels']
        
        print(f"Logits: {logits}")
        print(f"Labels: {labels}")
        
        # Test softmax implementations
        try:
            naive_probs = naive_softmax(logits)
            print(f"Naive softmax: {naive_probs}")
            print(f"Naive softmax sum: {np.sum(naive_probs, axis=-1)}")
        except Exception as e:
            print(f"Naive softmax failed: {e}")
            naive_probs = None
        
        stable_probs = stable_softmax(logits)
        print(f"Stable softmax: {stable_probs}")
        print(f"Stable softmax sum: {np.sum(stable_probs, axis=-1)}")
        
        # Test logsumexp implementations
        try:
            naive_lse = naive_logsumexp(logits)
            print(f"Naive logsumexp: {naive_lse}")
        except Exception as e:
            print(f"Naive logsumexp failed: {e}")
            naive_lse = None
        
        stable_lse = stable_logsumexp(logits)
        print(f"Stable logsumexp: {stable_lse}")
        
        # Test cross-entropy implementations
        if naive_probs is not None:
            try:
                naive_ce_probs = naive_cross_entropy_from_probs(naive_probs, labels)
                print(f"Naive CE from probs: {naive_ce_probs}")
            except Exception as e:
                print(f"Naive CE from probs failed: {e}")
        
        try:
            stable_ce_probs = stable_cross_entropy_from_probs(stable_probs, labels)
            print(f"Stable CE from probs: {stable_ce_probs}")
        except Exception as e:
            print(f"Stable CE from probs failed: {e}")
        
        try:
            naive_ce_logits = naive_cross_entropy_from_logits(logits, labels)
            print(f"Naive CE from logits: {naive_ce_logits}")
        except Exception as e:
            print(f"Naive CE from logits failed: {e}")
        
        stable_ce_logits = stable_cross_entropy_from_logits(logits, labels)
        print(f"Stable CE from logits: {stable_ce_logits}")
        
        # Add assertions for valid cases
        if naive_probs is not None and not np.any(np.isnan(naive_probs)) and not np.any(np.isinf(naive_probs)):
            assert np.allclose(np.sum(naive_probs, axis=-1), 1.0, atol=1e-6), "Naive softmax should sum to 1"
        
        assert np.allclose(np.sum(stable_probs, axis=-1), 1.0, atol=1e-6), "Stable softmax should sum to 1"
        
        if naive_lse is not None and not np.isnan(naive_lse) and not np.isinf(naive_lse):
            assert np.allclose(naive_lse, stable_lse, atol=1e-6, rtol=1e-6), "Logsumexp implementations should match for valid cases"
        
        print("✓ All assertions passed for this test case")


def demonstrate_failure_modes():
    """Specifically demonstrate failure modes of naive implementations."""
    print("\n" + "="*60)
    print("DEMONSTRATING FAILURE MODES")
    print("="*60)
    
    # Demonstrate overflow in naive softmax
    print("\n1. Overflow in naive softmax:")
    huge_logits = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32)
    print(f"Logits: {huge_logits}")
    
    try:
        naive_probs = naive_softmax(huge_logits)
        print(f"Naive softmax: {naive_probs}")
    except Exception as e:
        print(f"Naive softmax failed: {e}")
    
    stable_probs = stable_softmax(huge_logits)
    print(f"Stable softmax: {stable_probs}")
    print(f"Stable softmax sum: {np.sum(stable_probs, axis=-1)}")
    
    # Demonstrate underflow in naive logsumexp
    print("\n2. Underflow in naive logsumexp:")
    small_logits = np.array([[-1000.0, -1001.0, -1002.0]], dtype=np.float32)
    print(f"Logits: {small_logits}")
    
    try:
        naive_lse = naive_logsumexp(small_logits)
        print(f"Naive logsumexp: {naive_lse}")
    except Exception as e:
        print(f"Naive logsumexp failed: {e}")
    
    stable_lse = stable_logsumexp(small_logits)
    print(f"Stable logsumexp: {stable_lse}")
    
    # Demonstrate precision differences between float32 and float64
    print("\n3. Precision differences between float32 and float64:")
    test_logits = np.array([[50.0, 51.0, 52.0]])
    
    for dtype in [np.float32, np.float64]:
        logits = test_logits.astype(dtype)
        stable_result = stable_logsumexp(logits)
        print(f"{dtype.__name__}: {stable_result}")
    
    # Show floating point info
    print("\n4. Floating point precision information:")
    show_float_info()


if __name__ == "__main__":
    print("🔬 Numerical Stability Educational Visualizations")
    print("="*50)
    
    # Run visualizations first for educational purposes
    print("\n1. Visualizing Floating Point Limits...")
    visualize_float_limits()
    
    print("\n2. Demonstrating Overflow and Underflow...")
    plot_overflow_underflow_demo()
    
    print("\n3. Comparing logsumexp Implementations...")
    compare_logsumexp_implementations()
    
    print("\n4. Analyzing Softmax Stability...")
    visualize_softmax_stability()
    
    print("\n5. Running Comprehensive Tests...")
    run_comprehensive_tests()
    
    print("\n6. Demonstrating Specific Failure Modes...")
    demonstrate_failure_modes()
    
    print("\n🎓 Educational visualizations completed!")
    print("Check the generated PNG files for detailed visual explanations.")