"""
numerical_stability_demo.py

This educational module demonstrates numerical stability issues in machine learning
and provides stable alternatives. Perfect for understanding why stability tricks
are essential in ML pipelines.

Key concepts covered:
- Floating-point limitations (overflow/underflow)
- Stable implementations of log-sum-exp, softmax, and cross-entropy
- Visual demonstrations of failure modes and solutions

Designed by kira-ml for open-source ML education.
"""

from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

# ============================================================================
# SECTION 1: VISUALIZATION SETUP
# ============================================================================
# We configure plotting defaults for clear educational visuals
plt.style.use('default')
sns.set_palette("husl")  # Color-blind friendly palette
plt.rcParams['figure.figsize'] = [10, 6]  # Standard size for tutorials
plt.rcParams['font.size'] = 10  # Readable font size

# ============================================================================
# SECTION 2: FLOATING-POINT FUNDAMENTALS
# ============================================================================
# Understanding floating-point representation is crucial for debugging ML errors

def show_float_info() -> None:
    """
    Display key properties of float32 and float64 data types.
    
    Why this matters: ML uses float32 by default for efficiency, but it has
    limited range and precision compared to float64.
    """
    for dtype in (np.float32, np.float64):
        info = np.finfo(dtype)
        # Machine epsilon: smallest number that makes a difference when added to 1
        # Max: largest representable positive number
        # Tiny: smallest positive normalized number
        print(f"{dtype.__name__}: eps={info.eps:.3e}, max={info.max:.3e}, tiny={info.tiny:.3e}")


def visualize_float_limits() -> None:
    """
    Visual comparison of float32 vs float64 limits.
    
    Key insight: float32 has ~7 decimal digits precision, float64 has ~15.
    Overflow occurs when values exceed 'max', underflow when below 'tiny'.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Data for comparison
    dtypes = ['float32', 'float64']
    max_values = [np.finfo(np.float32).max, np.finfo(np.float64).max]
    min_values = [np.finfo(np.float32).tiny, np.finfo(np.float64).tiny]

    # Plot 1: Range comparison (log scale needed for huge differences)
    x_pos = np.arange(len(dtypes))
    ax1.bar(x_pos - 0.2, max_values, width=0.4, label='Max Value', alpha=0.8)
    ax1.bar(x_pos + 0.2, min_values, width=0.4, label='Min Positive Value', alpha=0.8)
    ax1.set_yscale('log')  # Log scale because values span 10^300 range!
    ax1.set_ylabel('Value (log scale)')
    ax1.set_title('Floating Point Range Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(dtypes)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Precision comparison (machine epsilon)
    eps_values = [np.finfo(np.float32).eps, np.finfo(np.float64).eps]
    ax2.bar(dtypes, eps_values, alpha=0.8, color='orange')
    ax2.set_yscale('log')
    ax2.set_ylabel('Machine Epsilon (log scale)')
    ax2.set_title('Precision Comparison (Machine Epsilon)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('float_limits_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_overflow_underflow_demo() -> None:
    """
    Visualize where exp(x) causes overflow and underflow.
    
    Critical for ML: Exponential functions appear in softmax, sigmoid, and 
    many probability calculations. They're prone to numerical issues.
    """
    # Wide range to show extreme behaviors
    x = np.linspace(-1000, 1000, 1000)
    exp_x = np.exp(x)  # This will produce inf for large x!

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Linear scale plot - shows overflow as vertical spikes to infinity
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

    # Log scale plot - reveals underflow as values dropping to zero
    ax2.plot(x, exp_x, 'b-', linewidth=2)
    ax2.axhline(y=np.finfo(np.float32).max, color='r', linestyle='--')
    ax2.axhline(y=np.finfo(np.float64).max, color='g', linestyle='--')
    ax2.set_yscale('log')  # Log scale helps visualize tiny values
    ax2.set_xlabel('x')
    ax2.set_ylabel('exp(x) - log scale')
    ax2.set_title('Exponential Function - Log Scale')
    ax2.grid(True, alpha=0.3)

    # Educational annotations
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

# ============================================================================
# SECTION 3: NUMERICALLY STABLE ALGORITHMS
# ============================================================================
# Core ML functions with both naive (unstable) and stable implementations

def naive_logsumexp(x: Any, axis: int = -1) -> np.ndarray:
    """
    NAIVE VERSION - Prone to overflow!
    
    Computes log(sum(exp(x))) directly. Problem: exp(x) can overflow to infinity
    when x > 709 for float64 or x > 88 for float32.
    
    Example failure: naive_logsumexp([1000, 1001, 1002]) → exp(1000) = inf
    """
    x = np.asarray(x)
    return np.log(np.sum(np.exp(x), axis=axis))


def stable_logsumexp(x: Any, axis: int = -1) -> np.ndarray:
    """
    STABLE VERSION - Uses the "max trick" to prevent overflow.
    
    Mathematical identity: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    By subtracting max(x) before exp(), we keep values in safe range.
    
    This pattern is used in TensorFlow and PyTorch internally!
    """
    x = np.asarray(x)
    # Find maximum along reduction axis (keep dimensions for broadcasting)
    x_max = np.max(x, axis=axis, keepdims=True)
    
    # Subtract max to prevent overflow, then compute safely
    shifted = x - x_max  # Now all values ≤ 0, so exp(shifted) ≤ 1
    s = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    
    # Add max back to get correct result
    return np.squeeze(x_max + s, axis=axis)


def naive_softmax(x: Any, axis: int = -1) -> np.ndarray:
    """
    NAIVE VERSION - Computes softmax directly.
    
    softmax(x_i) = exp(x_i) / sum(exp(x))
    Problem: Both numerator and denominator can overflow!
    """
    x = np.asarray(x)
    exps = np.exp(x)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def stable_softmax(x: Any, axis: int = -1) -> np.ndarray:
    """
    STABLE VERSION - Uses same max trick as logsumexp.
    
    softmax(x_i) = exp(x_i - max(x)) / sum(exp(x - max(x)))
    This is mathematically equivalent but numerically safe.
    """
    x = np.asarray(x)
    x_max = np.max(x, axis=axis, keepdims=True)
    shifted = x - x_max  # Now values ≤ 0, safe for exp()
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def naive_cross_entropy_from_probs(probs: Any, labels: Any) -> np.ndarray:
    """
    NAIVE VERSION - Computes cross-entropy directly.
    
    CE = -sum(labels * log(probs))
    Problem: log(0) = -inf if any probability is exactly 0.
    """
    return -np.sum(labels * np.log(probs), axis=-1)


def stable_cross_entropy_from_probs(probs: Any, labels: Any, epsilon: float = 1e-12) -> np.ndarray:
    """
    STABLE VERSION - Clips probabilities to avoid log(0).
    
    Best practice: Clip to [epsilon, 1-epsilon] to prevent:
    1. log(0) = -inf
    2. log(1) = 0 with potential precision issues
    
    Note: Choose epsilon small enough to not affect gradients significantly.
    """
    # Clip probabilities to safe range
    probs = np.clip(probs, epsilon, 1.0 - epsilon)
    return -np.sum(labels * np.log(probs), axis=-1)


def naive_cross_entropy_from_logits(logits: Any, labels: Any) -> np.ndarray:
    """
    NAIVE VERSION - Computes softmax then cross-entropy.
    
    This combines instability from both naive_softmax and naive_cross_entropy.
    Double trouble for numerical stability!
    """
    probs = naive_softmax(logits)
    return naive_cross_entropy_from_probs(probs, labels)


def stable_cross_entropy_from_logits(logits: Any, labels: Any) -> np.ndarray:
    """
    STABLE VERSION - Single-step computation using logsumexp trick.
    
    CE = logsumexp(logits) - sum(labels * logits)
    This avoids computing softmax explicitly, reducing numerical errors.
    
    Used in production libraries like PyTorch's CrossEntropyLoss.
    """
    log_sum_exp = stable_logsumexp(logits, axis=-1)
    weighted_logits = np.sum(labels * logits, axis=-1)
    return log_sum_exp - weighted_logits

# ============================================================================
# SECTION 4: TESTING AND VALIDATION
# ============================================================================
# Creating comprehensive test cases to demonstrate issues and solutions

def create_test_cases() -> List[Dict[str, Any]]:
    """
    Create test cases ranging from normal to extreme values.
    
    Good practice: Test your ML functions with:
    1. Normal values (typical use case)
    2. Extreme values (stress test)
    3. Edge cases (boundary conditions)
    """
    test_cases = []

    # Case 1: Normal values (typical ML scenario)
    test_cases.append({
        'name': 'Normal values',
        'logits': np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        'labels': np.array([[0.1, 0.3, 0.6]], dtype=np.float32)
    })

    # Case 2: Large positives (overflow territory for float32)
    test_cases.append({
        'name': 'Large positive values',
        'logits': np.array([[100.0, 101.0, 102.0]], dtype=np.float32),
        'labels': np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    })

    # Case 3: Very large positives (definite overflow)
    test_cases.append({
        'name': 'Very large positive values',
        'logits': np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32),
        'labels': np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    })

    # Case 4: Large negatives (underflow territory)
    test_cases.append({
        'name': 'Large negative values',
        'logits': np.array([[-1000.0, -1001.0, -1002.0]], dtype=np.float32),
        'labels': np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    })

    # Case 5: Mixed extremes (tests scaling behavior)
    test_cases.append({
        'name': 'Mixed large and small values',
        'logits': np.array([[-1000.0, 0.0, 1000.0]], dtype=np.float32),
        'labels': np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    })

    return test_cases


def compare_logsumexp_implementations() -> None:
    """
    Compare naive vs stable logsumexp across different input ranges.
    
    Educational goal: Show that naive fails for extreme values,
    while stable works for all cases.
    """
    test_ranges = [
        np.array([-1000, -999, -998]),  # Very negative (underflow risk)
        np.array([-10, -9, -8]),        # Negative (safe)
        np.array([0, 1, 2]),            # Around zero (safe)
        np.array([10, 11, 12]),         # Positive (safe)
        np.array([100, 101, 102])       # Large positive (overflow risk)
    ]

    results = {'naive': [], 'stable': [], 'range_label': []}

    for i, test_input in enumerate(test_ranges):
        range_label = f"Range {i+1}"
        results['range_label'].append(range_label)

        # Try naive version - may fail with overflow/underflow
        try:
            naive_result = naive_logsumexp(test_input)
            results['naive'].append(np.asarray(naive_result).item())
        except Exception:
            results['naive'].append(np.nan)  # Mark as failed

        # Stable version should always work
        stable_result = stable_logsumexp(test_input)
        results['stable'].append(np.asarray(stable_result).item())

    # Visualization of comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(results['range_label']))
    width = 0.35

    # Red bars for naive (may show NaN as missing)
    ax.bar(x_pos - width/2, results['naive'], width, label='Naive', alpha=0.7, color='red')
    # Green bars for stable (should all have values)
    ax.bar(x_pos + width/2, results['stable'], width, label='Stable', alpha=0.7, color='green')

    ax.set_xlabel('Input Range (increasing values)')
    ax.set_ylabel('logsumexp Result')
    ax.set_title('Naive vs Stable logsumexp Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(results['range_label'])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add value labels for educational clarity
    for i, v in enumerate(results['naive']):
        if not np.isnan(v):
            ax.text(i - width/2, v, f'{v:.2f}', ha='center', va='bottom')

    for i, v in enumerate(results['stable']):
        ax.text(i + width/2, v, f'{v:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('logsumexp_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_softmax_stability() -> None:
    """
    Visualize how softmax behaves as input magnitude increases.
    
    Shows: Stable implementation remains well-behaved, naive becomes unstable.
    """
    magnitudes = np.linspace(0, 1000, 50)
    naive_results = []
    stable_results = []

    # Test for increasing input magnitudes
    for mag in magnitudes:
        logits = np.array([[mag, mag + 1, mag + 2]])

        try:
            naive_probs = naive_softmax(logits)
            naive_results.append(naive_probs[0])
        except Exception:
            # Naive fails for large magnitudes
            naive_results.append([np.nan, np.nan, np.nan])

        stable_probs = stable_softmax(logits)
        stable_results.append(stable_probs[0])

    naive_results = np.array(naive_results)
    stable_results = np.array(stable_results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot probabilities for each class
    for i in range(3):
        ax1.plot(magnitudes, stable_results[:, i], label=f'Class {i+1} (stable)', linewidth=2)
        ax1.plot(magnitudes, naive_results[:, i], '--', label=f'Class {i+1} (naive)', alpha=0.7)

    ax1.set_xlabel('Input Magnitude')
    ax1.set_ylabel('Probability')
    ax1.set_title('Softmax Stability Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot errors (log scale shows orders of magnitude differences)
    errors = np.abs(naive_results - stable_results)
    for i in range(3):
        ax2.plot(magnitudes, errors[:, i], label=f'Class {i+1} error', linewidth=2)

    ax2.set_xlabel('Input Magnitude')
    ax2.set_ylabel('Absolute Error')
    ax2.set_yscale('log')  # Errors can be tiny or huge!
    ax2.set_title('Numerical Errors in Naive Softmax')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('softmax_stability.png', dpi=150, bbox_inches='tight')
    plt.show()


def run_comprehensive_tests() -> None:
    """
    Run all test cases and compare implementations.
    
    Educational goal: Show concrete examples where naive implementations fail
    and stable ones succeed.
    """
    test_cases = create_test_cases()

    for i, test_case in enumerate(test_cases):
        print(f"\n=== Test Case {i+1}: {test_case['name']} ===")
        logits = test_case['logits']
        labels = test_case['labels']

        print(f"Logits: {logits}")
        print(f"Labels: {labels}")

        # Test softmax
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

        # Test logsumexp
        try:
            naive_lse = naive_logsumexp(logits)
            print(f"Naive logsumexp: {naive_lse}")
        except Exception as e:
            print(f"Naive logsumexp failed: {e}")
            naive_lse = None

        stable_lse = stable_logsumexp(logits)
        print(f"Stable logsumexp: {stable_lse}")

        # Test cross-entropy variations
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

        # Assertions for valid cases
        if naive_probs is not None and not np.any(np.isnan(naive_probs)) and not np.any(np.isinf(naive_probs)):
            assert np.allclose(np.sum(naive_probs, axis=-1), 1.0, atol=1e-6), \
                "Naive softmax should sum to 1"

        assert np.allclose(np.sum(stable_probs, axis=-1), 1.0, atol=1e-6), \
            "Stable softmax should sum to 1"

        if naive_lse is not None and not np.isnan(naive_lse).any() and not np.isinf(naive_lse).any():
            assert np.allclose(np.asarray(naive_lse), np.asarray(stable_lse), atol=1e-6, rtol=1e-6), \
                "Logsumexp implementations should match for valid cases"

        print("✓ All assertions passed for this test case")


def demonstrate_failure_modes() -> None:
    """
    Explicitly demonstrate specific failure cases.
    
    Shows: What goes wrong, why it goes wrong, and how stable fixes help.
    """
    print("\n" + "="*60)
    print("DEMONSTRATING FAILURE MODES")
    print("="*60)

    # 1) Overflow example
    print("\n1. Overflow in naive softmax:")
    huge_logits = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32)
    print(f"Logits: {huge_logits}")
    print(f"exp(1000) would be ~10^434 - far beyond float32 max!")

    try:
        naive_probs = naive_softmax(huge_logits)
        print(f"Naive softmax: {naive_probs}")
    except Exception as e:
        print(f"Naive softmax failed: {e}")

    stable_probs = stable_softmax(huge_logits)
    print(f"Stable softmax: {stable_probs}")
    print(f"Stable softmax sum: {np.sum(stable_probs, axis=-1)}")

    # 2) Underflow example
    print("\n2. Underflow in naive logsumexp:")
    small_logits = np.array([[-1000.0, -1001.0, -1002.0]], dtype=np.float32)
    print(f"Logits: {small_logits}")
    print(f"exp(-1000) would be ~10^-434 - far below float32 minimum!")

    try:
        naive_lse = naive_logsumexp(small_logits)
        print(f"Naive logsumexp: {naive_lse}")
    except Exception as e:
        print(f"Naive logsumexp failed: {e}")

    stable_lse = stable_logsumexp(small_logits)
    print(f"Stable logsumexp: {stable_lse}")

    # 3) Precision comparison
    print("\n3. Precision differences between float32 and float64:")
    test_logits = np.array([[50.0, 51.0, 52.0]])
    print("Same computation, different precision:")

    for dtype in [np.float32, np.float64]:
        logits = test_logits.astype(dtype)
        stable_result = stable_logsumexp(logits)
        print(f"{dtype.__name__}: {stable_result}")

    # 4) Floating point info
    print("\n4. Floating point precision information:")
    show_float_info()

# ============================================================================
# SECTION 5: MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("🔬 Numerical Stability Educational Visualizations")
    print("="*50)
    print("By kira-ml - Making ML fundamentals accessible!")
    print()

    # Step-by-step educational journey
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
    print("\nKey takeaways:")
    print("1. Always use stable implementations in production ML code")
    print("2. Understand floating-point limitations for debugging")
    print("3. Test with extreme values to catch numerical issues early")