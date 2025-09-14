"""
numerical_stability_demo.py

In this implementation I demonstrate common numerical stability issues that arise
when working with floating-point numbers in machine learning pipelines, and I
provide numerically stable alternatives. The code is structured as an
educational module that:

- Shows floating-point limits for float32 and float64.
- Visualizes overflow and underflow regions for the exponential function.
- Compares naive vs stable implementations of log-sum-exp, softmax, and
  cross-entropy.
- Runs a suite of test cases and explicitly demonstrates failure modes.

I wrote this to be used as an instructional script for engineers and
researchers who want to understand *why* stability tricks (like the
log-sum-exp shift) are necessary and *how* to apply them safely in both
research and production ML code.

Design notes (why I organized the module this way)
- Functions are kept small and single-purpose so they can be reused in larger
  training/evaluation pipelines or unit tests.
- Stable implementations follow common numerical engineering patterns:
    * subtracting the maximum (for log-sum-exp / softmax)
    * clipping probabilities before log (for cross-entropy)
  These are standard, battle-tested patterns used in production ML libraries.
- Visualizations are separate from computation so the computational functions
  can be unit-tested independently of plotting.
"""

from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

# ------------ Global plotting configuration (educational visuals) -------------
# I set a neutral style and a qualitative palette to make plots readable and
# consistent for teaching. These global settings are fine for notebooks or
# demo scripts — in production dashboards you may want to set styles locally.
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 10


# ----------------------------- Utility functions ------------------------------
def show_float_info() -> None:
    """
    Print machine floating-point characteristics for float32 and float64.

    Notes
    -----
    I print machine epsilon, maximum representable value, and the smallest
    positive normal value (tiny). These values help explain why very large
    exponentials overflow and very small exponentials underflow.
    """
    for dtype in (np.float32, np.float64):
        info = np.finfo(dtype)
        # Using scientific formatting to make magnitude differences clear.
        print(f"{dtype.__name__}: eps={info.eps:.3e}, max={info.max:.3e}, tiny={info.tiny:.3e}")


# ------------------------------- Visualizations -------------------------------
def visualize_float_limits() -> None:
    """
    Create a two-panel plot that visualizes floating-point ranges and epsilon.

    The left panel compares the maximum representable value and smallest positive
    normal (tiny) value between float32 and float64 on a log scale. The right
    panel compares machine epsilon for both dtypes.

    This function saves a PNG and displays the figure. The plotting code is
    separated from the numeric computations to make unit testing easy.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Collect basic numeric facts for comparison.
    dtypes = ['float32', 'float64']
    max_values = [np.finfo(np.float32).max, np.finfo(np.float64).max]
    min_values = [np.finfo(np.float32).tiny, np.finfo(np.float64).tiny]

    # Bar plot: max vs min positive (tiny). We use log scale to compress orders
    # of magnitude differences into a readable plot.
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

    # Machine epsilon comparison: shows relative precision difference.
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
    Plot exp(x) on linear and log scales to show where overflow and underflow occur.

    I generate exp(x) over a wide domain and draw horizontal lines for the max
    representable floats. Annotations are added to highlight where float32
    would overflow/underflow. This visual is pedagogical — it intentionally
    uses an extreme x-range to make the phenomenon explicit.
    """
    # Wide range to show extreme behaviors; note that exp(1000) is astronomically
    # large and will overflow to inf for float64 in many environments.
    x = np.linspace(-1000, 1000, 1000)
    exp_x = np.exp(x)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Linear scale: the plot will go to inf; axis limits are left automatic to
    # emphasize overflow visually (but we draw reference lines).
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

    # Log scale: allows us to see multiplicative differences without clipping.
    ax2.plot(x, exp_x, 'b-', linewidth=2)
    ax2.axhline(y=np.finfo(np.float32).max, color='r', linestyle='--')
    ax2.axhline(y=np.finfo(np.float64).max, color='g', linestyle='--')
    ax2.set_yscale('log')
    ax2.set_xlabel('x')
    ax2.set_ylabel('exp(x) - log scale')
    ax2.set_title('Exponential Function - Log Scale')
    ax2.grid(True, alpha=0.3)

    # Annotations: these are approximate visual markers (not computed crossing
    # points). They illustrate the regions where underflow/overflow are observed.
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


# ---------------------- Numerical algorithms (stable vs naive) -----------------
def naive_logsumexp(x: Any, axis: int = -1) -> np.ndarray:
    """
    Numerically *unstable* log-sum-exp.

    Parameters
    ----------
    x : array-like
        Input array of values (e.g., logits).
    axis : int, optional
        Axis along which to compute log-sum-exp.

    Returns
    -------
    ndarray
        log(sum(exp(x), axis=axis))

    Warning
    -------
    This naive form can overflow when elements of `x` are large (e.g., > 700 for
    float64) because exp(x) grows extremely rapidly.
    """
    x = np.asarray(x)
    return np.log(np.sum(np.exp(x), axis=axis))


def stable_logsumexp(x: Any, axis: int = -1) -> np.ndarray:
    """
    Numerically stable log-sum-exp implementation using the max-shift trick.

    Parameters
    ----------
    x : array-like
        Input array of values (e.g., logits).
    axis : int, optional
        Axis along which to compute log-sum-exp. Default is the last axis.

    Returns
    -------
    ndarray
        log(sum(exp(x), axis=axis))

    Notes
    -----
    I subtract the max along the reduction axis before exponentiating. This
    prevents exp from overflowing while preserving the correct result
    algebraically. The `keepdims=True` pattern ensures correct broadcasting when
    we add the max back.
    """
    x = np.asarray(x)
    # keepdims=True so we can add x_max back to the reduced log-sum easily.
    x_max = np.max(x, axis=axis, keepdims=True)
    shifted = x - x_max
    # Compute sum(exp(shifted)) in the numerically safe regime.
    s = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    # Squeeze only the reduced axis to return an array with expected dimensions.
    return np.squeeze(x_max + s, axis=axis)


def naive_softmax(x: Any, axis: int = -1) -> np.ndarray:
    """
    Naive softmax computed directly from exponentials.

    Parameters
    ----------
    x : array-like
        Logits or pre-softmax scores.
    axis : int, optional
        Axis along which to compute softmax.

    Returns
    -------
    ndarray
        Probabilities that sum to 1 along `axis` in ideal (non-overflowing) cases.

    Warning
    -------
    This implementation is susceptible to overflow when `x` contains large values.
    """
    x = np.asarray(x)
    exps = np.exp(x)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def stable_softmax(x: Any, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax using the max-shift trick.

    Subtracting the max reduces the dynamic range before exponentiation, which
    avoids overflow and yields identical results in exact arithmetic.

    Parameters
    ----------
    x : array-like
        Logits or pre-softmax scores.
    axis : int, optional
        Axis along which to compute softmax.

    Returns
    -------
    ndarray
        Numerically stable softmax probabilities.
    """
    x = np.asarray(x)
    x_max = np.max(x, axis=axis, keepdims=True)
    shifted = x - x_max
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def naive_cross_entropy_from_probs(probs: Any, labels: Any) -> np.ndarray:
    """
    Naive cross-entropy computed directly from predicted probabilities.

    Parameters
    ----------
    probs : array-like
        Predicted probabilities (should sum to 1 along the last axis).
    labels : array-like
        One-hot or soft labels with same shape as probs.

    Returns
    -------
    ndarray
        Per-example cross-entropy losses.

    Warning
    -------
    If `probs` contains zeros, log(0) will produce -inf and break training.
    """
    return -np.sum(labels * np.log(probs), axis=-1)


def stable_cross_entropy_from_probs(probs: Any, labels: Any, epsilon: float = 1e-12) -> np.ndarray:
    """
    Stable cross-entropy from probabilities by clipping `probs` to avoid log(0).

    Parameters
    ----------
    probs : array-like
        Predicted probabilities.
    labels : array-like
        One-hot or soft labels.
    epsilon : float, optional
        Small constant to clip probabilities into (epsilon, 1-epsilon).

    Returns
    -------
    ndarray
        Per-example cross-entropy losses.

    Notes
    -----
    Clipping is a simple and effective guard in production pipelines, but one
    should choose `epsilon` small enough not to bias gradients noticeably.
    """
    # Clip to (epsilon, 1 - epsilon) to avoid log(0) and to keep probabilities valid.
    probs = np.clip(probs, epsilon, 1.0 - epsilon)
    return -np.sum(labels * np.log(probs), axis=-1)


def naive_cross_entropy_from_logits(logits: Any, labels: Any) -> np.ndarray:
    """
    Compute cross-entropy by converting logits to probabilities using naive softmax.

    This compounds the instability of naive softmax (and naive cross-entropy).
    """
    probs = naive_softmax(logits)
    return naive_cross_entropy_from_probs(probs, labels)


def stable_cross_entropy_from_logits(logits: Any, labels: Any) -> np.ndarray:
    """
    Stable cross-entropy computed directly from logits using the log-sum-exp trick.

    The expression used:
        cross_entropy = logsumexp(logits) - sum(labels * logits)

    This is algebraically equivalent to -sum(labels * log(softmax(logits)))
    but avoids explicitly computing softmax, giving better numerical stability.
    """
    # stable_logsumexp returns shape with reduced axis removed.
    log_sum_exp = stable_logsumexp(logits, axis=-1)
    weighted_logits = np.sum(labels * logits, axis=-1)
    return log_sum_exp - weighted_logits


# ------------------------------- Test harness --------------------------------
def create_test_cases() -> List[Dict[str, Any]]:
    """
    Create a variety of test cases illustrating normal and pathological inputs.

    Returns
    -------
    list of dict
        Each dict contains 'name', 'logits', and 'labels'. The labels are
        intentionally soft/one-hot to show differences across different inputs.
    """
    test_cases = []

    # Normal case (small logits)
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

    # Mixed large and small values (stress test for relative scaling)
    test_cases.append({
        'name': 'Mixed large and small values',
        'logits': np.array([[-1000.0, 0.0, 1000.0]], dtype=np.float32),
        'labels': np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    })

    return test_cases


def compare_logsumexp_implementations() -> None:
    """
    Compare naive and stable log-sum-exp implementations across several ranges.

    The function catches exceptions in the naive implementation and records NaNs
    for failed runs so that visualization still works. It saves and displays a
    bar plot comparing results.
    """
    test_ranges = [
        np.array([-1000, -999, -998]),  # Very negative
        np.array([-10, -9, -8]),        # Negative
        np.array([0, 1, 2]),            # Around zero
        np.array([10, 11, 12]),         # Positive
        np.array([100, 101, 102])       # Large positive
    ]

    results = {'naive': [], 'stable': [], 'range_label': []}

    for i, test_input in enumerate(test_ranges):
        range_label = f"Range {i+1}"
        results['range_label'].append(range_label)

        # Naive may overflow/produce inf; catch any exceptions for the demo.
        try:
            naive_result = naive_logsumexp(test_input)
            results['naive'].append(np.asarray(naive_result).item())  # store scalar
        except Exception:
            results['naive'].append(np.nan)

        stable_result = stable_logsumexp(test_input)
        results['stable'].append(np.asarray(stable_result).item())

    # Plot comparison (bar chart). Using log-scaling here would hide some small
    # values; since values can span many orders, the demo uses linear values
    # and expects NaNs for failed naive computations.
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

    # Add value labels on bars where values are finite.
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
    Visualize softmax probabilities from naive and stable implementations
    as input magnitude increases.

    This demonstrates that the naive implementation becomes numerically
    unreliable for very large inputs while the stable implementation
    remains well-behaved.
    """
    magnitudes = np.linspace(0, 1000, 50)
    naive_results = []
    stable_results = []

    # For each magnitude, construct a simple 3-class logits vector that is
    # offset by the magnitude to stress the dynamic range.
    for mag in magnitudes:
        logits = np.array([[mag, mag + 1, mag + 2]])

        try:
            naive_probs = naive_softmax(logits)
            naive_results.append(naive_probs[0])
        except Exception:
            # When naive computation fails, store NaNs so plotting stays consistent.
            naive_results.append([np.nan, np.nan, np.nan])

        stable_probs = stable_softmax(logits)
        stable_results.append(stable_probs[0])

    naive_results = np.array(naive_results)
    stable_results = np.array(stable_results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot the probabilities for each class. Stable lines should remain smooth.
    for i in range(3):
        ax1.plot(magnitudes, stable_results[:, i], label=f'Class {i+1} (stable)', linewidth=2)
        ax1.plot(magnitudes, naive_results[:, i], '--', label=f'Class {i+1} (naive)', alpha=0.7)

    ax1.set_xlabel('Input Magnitude')
    ax1.set_ylabel('Probability')
    ax1.set_title('Softmax Stability Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot the absolute error between naive and stable implementations on a log y-scale.
    errors = np.abs(naive_results - stable_results)
    for i in range(3):
        ax2.plot(magnitudes, errors[:, i], label=f'Class {i+1} error', linewidth=2)

    ax2.set_xlabel('Input Magnitude')
    ax2.set_ylabel('Absolute Error')
    ax2.set_yscale('log')  # errors can span many orders of magnitude
    ax2.set_title('Numerical Errors in Naive Softmax')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('softmax_stability.png', dpi=150, bbox_inches='tight')
    plt.show()


def run_comprehensive_tests() -> None:
    """
    Execute a suite of tests comparing naive vs stable implementations.

    For each test case this function:
    - Prints the inputs.
    - Attempts naive computations and records failures.
    - Computes stable counterparts and prints them.
    - Performs assertions when results are finite to check expected equalities.

    Notes for production:
    - In production test suites replace prints with logging and make assertions
      raise test failures rather than halting scripts.
    """
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

        # Assertions: only run them when the numeric values are finite to avoid
        # raising errors for intentionally pathological inputs in this demo.
        if naive_probs is not None and not np.any(np.isnan(naive_probs)) and not np.any(np.isinf(naive_probs)):
            assert np.allclose(np.sum(naive_probs, axis=-1), 1.0, atol=1e-6), \
                "Naive softmax should sum to 1"

        assert np.allclose(np.sum(stable_probs, axis=-1), 1.0, atol=1e-6), \
            "Stable softmax should sum to 1"

        if naive_lse is not None and not np.isnan(naive_lse).any() and not np.isinf(naive_lse).any():
            # When both are finite, they should match up to numerical tolerance.
            assert np.allclose(np.asarray(naive_lse), np.asarray(stable_lse), atol=1e-6, rtol=1e-6), \
                "Logsumexp implementations should match for valid cases"

        print("✓ All assertions passed for this test case")


def demonstrate_failure_modes() -> None:
    """
    Explicitly demonstrate failure modes of naive implementations with prints.

    This function runs targeted examples showing:
    - Overflow in naive softmax (huge positive logits).
    - Underflow in naive log-sum-exp (very negative logits).
    - Precision differences between float32 and float64.
    - A summary of floating-point info.
    """
    print("\n" + "="*60)
    print("DEMONSTRATING FAILURE MODES")
    print("="*60)

    # 1) Overflow in naive softmax
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

    # 2) Underflow in naive logsumexp
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

    # 3) Precision differences between float32 and float64
    print("\n3. Precision differences between float32 and float64:")
    test_logits = np.array([[50.0, 51.0, 52.0]])

    for dtype in [np.float32, np.float64]:
        logits = test_logits.astype(dtype)
        stable_result = stable_logsumexp(logits)
        # This helps the reader see how dtype affects result precision.
        print(f"{dtype.__name__}: {stable_result}")

    # 4) Floating point info
    print("\n4. Floating point precision information:")
    show_float_info()


# ---------------------------------- Runner -----------------------------------
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
