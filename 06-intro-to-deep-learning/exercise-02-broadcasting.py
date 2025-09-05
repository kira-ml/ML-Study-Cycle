import numpy as np
from time import perf_counter


def unsqueeze_to(tensor: np.ndarray, target_ndim: int) -> np.ndarray:
    """Add dimensions to tensor until it reaches target_ndim."""
    while tensor.ndim < target_ndim:
        tensor = np.expand_dims(tensor, axis=0)
    return tensor


def tile_like(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Tile source array to match the shape of reference array."""
    if source.ndim != reference.ndim:
        raise ValueError(f"Source and reference must have same number of dimensions. "
                         f"Got {source.ndim} and {reference.ndim}")
    
    reps = tuple(ref_dim // src_dim for ref_dim, src_dim in zip(reference.shape, source.shape))
    return np.tile(source, reps)


def fused_scale_shift(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Perform element-wise scaling and shifting: a * b + c."""
    return a * b + c


def permute_axes(tensor: np.ndarray, order: tuple) -> np.ndarray:
    """Reorder tensor axes according to the specified order."""
    return np.transpose(tensor, axes=order)


def loop_multiply_add(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Multiply and add using explicit loops."""
    result = np.zeros_like(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            result[i, j] = a[i, j] * b[j] + c[j]
    return result


def vectorized_multiply_add(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Multiply and add using vectorized operations."""
    return a * b + c


def main():
    """Main function demonstrating the operations."""
    print("=== Performance Comparison ===")
    N, C = 16, 32
    a = np.random.randn(N, C)
    b = np.random.randn(C)
    c = np.random.randn(C)

    start = perf_counter()
    out1 = loop_multiply_add(a, b, c)
    loop_time = perf_counter() - start

    start = perf_counter()
    out2 = vectorized_multiply_add(a, b, c)
    vec_time = perf_counter() - start

    print("Output equal:", np.allclose(out1, out2))
    print(f"Loop time: {loop_time:.6f}s, Vectorized time: {vec_time:.6f}s")
    
    # Demonstrate other functions
    print("\n=== Function Demonstrations ===")
    
    # unsqueeze_to
    x = np.array([1, 2, 3])
    y = unsqueeze_to(x, 3)
    print("Original shape:", x.shape)
    print("Unsqueezed shape:", y.shape)
    
    # tile_like
    src = np.array([[1], [2]])
    ref = np.array([[1, 2, 3], [4, 5, 6]])  # Fixed: was tuple, now array
    tiled = tile_like(src, ref)
    print("Source shape:", src.shape)
    print("Reference shape:", ref.shape)
    print("Tiled shape:", tiled.shape)
    print("Tiled array:\n", tiled)
    
    # fused_scale_shift
    a = np.array([[1, 2], [3, 4]])
    b = np.array([10, 20])
    c = np.array([1, 1])
    result = fused_scale_shift(a, b, c)
    print("Fused scale shift result:\n", result)
    
    # permute_axes
    x = np.random.randn(2, 3, 4)
    y = permute_axes(x, (2, 0, 1))  # Fixed: valid permutation
    print("Original shape:", x.shape)
    print("Permuted shape:", y.shape)
    
    # Additional verification
    print("\n=== Additional Verification ===")
    print("Dot product test:", np.dot(np.array([1, 2, 3]), np.array([4, 5, 6])))


if __name__ == "__main__":
    main()