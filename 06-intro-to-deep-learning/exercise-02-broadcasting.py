"""
NumPy Broadcasting Operations for Machine Learning

This module demonstrates fundamental NumPy operations that are essential for 
understanding broadcasting in machine learning contexts. It includes implementations
of common tensor operations such as dimension expansion, tiling, axis permutation,
and fused scale-shift operations.

The implementations contrast explicit loop-based approaches with vectorized 
NumPy operations to illustrate performance differences and best practices
in numerical computing for ML applications.

Key Concepts Demonstrated:
- Broadcasting rules and their applications
- Tensor shape manipulation for ML pipelines
- Performance implications of vectorization
- Common patterns in deep learning frameworks
"""

import numpy as np
from time import perf_counter


def unsqueeze_to(tensor: np.ndarray, target_ndim: int) -> np.ndarray:
    """Expand tensor dimensions to match target number of dimensions.
    
    This function adds new axes of size 1 at the beginning of the tensor
    until it reaches the desired number of dimensions. This pattern is
    commonly used in deep learning to align tensor shapes for broadcasting.
    
    Parameters
    ----------
    tensor : np.ndarray
        Input tensor to expand
    target_ndim : int
        Desired number of dimensions
        
    Returns
    -------
    np.ndarray
        Tensor with expanded dimensions (new axes have size 1)
        
    Examples
    --------
    >>> x = np.array([1, 2, 3])  # Shape: (3,)
    >>> unsqueeze_to(x, 3).shape
    (1, 1, 3)
    """
    # In ML frameworks like PyTorch/TensorFlow, this pattern is essential
    # for aligning batch dimensions or channel dimensions during operations
    while tensor.ndim < target_ndim:
        tensor = np.expand_dims(tensor, axis=0)
    return tensor


def tile_like(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Replicate source array to match the shape of reference array.
    
    This operation is fundamental in many ML scenarios where tensors need
    to be expanded to match compatible shapes for element-wise operations.
    The function validates dimension compatibility before tiling.
    
    Parameters
    ----------
    source : np.ndarray
        Array to be tiled (must have same number of dims as reference)
    reference : np.ndarray
        Target shape reference array
        
    Returns
    -------
    np.ndarray
        Tiled array matching reference shape
        
    Raises
    ------
    ValueError
        If source and reference have different number of dimensions
        
    Examples
    --------
    >>> src = np.array([[1], [2]])          # Shape: (2, 1)
    >>> ref = np.array([[1, 2], [3, 4]])    # Shape: (2, 2)
    >>> tile_like(src, ref)
    array([[1, 1],
           [2, 2]])
    """
    # Dimension validation prevents runtime shape mismatches that are
    # common sources of bugs in ML pipelines
    if source.ndim != reference.ndim:
        raise ValueError(f"Source and reference must have same number of dimensions. "
                         f"Got {source.ndim} and {reference.ndim}")
    
    # Calculate repetition factors for each dimension
    # This leverages NumPy's broadcasting rules: ref_dim must be divisible by src_dim
    reps = tuple(ref_dim // src_dim for ref_dim, src_dim in zip(reference.shape, source.shape))
    return np.tile(source, reps)


def fused_scale_shift(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Perform element-wise fused scaling and shifting operation: a * b + c.
    
    This represents a common pattern in neural networks (BatchNorm, LayerNorm)
    where inputs are scaled by one parameter and shifted by another. The
    vectorized implementation is significantly more efficient than loops.
    
    Parameters
    ----------
    a : np.ndarray
        Input tensor (typically activations)
    b : np.ndarray
        Scaling factors (typically learned parameters)
    c : np.ndarray
        Shift values (typically learned parameters)
        
    Returns
    -------
    np.ndarray
        Result of a * b + c with broadcasting applied
        
    Notes
    -----
    This operation benefits from NumPy's broadcasting, allowing parameter
    arrays (b, c) to be smaller than the input tensor (a) when appropriate.
    """
    # Vectorized operations leverage optimized BLAS routines under the hood,
    # providing orders of magnitude better performance than explicit loops
    return a * b + c


def permute_axes(tensor: np.ndarray, order: tuple) -> np.ndarray:
    """Reorder tensor axes according to the specified permutation.
    
    Axis permutation is crucial in deep learning for operations like
    converting between channel-first (NCHW) and channel-last (NHWC) formats,
    transposing weight matrices, or preparing data for specific operations.
    
    Parameters
    ----------
    tensor : np.ndarray
        Input tensor to permute
    order : tuple
        New axis order (e.g., (2, 0, 1) moves axis 2 to position 0)
        
    Returns
    -------
    np.ndarray
        Tensor with reordered axes
        
    Examples
    --------
    >>> x = np.random.randn(2, 3, 4)  # NHWC format
    >>> permute_axes(x, (0, 2, 1)).shape  # To NCHW-like format
    (2, 4, 3)
    """
    # np.transpose is the standard NumPy equivalent of PyTorch's permute
    # or TensorFlow's transpose operations
    return np.transpose(tensor, axes=order)


def loop_multiply_add(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Multiply and add using explicit loops (educational implementation).
    
    This explicit loop implementation demonstrates what vectorized operations
    abstract away. While pedagogically useful, it's significantly slower
    than NumPy's optimized routines and should never be used in practice.
    
    Parameters
    ----------
    a : np.ndarray
        2D input tensor of shape (N, C)
    b : np.ndarray
        1D scaling factors of shape (C,)
    c : np.ndarray
        1D shift values of shape (C,)
        
    Returns
    -------
    np.ndarray
        Result of a * b + c computed via explicit loops
    """
    # Initialize output array with same shape as input
    # In production code, we'd use np.empty for performance, but zeros_like
    # makes the intent clearer for educational purposes
    result = np.zeros_like(a)
    
    # Nested loops explicitly iterate over each element
    # This approach has O(N*C) Python interpreter overhead
    for i in range(a.shape[0]):      # Iterate over first dimension (N)
        for j in range(a.shape[1]):  # Iterate over second dimension (C)
            # Element-wise operation: scale by b[j] and shift by c[j]
            # This mimics how parameters might be applied per channel
            result[i, j] = a[i, j] * b[j] + c[j]
    return result


def vectorized_multiply_add(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Multiply and add using vectorized NumPy operations (production implementation).
    
    This implementation leverages NumPy's broadcasting to perform the same
    computation as loop_multiply_add but with optimized C-level operations.
    It's typically 10-100x faster and more readable than explicit loops.
    
    Parameters
    ----------
    a : np.ndarray
        2D input tensor of shape (N, C)
    b : np.ndarray
        1D scaling factors of shape (C,)
    c : np.ndarray
        1D shift values of shape (C,)
        
    Returns
    -------
    np.ndarray
        Result of a * b + c computed via broadcasting
    """
    # NumPy broadcasting automatically aligns b and c with the last
    # dimension of a, applying the operation element-wise
    # This is equivalent to the loop version but executed in optimized C code
    return a * b + c


def main():
    """Demonstrate and benchmark NumPy broadcasting operations.
    
    This function showcases the practical applications of the implemented
    operations and empirically validates the performance benefits of
    vectorization over explicit loops.
    """
    print("=== Performance Comparison: Loops vs Vectorization ===")
    
    # Typical ML batch size (N) and feature/channel count (C)
    N, C = 16, 32
    a = np.random.randn(N, C)  # Batch of samples
    b = np.random.randn(C)     # Per-channel scale parameters
    c = np.random.randn(C)     # Per-channel shift parameters

    # Benchmark loop-based implementation
    start = perf_counter()
    out1 = loop_multiply_add(a, b, c)
    loop_time = perf_counter() - start

    # Benchmark vectorized implementation
    start = perf_counter()
    out2 = vectorized_multiply_add(a, b, c)
    vec_time = perf_counter() - start

    print("Output equality check:", np.allclose(out1, out2))
    print(f"Loop implementation time: {loop_time:.6f}s")
    print(f"Vectorized implementation time: {vec_time:.6f}s")
    print(f"Performance improvement: {loop_time/vec_time:.1f}x faster")
    
    # Demonstrate tensor manipulation functions
    print("\n=== Tensor Shape Manipulation Functions ===")
    
    # unsqueeze_to: Adding dimensions for broadcasting compatibility
    x = np.array([1, 2, 3])
    y = unsqueeze_to(x, 3)
    print(f"Original shape: {x.shape}")
    print(f"Unsqueezed to 3D: {y.shape}")
    
    # tile_like: Replicating arrays to match target shapes
    src = np.array([[1], [2]])                    # Shape: (2, 1)
    ref = np.array([[1, 2, 3], [4, 5, 6]])       # Shape: (2, 3)
    tiled = tile_like(src, ref)
    print(f"Source shape: {src.shape}")
    print(f"Reference shape: {ref.shape}")
    print(f"Tiled result shape: {tiled.shape}")
    print("Tiled array:\n", tiled)
    
    # fused_scale_shift: Common ML operation pattern
    a = np.array([[1, 2], [3, 4]])
    b = np.array([10, 20])  # Scale factors
    c = np.array([1, 1])    # Shift values
    result = fused_scale_shift(a, b, c)
    print("Fused scale-shift result:\n", result)
    
    # permute_axes: Reordering tensor dimensions
    x = np.random.randn(2, 3, 4)  # Example: (batch, height, width)
    y = permute_axes(x, (2, 0, 1)) # Reorder to (width, batch, height)
    print(f"Original shape: {x.shape}")
    print(f"Permuted shape: {y.shape}")
    
    # Additional verification of basic operations
    print("\n=== Basic NumPy Operations Verification ===")
    dot_product = np.dot(np.array([1, 2, 3]), np.array([4, 5, 6]))
    print(f"Dot product test: {dot_product}")


if __name__ == "__main__":
    main()