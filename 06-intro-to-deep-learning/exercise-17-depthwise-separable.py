import numpy as np
import time
from typing import Tuple, Dict

class StandardConv2D:
    """Standard 2D Convolution"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights (out_channels, in_channels, kernel_size, kernel_size)
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.bias = np.zeros(out_channels)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for standard convolution"""
        batch_size, in_h, in_w, in_c = x.shape
        
        # Apply padding
        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (self.padding, self.padding), 
                                 (self.padding, self.padding), (0, 0)), mode='constant')
        else:
            x_padded = x
        
        # Calculate output dimensions
        out_h = (in_h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (in_w + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, out_h, out_w, self.out_channels))
        
        # Perform convolution
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for h in range(out_h):
                    for w in range(out_w):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        
                        # Extract patch and perform convolution
                        patch = x_padded[b, h_start:h_end, w_start:w_end, :]
                        output[b, h, w, oc] = np.sum(patch * self.weights[oc]) + self.bias[oc]
        
        return output
    
    def get_parameters(self) -> int:
        """Get number of parameters"""
        return np.prod(self.weights.shape) + len(self.bias)
    
    def get_flops(self, input_shape: Tuple[int, int, int]) -> int:
        """Calculate theoretical FLOPs for one forward pass"""
        batch_size, h, w, c = input_shape
        out_h = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (w + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Multiplications: kernel_size^2 * in_channels per output position
        # Additions: (kernel_size^2 * in_channels - 1) + 1 (bias) per output position
        flops_per_position = (self.kernel_size * self.kernel_size * self.in_channels) * 2
        
        total_flops = batch_size * out_h * out_w * self.out_channels * flops_per_position
        return total_flops

class DepthwiseConv2D:
    """Depthwise Convolution - applies a separate filter to each input channel"""
    
    def __init__(self, in_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0):
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights (in_channels, kernel_size, kernel_size)
        scale = np.sqrt(2.0 / (kernel_size * kernel_size))
        self.weights = np.random.randn(in_channels, kernel_size, kernel_size) * scale
        self.bias = np.zeros(in_channels)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for depthwise convolution"""
        batch_size, in_h, in_w, in_c = x.shape
        
        # Apply padding
        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (self.padding, self.padding), 
                                 (self.padding, self.padding), (0, 0)), mode='constant')
        else:
            x_padded = x
        
        # Calculate output dimensions
        out_h = (in_h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (in_w + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, out_h, out_w, self.in_channels))
        
        # Perform depthwise convolution
        for b in range(batch_size):
            for c in range(self.in_channels):
                for h in range(out_h):
                    for w in range(out_w):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        
                        # Extract patch for this channel only
                        patch = x_padded[b, h_start:h_end, w_start:w_end, c]
                        output[b, h, w, c] = np.sum(patch * self.weights[c]) + self.bias[c]
        
        return output
    
    def get_parameters(self) -> int:
        """Get number of parameters"""
        return np.prod(self.weights.shape) + len(self.bias)
    
    def get_flops(self, input_shape: Tuple[int, int, int]) -> int:
        """Calculate theoretical FLOPs for one forward pass"""
        batch_size, h, w, c = input_shape
        out_h = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (w + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Multiplications: kernel_size^2 per output position per channel
        # Additions: (kernel_size^2 - 1) + 1 (bias) per output position per channel
        flops_per_position = (self.kernel_size * self.kernel_size) * 2
        
        total_flops = batch_size * out_h * out_w * self.in_channels * flops_per_position
        return total_flops

class PointwiseConv2D:
    """Pointwise Convolution (1x1 convolution)"""
    
    def __init__(self, in_channels: int, out_channels: int):
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Initialize weights (out_channels, in_channels, 1, 1)
        scale = np.sqrt(2.0 / in_channels)
        self.weights = np.random.randn(out_channels, in_channels) * scale
        self.bias = np.zeros(out_channels)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for pointwise convolution"""
        batch_size, h, w, in_c = x.shape
        
        # Initialize output
        output = np.zeros((batch_size, h, w, self.out_channels))
        
        # Perform pointwise convolution (matrix multiplication across channels)
        for b in range(batch_size):
            for i in range(h):
                for j in range(w):
                    # For each spatial position, compute dot product across channels
                    output[b, i, j, :] = np.dot(x[b, i, j, :], self.weights.T) + self.bias
        
        return output
    
    def get_parameters(self) -> int:
        """Get number of parameters"""
        return np.prod(self.weights.shape) + len(self.bias)
    
    def get_flops(self, input_shape: Tuple[int, int, int]) -> int:
        """Calculate theoretical FLOPs for one forward pass"""
        batch_size, h, w, c = input_shape
        
        # Multiplications: in_channels per output channel per position
        # Additions: (in_channels - 1) + 1 (bias) per output position
        flops_per_position = self.in_channels * 2
        
        total_flops = batch_size * h * w * self.out_channels * flops_per_position
        return total_flops

class DepthwiseSeparableConv2D:
    """Depthwise Separable Convolution - combines depthwise and pointwise conv"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Depthwise convolution
        self.depthwise = DepthwiseConv2D(in_channels, kernel_size, stride, padding)
        
        # Pointwise convolution
        self.pointwise = PointwiseConv2D(in_channels, out_channels)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for depthwise separable convolution"""
        # Apply depthwise convolution
        depthwise_out = self.depthwise.forward(x)
        
        # Apply pointwise convolution
        output = self.pointwise.forward(depthwise_out)
        
        return output
    
    def get_parameters(self) -> int:
        """Get total number of parameters"""
        return self.depthwise.get_parameters() + self.pointwise.get_parameters()
    
    def get_flops(self, input_shape: Tuple[int, int, int]) -> int:
        """Calculate total FLOPs"""
        depthwise_flops = self.depthwise.get_flops(input_shape)
        
        # Calculate output shape after depthwise conv
        batch_size, h, w, c = input_shape
        out_h = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (w + 2 * self.padding - self.kernel_size) // self.stride + 1
        depthwise_output_shape = (batch_size, out_h, out_w, self.in_channels)
        
        pointwise_flops = self.pointwise.get_flops(depthwise_output_shape)
        
        return depthwise_flops + pointwise_flops

def compare_operations():
    """Compare standard conv vs depthwise separable conv"""
    print("Comparison: Standard Conv vs Depthwise Separable Conv")
    print("=" * 60)
    
    # Test configuration
    batch_size = 2
    input_shape = (32, 32, 64)  # (height, width, channels)
    kernel_size = 3
    stride = 1
    padding = 1
    out_channels = 128
    
    # Create sample input
    x = np.random.randn(batch_size, *input_shape)
    print(f"Input shape: {x.shape}")
    
    # Standard convolution
    print("\n1. Standard Convolution:")
    standard_conv = StandardConv2D(input_shape[2], out_channels, kernel_size, stride, padding)
    
    start_time = time.time()
    standard_output = standard_conv.forward(x)
    standard_time = time.time() - start_time
    
    standard_params = standard_conv.get_parameters()
    standard_flops = standard_conv.get_flops((batch_size, *input_shape))
    
    print(f"   Output shape: {standard_output.shape}")
    print(f"   Parameters: {standard_params:,}")
    print(f"   FLOPs: {standard_flops:,}")
    print(f"   Forward time: {standard_time:.4f}s")
    
    # Depthwise separable convolution
    print("\n2. Depthwise Separable Convolution:")
    separable_conv = DepthwiseSeparableConv2D(input_shape[2], out_channels, kernel_size, stride, padding)
    
    start_time = time.time()
    separable_output = separable_conv.forward(x)
    separable_time = time.time() - start_time
    
    separable_params = separable_conv.get_parameters()
    separable_flops = separable_conv.get_flops((batch_size, *input_shape))
    
    print(f"   Output shape: {separable_output.shape}")
    print(f"   Parameters: {separable_params:,}")
    print(f"   FLOPs: {separable_flops:,}")
    print(f"   Forward time: {separable_time:.4f}s")
    
    # Individual components
    print("\n3. Individual Components:")
    depthwise_conv = DepthwiseConv2D(input_shape[2], kernel_size, stride, padding)
    pointwise_conv = PointwiseConv2D(input_shape[2], out_channels)
    
    depthwise_params = depthwise_conv.get_parameters()
    pointwise_params = pointwise_conv.get_parameters()
    
    print(f"   Depthwise conv parameters: {depthwise_params:,}")
    print(f"   Pointwise conv parameters: {pointwise_params:,}")
    print(f"   Total separable parameters: {depthwise_params + pointwise_params:,}")
    
    # Comparison metrics
    print("\n4. Comparison Metrics:")
    param_ratio = separable_params / standard_params
    flops_ratio = separable_flops / standard_flops
    speed_ratio = separable_time / standard_time
    
    print(f"   Parameter reduction: {1/param_ratio:.2f}x fewer parameters")
    print(f"   FLOPs reduction: {1/flops_ratio:.2f}x fewer FLOPs")
    print(f"   Speed improvement: {1/speed_ratio:.2f}x faster")
    print(f"   Memory efficiency: {param_ratio:.2%} of standard conv parameters")
    print(f"   Compute efficiency: {flops_ratio:.2%} of standard conv FLOPs")
    
    return {
        'standard': {'output': standard_output, 'params': standard_params, 'flops': standard_flops, 'time': standard_time},
        'separable': {'output': separable_output, 'params': separable_params, 'flops': separable_flops, 'time': separable_time}
    }

def test_different_configurations():
    """Test with different input configurations"""
    print("\n" + "=" * 60)
    print("Testing Different Configurations")
    print("=" * 60)
    
    configurations = [
        # (input_shape, kernel_size, out_channels, description)
        ((16, 16, 32), 3, 64, "Small feature map"),
        ((32, 32, 64), 5, 128, "Medium feature map"),
        ((8, 8, 256), 3, 512, "High channels"),
    ]
    
    batch_size = 2
    
    for input_shape, kernel_size, out_channels, description in configurations:
        print(f"\n{description}:")
        print(f"  Input: {input_shape}, Kernel: {kernel_size}, Output channels: {out_channels}")
        
        # Standard convolution
        std_conv = StandardConv2D(input_shape[2], out_channels, kernel_size)
        std_params = std_conv.get_parameters()
        std_flops = std_conv.get_flops((batch_size, *input_shape))
        
        # Depthwise separable
        sep_conv = DepthwiseSeparableConv2D(input_shape[2], out_channels, kernel_size)
        sep_params = sep_conv.get_parameters()
        sep_flops = sep_conv.get_flops((batch_size, *input_shape))
        
        param_reduction = std_params / sep_params
        flops_reduction = std_flops / sep_flops
        
        print(f"  Standard: {std_params:,} params, {std_flops:,} FLOPs")
        print(f"  Separable: {sep_params:,} params, {sep_flops:,} FLOPs")
        print(f"  Reduction: {param_reduction:.2f}x params, {flops_reduction:.2f}x FLOPs")

def analyze_theoretical_savings():
    """Analyze theoretical parameter and computation savings"""
    print("\n" + "=" * 60)
    print("Theoretical Savings Analysis")
    print("=" * 60)
    
    # For a standard convolution:
    # Parameters = K × K × C_in × C_out + C_out
    # For depthwise separable:
    # Parameters = (K × K × C_in) + (C_in × C_out) + C_in + C_out
    
    kernel_sizes = [3, 5, 7]
    channel_ratios = [1, 2, 4]  # C_out / C_in
    
    print("Theoretical parameter ratios (Separable / Standard):")
    print("Kernel | C_out/C_in | Parameter Ratio")
    print("-" * 40)
    
    for k in kernel_sizes:
        for ratio in channel_ratios:
            # Standard conv parameters (ignoring bias for simplicity)
            std_params = k * k * 1 * ratio  # C_in = 1 for normalized calculation
            
            # Separable conv parameters
            sep_params = (k * k * 1) + (1 * ratio)  # Depthwise + Pointwise
            
            param_ratio = sep_params / std_params
            
            print(f"  {k}×{k}  |     {ratio}      |     {param_ratio:.3f}")

def verify_output_correctness():
    """Verify that both implementations produce correct output shapes"""
    print("\n" + "=" * 60)
    print("Output Shape Verification")
    print("=" * 60)
    
    test_cases = [
        ((16, 16, 32), 3, 64, 1, 0),
        ((16, 16, 32), 3, 64, 1, 1),
        ((16, 16, 32), 3, 64, 2, 0),
        ((16, 16, 32), 5, 128, 1, 2),
    ]
    
    batch_size = 2
    
    for input_shape, k, out_c, s, p in test_cases:
        x = np.random.randn(batch_size, *input_shape)
        
        std_conv = StandardConv2D(input_shape[2], out_c, k, s, p)
        sep_conv = DepthwiseSeparableConv2D(input_shape[2], out_c, k, s, p)
        
        std_out = std_conv.forward(x)
        sep_out = sep_conv.forward(x)
        
        print(f"Input: {input_shape}, Kernel: {k}, Stride: {s}, Padding: {p}")
        print(f"  Standard output: {std_out.shape}")
        print(f"  Separable output: {sep_out.shape}")
        print(f"  Shapes match: {std_out.shape == sep_out.shape}")
        print()

def main():
    """Main function to run all comparisons and tests"""
    print("Exercise 17: Depthwise & Separable Convolutions")
    print("=" * 60)
    
    # Compare standard vs separable convolution
    results = compare_operations()
    
    # Test different configurations
    test_different_configurations()
    
    # Theoretical analysis
    analyze_theoretical_savings()
    
    # Verify output correctness
    verify_output_correctness()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("- Implemented Standard, Depthwise, Pointwise, and DepthwiseSeparable convolutions")
    print("- Depthwise separable conv provides significant parameter and FLOP reduction")
    print("- Typical savings: 8-9x fewer parameters and computations for 3x3 kernels")
    print("- Maintains same output shape as standard convolution")
    print("- Essential for mobile and efficient neural networks")

if __name__ == "__main__":
    main()