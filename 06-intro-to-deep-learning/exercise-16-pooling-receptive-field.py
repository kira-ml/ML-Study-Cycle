import numpy as np
import matplotlib.pyplot as plt

class PoolingLayer:
    """Base class for pooling layers"""
    
    def __init__(self, pool_size=2, stride=2, padding=0):
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding
        self.cache = None
    
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, dout):
        raise NotImplementedError

class MaxPool2D(PoolingLayer):
    """2D Max Pooling layer"""
    
    def forward(self, x):
        """
        Forward pass for max pooling
        x: input of shape (batch_size, height, width, channels)
        """
        batch_size, h_in, w_in, channels = x.shape
        
        # Calculate output dimensions
        h_out = (h_in + 2 * self.padding - self.pool_size) // self.stride + 1
        w_out = (w_in + 2 * self.padding - self.pool_size) // self.stride + 1
        
        # Apply padding if needed
        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (self.padding, self.padding), 
                                 (self.padding, self.padding), (0, 0)), mode='constant')
        else:
            x_padded = x
        
        # Initialize output
        output = np.zeros((batch_size, h_out, w_out, channels))
        
        # Create mask for storing max positions (for backward pass)
        self.cache = {'x_shape': x.shape, 'mask': np.zeros_like(x_padded)}
        
        # Perform max pooling
        for b in range(batch_size):
            for c in range(channels):
                for h in range(h_out):
                    for w in range(w_out):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size
                        
                        window = x_padded[b, h_start:h_end, w_start:w_end, c]
                        output[b, h, w, c] = np.max(window)
                        
                        # Store position of max value for backward pass
                        max_pos = np.unravel_index(np.argmax(window), window.shape)
                        self.cache['mask'][b, h_start + max_pos[0], w_start + max_pos[1], c] = 1
        
        return output
    
    def backward(self, dout):
        """
        Backward pass for max pooling
        dout: upstream derivatives, shape (batch_size, h_out, w_out, channels)
        """
        x_shape = self.cache['x_shape']
        batch_size, h_in, w_in, channels = x_shape
        
        # Initialize gradient with padding
        dx = np.zeros((batch_size, 
                      h_in + 2 * self.padding, 
                      w_in + 2 * self.padding, 
                      channels))
        
        h_out = dout.shape[1]
        w_out = dout.shape[2]
        
        # Distribute gradients to max positions
        for b in range(batch_size):
            for c in range(channels):
                for h in range(h_out):
                    for w in range(w_out):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size
                        
                        # Only the max position gets the gradient
                        dx[b, h_start:h_end, w_start:w_end, c] += \
                            self.cache['mask'][b, h_start:h_end, w_start:w_end, c] * dout[b, h, w, c]
        
        # Remove padding if it was applied
        if self.padding > 0:
            dx = dx[:, self.padding:-self.padding, self.padding:-self.padding, :]
        
        return dx

class AvgPool2D(PoolingLayer):
    """2D Average Pooling layer"""
    
    def forward(self, x):
        """
        Forward pass for average pooling
        x: input of shape (batch_size, height, width, channels)
        """
        batch_size, h_in, w_in, channels = x.shape
        
        # Calculate output dimensions
        h_out = (h_in + 2 * self.padding - self.pool_size) // self.stride + 1
        w_out = (w_in + 2 * self.padding - self.pool_size) // self.stride + 1
        
        # Apply padding if needed
        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (self.padding, self.padding), 
                                 (self.padding, self.padding), (0, 0)), mode='constant')
        else:
            x_padded = x
        
        # Initialize output
        output = np.zeros((batch_size, h_out, w_out, channels))
        
        # Store input for backward pass
        self.cache = {'x_shape': x.shape, 'x_padded': x_padded}
        
        # Perform average pooling
        for b in range(batch_size):
            for c in range(channels):
                for h in range(h_out):
                    for w in range(w_out):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size
                        
                        window = x_padded[b, h_start:h_end, w_start:w_end, c]
                        output[b, h, w, c] = np.mean(window)
        
        return output
    
    def backward(self, dout):
        """
        Backward pass for average pooling
        dout: upstream derivatives, shape (batch_size, h_out, w_out, channels)
        """
        x_shape = self.cache['x_shape']
        x_padded = self.cache['x_padded']
        batch_size, h_in, w_in, channels = x_shape
        
        # Initialize gradient with padding
        dx = np.zeros_like(x_padded)
        
        h_out = dout.shape[1]
        w_out = dout.shape[2]
        
        # Distribute gradients equally to all positions in the window
        for b in range(batch_size):
            for c in range(channels):
                for h in range(h_out):
                    for w in range(w_out):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size
                        
                        # Average gradient across the window
                        grad = dout[b, h, w, c] / (self.pool_size * self.pool_size)
                        dx[b, h_start:h_end, w_start:w_end, c] += grad
        
        # Remove padding if it was applied
        if self.padding > 0:
            dx = dx[:, self.padding:-self.padding, self.padding:-self.padding, :]
        
        return dx

class ReceptiveFieldAnalyzer:
    """Utility for computing receptive field size and effective stride"""
    
    def __init__(self):
        self.layers = []
    
    def add_conv_layer(self, kernel_size, stride=1, padding=0, dilation=1):
        """Add a convolutional layer to the analysis"""
        self.layers.append({
            'type': 'conv',
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'dilation': dilation
        })
    
    def add_pool_layer(self, pool_size, stride=None, padding=0):
        """Add a pooling layer to the analysis"""
        if stride is None:
            stride = pool_size
            
        self.layers.append({
            'type': 'pool',
            'pool_size': pool_size,
            'stride': stride,
            'padding': padding
        })
    
    def analyze(self):
        """
        Compute receptive field size and effective stride
        Returns: (receptive_field_size, effective_stride)
        """
        receptive_field = 1
        effective_stride = 1
        current_center = 0.5  # Starting from center of first pixel
        
        for layer in self.layers:
            if layer['type'] in ['conv', 'pool']:
                kernel_size = layer.get('kernel_size', layer.get('pool_size', 1))
                stride = layer.get('stride', 1)
                padding = layer.get('padding', 0)
                dilation = layer.get('dilation', 1)
                
                # Effective kernel size considering dilation
                effective_kernel_size = (kernel_size - 1) * dilation + 1
                
                # Update receptive field
                receptive_field = receptive_field + (effective_kernel_size - 1) * effective_stride
                
                # Update effective stride
                effective_stride = effective_stride * stride
                
                # Update center position (for more precise analysis)
                current_center = current_center * stride + (padding - (effective_kernel_size - 1) / 2)
        
        return int(receptive_field), int(effective_stride), current_center

def test_pooling_operations():
    """Test pooling operations with one-hot input"""
    print("Testing Pooling Operations")
    print("=" * 50)
    
    # Create one-hot input
    input_data = np.zeros((1, 4, 4, 1))
    input_data[0, 1, 2, 0] = 1.0  # One-hot at position (1,2)
    
    print("Input shape:", input_data.shape)
    print("Input (channel 0):")
    print(input_data[0, :, :, 0])
    
    # Test Max Pooling
    max_pool = MaxPool2D(pool_size=2, stride=2)
    max_output = max_pool.forward(input_data)
    print("\nMax Pooling Output (channel 0):")
    print(max_output[0, :, :, 0])
    
    # Test backward pass with ones
    dout_max = np.ones_like(max_output)
    dx_max = max_pool.backward(dout_max)
    print("\nMax Pooling Gradient (channel 0):")
    print(dx_max[0, :, :, 0])
    
    # Test Average Pooling
    avg_pool = AvgPool2D(pool_size=2, stride=2)
    avg_output = avg_pool.forward(input_data)
    print("\nAverage Pooling Output (channel 0):")
    print(avg_output[0, :, :, 0])
    
    # Test backward pass with ones
    dout_avg = np.ones_like(avg_output)
    dx_avg = avg_pool.backward(dout_avg)
    print("\nAverage Pooling Gradient (channel 0):")
    print(dx_avg[0, :, :, 0])
    
    return input_data, max_output, avg_output, dx_max, dx_avg

def test_receptive_field_analysis():
    """Test receptive field analysis with various layer configurations"""
    print("\n" + "=" * 50)
    print("Receptive Field Analysis")
    print("=" * 50)
    
    # Test case 1: Simple CNN
    print("Test Case 1: Simple CNN")
    analyzer1 = ReceptiveFieldAnalyzer()
    analyzer1.add_conv_layer(kernel_size=3, stride=1, padding=1)
    analyzer1.add_conv_layer(kernel_size=3, stride=1, padding=1)
    analyzer1.add_pool_layer(pool_size=2, stride=2)
    analyzer1.add_conv_layer(kernel_size=3, stride=1, padding=1)
    
    rf1, es1, center1 = analyzer1.analyze()
    print(f"Receptive Field: {rf1}, Effective Stride: {es1}, Center Offset: {center1:.2f}")
    
    # Test case 2: VGG-like architecture
    print("\nTest Case 2: VGG-like")
    analyzer2 = ReceptiveFieldAnalyzer()
    analyzer2.add_conv_layer(kernel_size=3, stride=1, padding=1)
    analyzer2.add_conv_layer(kernel_size=3, stride=1, padding=1)
    analyzer2.add_pool_layer(pool_size=2, stride=2)
    analyzer2.add_conv_layer(kernel_size=3, stride=1, padding=1)
    analyzer2.add_conv_layer(kernel_size=3, stride=1, padding=1)
    analyzer2.add_pool_layer(pool_size=2, stride=2)
    
    rf2, es2, center2 = analyzer2.analyze()
    print(f"Receptive Field: {rf2}, Effective Stride: {es2}, Center Offset: {center2:.2f}")
    
    # Test case 3: Network with dilation
    print("\nTest Case 3: With Dilation")
    analyzer3 = ReceptiveFieldAnalyzer()
    analyzer3.add_conv_layer(kernel_size=3, stride=1, padding=1)
    analyzer3.add_conv_layer(kernel_size=3, stride=1, padding=1, dilation=2)
    analyzer3.add_pool_layer(pool_size=2, stride=2)
    
    rf3, es3, center3 = analyzer3.analyze()
    print(f"Receptive Field: {rf3}, Effective Stride: {es3}, Center Offset: {center3:.2f}")
    
    return [(rf1, es1), (rf2, es2), (rf3, es3)]

def visualize_results(input_data, max_output, avg_output, dx_max, dx_avg):
    """Visualize the pooling operations and their effects"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Input
    axes[0, 0].imshow(input_data[0, :, :, 0], cmap='viridis')
    axes[0, 0].set_title('Input (One-hot)')
    axes[0, 0].grid(False)
    
    # Max Pooling Output
    axes[0, 1].imshow(max_output[0, :, :, 0], cmap='viridis')
    axes[0, 1].set_title('Max Pooling Output')
    axes[0, 1].grid(False)
    
    # Max Pooling Gradient
    axes[0, 2].imshow(dx_max[0, :, :, 0], cmap='viridis')
    axes[0, 2].set_title('Max Pooling Gradient')
    axes[0, 2].grid(False)
    
    # Input (same as above)
    axes[1, 0].imshow(input_data[0, :, :, 0], cmap='viridis')
    axes[1, 0].set_title('Input (One-hot)')
    axes[1, 0].grid(False)
    
    # Average Pooling Output
    axes[1, 1].imshow(avg_output[0, :, :, 0], cmap='viridis')
    axes[1, 1].set_title('Average Pooling Output')
    axes[1, 1].grid(False)
    
    # Average Pooling Gradient
    axes[1, 2].imshow(dx_avg[0, :, :, 0], cmap='viridis')
    axes[1, 2].set_title('Average Pooling Gradient')
    axes[1, 2].grid(False)
    
    plt.tight_layout()
    plt.savefig('pooling_operations.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run all tests"""
    print("Exercise 16: Pooling, Stride Handling & Receptive Field Analysis")
    print("=" * 60)
    
    # Test pooling operations
    input_data, max_output, avg_output, dx_max, dx_avg = test_pooling_operations()
    
    # Test receptive field analysis
    rf_results = test_receptive_field_analysis()
    
    # Visualize results
    visualize_results(input_data, max_output, avg_output, dx_max, dx_avg)
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("- Implemented MaxPool2D and AvgPool2D with forward/backward passes")
    print("- Created ReceptiveFieldAnalyzer for computing RF size and effective stride")
    print("- Validated with one-hot input to track feature propagation")
    print("- Visualizations saved as 'pooling_operations.png'")

if __name__ == "__main__":
    main()