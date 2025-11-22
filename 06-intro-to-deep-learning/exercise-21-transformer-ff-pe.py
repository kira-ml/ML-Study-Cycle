"""
Exercise 21 — Transformer Feed-Forward & Positional Encodings
Filename: exercise-21-transformer-ff-pe.py

Core Concept: Position encodings and transformer-style feed-forward blocks 
(layer norm, residuals, and the complete encoder block structure).

This implementation demonstrates:
1. Sinusoidal positional encodings for sequence order information
2. Transformer feed-forward networks with GELU activation
3. Layer normalization and residual connections
4. Complete encoder block integration
5. Shape consistency and gradient flow verification

Key Insights:
- Positional encodings inject order information without additional parameters
- Residual connections enable deep network training by preserving gradient flow
- Layer normalization stabilizes training and improves convergence
- Feed-forward blocks provide non-linear transformation capacity
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, List
import seaborn as sns
from exercise_20_attention import ScaledDotProductAttention, stable_softmax


class PositionalEncoding:
    """
    Sinusoidal Positional Encodings for Transformer models.
    
    Core Idea: Inject sequence position information without learnable parameters
    using sine and cosine functions of different frequencies.
    
    Mathematical Formulation:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Why sinusoidal encodings:
    - Fixed patterns that don't require learning
    - Can extrapolate to longer sequences than seen during training
    - Each dimension corresponds to a different wavelength/frequency
    - Relative positions can be represented via linear transformations
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 5000):
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pe = self._create_positional_encoding()
        
    def _create_positional_encoding(self) -> np.ndarray:
        """Create sinusoidal positional encoding matrix."""
        position = np.arange(self.max_seq_len).reshape(-1, 1)  # (max_seq_len, 1)
        div_term = np.exp(np.arange(0, self.d_model, 2) * 
                         (-np.log(10000.0) / self.d_model))  # (d_model/2,)
        
        # Initialize positional encoding matrix
        pe = np.zeros((self.max_seq_len, self.d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = np.sin(position * div_term)
        # Apply cosine to odd indices  
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe  # (max_seq_len, d_model)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Add positional encodings to input sequence.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            x_with_pe: Input with positional encodings added
        """
        batch_size, seq_len, d_model = x.shape
        
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")
        
        # Get positional encodings for current sequence length
        positional_encodings = self.pe[:seq_len, :]  # (seq_len, d_model)
        
        # Add to input (broadcast across batch dimension)
        return x + positional_encodings[np.newaxis, :, :]
    
    def visualize(self, seq_len: int = 50):
        """Visualize positional encodings for better understanding."""
        plt.figure(figsize=(12, 8))
        
        # Plot full positional encoding matrix
        plt.subplot(2, 2, 1)
        plt.imshow(self.pe[:seq_len, :].T, aspect='auto', cmap='RdBu')
        plt.colorbar()
        plt.title('Positional Encodings Matrix')
        plt.xlabel('Position')
        plt.ylabel('Dimension')
        
        # Plot first few dimensions across positions
        plt.subplot(2, 2, 2)
        for i in range(6):
            plt.plot(self.pe[:seq_len, i], label=f'Dim {i}')
        plt.title('First 6 Dimensions vs Position')
        plt.xlabel('Position')
        plt.ylabel('Encoding Value')
        plt.legend()
        
        # Plot wavelength analysis
        plt.subplot(2, 2, 3)
        wavelengths = 2 * np.pi * 10000 ** (np.arange(self.d_model//2) / self.d_model)
        plt.plot(wavelengths[:20])
        plt.title('Wavelengths of Different Dimensions')
        plt.xlabel('Dimension Index')
        plt.ylabel('Wavelength')
        plt.yscale('log')
        
        # Plot correlation between positions
        plt.subplot(2, 2, 4)
        correlation = np.corrcoef(self.pe[:seq_len, :].T)
        plt.imshow(correlation, cmap='viridis')
        plt.colorbar()
        plt.title('Dimension Correlation Matrix')
        plt.xlabel('Dimension')
        plt.ylabel('Dimension')
        
        plt.tight_layout()
        plt.savefig('positional_encodings.png', dpi=150, bbox_inches='tight')
        plt.show()


def gelu(x: np.ndarray) -> np.ndarray:
    """
    Gaussian Error Linear Unit (GELU) activation function.
    
    GELU is commonly used in Transformer feed-forward networks because:
    - It's smooth and differentiable everywhere
    - It combines properties of ReLU and dropout
    - It often performs better than ReLU in deep networks
    
    Approximation: GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def layer_norm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> Tuple[np.ndarray, Tuple]:
    """
    Layer Normalization forward pass.
    
    Why LayerNorm is crucial in Transformers:
    - Stabilizes training by normalizing activations
    - Enables faster convergence and better performance
    - Applied to each sequence element independently
    - Learnable parameters (gamma, beta) maintain representation power
    
    Formula: LN(x) = γ * (x - μ) / √(σ² + ε) + β
    """
    # Compute statistics along the feature dimension
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    
    # Normalize
    x_normalized = (x - mean) / np.sqrt(variance + eps)
    
    # Scale and shift
    out = gamma * x_normalized + beta
    
    cache = (x, x_normalized, gamma, mean, variance, eps)
    return out, cache


def layer_norm_backward(dout: np.ndarray, cache: Tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Layer Normalization backward pass.
    
    Implements the gradient computation for layer normalization,
    which is essential for stable Transformer training.
    """
    x, x_normalized, gamma, mean, variance, eps = cache
    
    # Get dimensions
    batch_size, seq_len, d_model = x.shape
    
    # Gradient for gamma and beta
    dgamma = np.sum(dout * x_normalized, axis=(0, 1), keepdims=True)
    dbeta = np.sum(dout, axis=(0, 1), keepdims=True)
    
    # Gradient for normalized input
    dx_normalized = dout * gamma
    
    # Gradient for variance
    dvar = np.sum(dx_normalized * (x - mean) * -0.5 * (variance + eps) ** (-1.5), axis=-1, keepdims=True)
    
    # Gradient for mean
    dmean1 = np.sum(dx_normalized * -1 / np.sqrt(variance + eps), axis=-1, keepdims=True)
    dmean2 = dvar * np.mean(-2 * (x - mean), axis=-1, keepdims=True)
    dmean = dmean1 + dmean2
    
    # Gradient for input
    dx1 = dx_normalized / np.sqrt(variance + eps)
    dx2 = dvar * 2 * (x - mean) / (d_model * batch_size * seq_len)
    dx3 = dmean / (d_model * batch_size * seq_len)
    
    dx = dx1 + dx2 + dx3
    
    return dx, dgamma, dbeta


class FeedForwardNetwork:
    """
    Transformer Feed-Forward Network (Position-wise Fully Connected).
    
    Architecture: 
    Input → Linear → GELU → Linear → Output
    
    Why this design:
    - Applied independently to each position (position-wise)
    - Provides non-linear transformation capacity
    - Typically uses expansion factor (4x) in hidden dimension
    - GELU activation works well with LayerNorm
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        # Expansion factor typically 4x in original Transformer
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        
        # Weights and biases
        self.W1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.b1 = np.zeros((1, 1, d_ff))
        self.W2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff) 
        self.b2 = np.zeros((1, 1, d_model))
        
        # Cache for backward pass
        self.cache = None
        
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            training: Whether in training mode (affects dropout)
            
        Returns:
            output: Transformed tensor of same shape as input
        """
        batch_size, seq_len, d_model = x.shape
        
        # First linear transformation with expansion
        h = np.matmul(x, self.W1) + self.b1  # (batch_size, seq_len, d_ff)
        
        # GELU activation
        h_activated = gelu(h)
        
        # Optional dropout (not always used in FFN in modern implementations)
        if training and self.dropout > 0:
            mask = (np.random.random(h_activated.shape) > self.dropout).astype(np.float32)
            h_activated = h_activated * mask / (1.0 - self.dropout)
        
        # Second linear transformation (project back to d_model)
        output = np.matmul(h_activated, self.W2) + self.b2  # (batch_size, seq_len, d_model)
        
        self.cache = (x, h, h_activated)
        return output
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass through feed-forward network.
        """
        x, h, h_activated = self.cache
        batch_size, seq_len, d_model = x.shape
        
        # Gradient for second linear layer
        self.dW2 = np.matmul(h_activated.swapaxes(1, 2), dout).mean(axis=0)  # (d_ff, d_model)
        self.db2 = np.sum(dout, axis=(0, 1), keepdims=True) / (batch_size * seq_len)
        dh_activated = np.matmul(dout, self.W2.T)  # (batch_size, seq_len, d_ff)
        
        # Gradient through GELU
        # GELU derivative: 0.5 * (1 + tanh(√(2/π)(x + 0.044715x³))) + 
        #                  0.5x * (1 - tanh²(√(2/π)(x + 0.044715x³))) * √(2/π)(1 + 0.134145x²)
        gelu_derivative = 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (h + 0.044715 * h**3))) + \
                         0.5 * h * (1 - np.tanh(np.sqrt(2/np.pi) * (h + 0.044715 * h**3))**2) * \
                         np.sqrt(2/np.pi) * (1 + 0.134145 * h**2)
        dh = dh_activated * gelu_derivative
        
        # Gradient for first linear layer
        self.dW1 = np.matmul(x.swapaxes(1, 2), dh).mean(axis=0)  # (d_model, d_ff)
        self.db1 = np.sum(dh, axis=(0, 1), keepdims=True) / (batch_size * seq_len)
        dx = np.matmul(dh, self.W1.T)  # (batch_size, seq_len, d_model)
        
        return dx


class ResidualConnection:
    """
    Residual Connection with Layer Normalization.
    
    Architecture options:
    1. Pre-LN: LayerNorm → Sublayer → Residual (modern, more stable)
    2. Post-LN: Sublayer → LayerNorm → Residual (original Transformer)
    
    Why residuals work:
    - Enable training of very deep networks
    - Preserve gradient flow through identity connections
    - Help prevent vanishing/exploding gradients
    - Combined with LayerNorm for stabilization
    """
    
    def __init__(self, d_model: int, pre_norm: bool = True):
        self.d_model = d_model
        self.pre_norm = pre_norm
        
        # LayerNorm parameters
        self.gamma = np.ones((1, 1, d_model))
        self.beta = np.zeros((1, 1, d_model))
        
        self.cache = None
        
    def forward(self, x: np.ndarray, sublayer_fn, *sublayer_args) -> np.ndarray:
        """
        Forward pass through residual connection.
        
        Args:
            x: Input tensor
            sublayer_fn: Function to apply in the sublayer (attention or FFN)
            sublayer_args: Arguments to pass to sublayer_fn
            
        Returns:
            output: Result after residual connection and layer norm
        """
        if self.pre_norm:
            # Pre-LN: LayerNorm then sublayer then residual
            x_norm, ln_cache = layer_norm_forward(x, self.gamma, self.beta)
            sublayer_output = sublayer_fn(x_norm, *sublayer_args)
            output = x + sublayer_output
        else:
            # Post-LN: Sublayer then LayerNorm then residual (original)
            sublayer_output = sublayer_fn(x, *sublayer_args)
            output_norm, ln_cache = layer_norm_forward(x + sublayer_output, self.gamma, self.beta)
            output = output_norm
        
        self.cache = (x, sublayer_output, ln_cache, self.pre_norm)
        return output
    
    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward pass through residual connection.
        """
        x, sublayer_output, ln_cache, pre_norm = self.cache
        
        if pre_norm:
            # Pre-LN backward: dout → residual → sublayer → LayerNorm
            dx_residual = dout  # Gradient through residual connection
            
            # Gradient through sublayer
            dsublayer_output = dx_residual
            
            # Gradient through LayerNorm
            dx_norm, dgamma, dbeta = layer_norm_backward(dsublayer_output, ln_cache)
            dx = dx_residual + dx_norm  # Sum gradients from both paths
        else:
            # Post-LN backward: dout → LayerNorm → residual
            dx_norm, dgamma, dbeta = layer_norm_backward(dout, ln_cache)
            dx_residual = dx_norm  # Gradient through residual
            
            # Gradient splits to identity and sublayer paths
            dx = dx_residual
            dsublayer_output = dx_residual
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx, dsublayer_output


class TransformerEncoderBlock:
    """
    Complete Transformer Encoder Block.
    
    Architecture:
    Input → Pre-LN → Multi-Head Attention → Residual → Pre-LN → FFN → Residual → Output
    
    Key Components:
    1. Multi-Head Self-Attention (using our previous implementation)
    2. Feed-Forward Network
    3. Residual Connections with Layer Normalization
    4. Shape consistency throughout (batch_size, seq_len, d_model)
    """
    
    def __init__(self, d_model: int, d_ff: int, num_heads: int = 2, dropout: float = 0.1):
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Self-attention mechanism (using single-head for simplicity)
        self.self_attention = ScaledDotProductAttention(d_model, dropout)
        
        # Feed-forward network
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        
        # Residual connections
        self.attention_residual = ResidualConnection(d_model, pre_norm=True)
        self.ffn_residual = ResidualConnection(d_model, pre_norm=True)
        
        # Cache for debugging and analysis
        self.cache = None
        
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None, training: bool = True) -> np.ndarray:
        """
        Forward pass through encoder block.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask
            training: Whether in training mode
            
        Returns:
            output: Transformed tensor (batch_size, seq_len, d_model)
        """
        # Self-attention sublayer with residual
        def self_attention_sublayer(x_norm):
            # For self-attention: Q, K, V all come from the same source
            return self.self_attention.forward(x_norm, x_norm, x_norm, mask, training)[0]
        
        x = self.attention_residual.forward(x, self_attention_sublayer)
        
        # Feed-forward sublayer with residual
        def ffn_sublayer(x_norm):
            return self.feed_forward.forward(x_norm, training)
        
        output = self.ffn_residual.forward(x, ffn_sublayer)
        
        self.cache = {
            'input': x,
            'attention_output': x,  # After first residual
            'output': output
        }
        
        return output
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass through encoder block.
        """
        # Backward through FFN residual
        dx_ffn, d_ffn_output = self.ffn_residual.backward(dout)
        
        # Backward through attention residual  
        dx_attention, d_attention_output = self.attention_residual.backward(dx_ffn)
        
        return dx_attention


def verify_shape_consistency():
    """
    Verify shape consistency throughout the Transformer encoder block.
    
    This demonstrates that all components maintain the same shape
    (batch_size, seq_len, d_model), which is crucial for residual connections.
    """
    print("=" * 70)
    print("SHAPE CONSISTENCY VERIFICATION")
    print("=" * 70)
    
    # Test parameters
    batch_size = 2
    seq_len = 16
    d_model = 64
    d_ff = 256
    
    # Initialize components
    encoder_block = TransformerEncoderBlock(d_model, d_ff)
    positional_encoding = PositionalEncoding(d_model)
    
    # Generate random input
    x = np.random.randn(batch_size, seq_len, d_model) * 0.1
    print(f"Input shape: {x.shape}")
    
    # Add positional encodings
    x_pe = positional_encoding.forward(x)
    print(f"After positional encoding: {x_pe.shape}")
    
    # Forward through encoder block
    output = encoder_block.forward(x_pe, training=False)
    print(f"Encoder block output: {output.shape}")
    
    # Verify shape consistency
    assert x.shape == output.shape, "Input and output shapes must match!"
    print("✓ Shape consistency verified!")
    
    # Parameter count analysis
    total_params = 0
    
    # Self-attention parameters (simplified - using our single-head implementation)
    attention_params = encoder_block.self_attention.W.size + encoder_block.self_attention.b.size
    total_params += attention_params
    
    # Feed-forward parameters
    ffn_params = (encoder_block.feed_forward.W1.size + encoder_block.feed_forward.b1.size +
                  encoder_block.feed_forward.W2.size + encoder_block.feed_forward.b2.size)
    total_params += ffn_params
    
    # LayerNorm parameters (2 residual connections × 2 parameters each)
    ln_params = (encoder_block.attention_residual.gamma.size + 
                 encoder_block.attention_residual.beta.size +
                 encoder_block.ffn_residual.gamma.size + 
                 encoder_block.ffn_residual.beta.size)
    total_params += ln_params
    
    print(f"\nParameter Analysis:")
    print(f"Self-Attention: {attention_params:,} parameters")
    print(f"Feed-Forward: {ffn_params:,} parameters") 
    print(f"LayerNorm: {ln_params:,} parameters")
    print(f"Total: {total_params:,} parameters")
    
    return encoder_block, x_pe, output


def analyze_internal_representations(encoder_block, x_pe, output):
    """
    Analyze internal representations and gradients.
    """
    print("\n" + "=" * 70)
    print("INTERNAL REPRESENTATION ANALYSIS")
    print("=" * 70)
    
    # Compute gradients (simplified backward pass)
    dout = np.ones_like(output)
    dx = encoder_block.backward(dout)
    
    print("Gradient flow analysis:")
    print(f"Output gradient norm: {np.linalg.norm(dout):.6f}")
    print(f"Input gradient norm: {np.linalg.norm(dx):.6f}")
    print(f"Gradient preservation ratio: {np.linalg.norm(dx) / np.linalg.norm(dout):.4f}")
    
    # Analyze activation statistics
    print("\nActivation statistics:")
    print(f"Input - Mean: {np.mean(x_pe):.6f}, Std: {np.std(x_pe):.6f}")
    print(f"Output - Mean: {np.mean(output):.6f}, Std: {np.std(output):.6f}")
    
    # Check for gradient issues
    if np.linalg.norm(dx) < 1e-8:
        print("⚠️  Warning: Very small gradients - possible vanishing gradient issue")
    elif np.linalg.norm(dx) > 1e8:
        print("⚠️  Warning: Very large gradients - possible exploding gradient issue")
    else:
        print("✓ Healthy gradient flow maintained")


def demonstrate_positional_encodings():
    """
    Demonstrate properties of sinusoidal positional encodings.
    """
    print("\n" + "=" * 70)
    print("POSITIONAL ENCODING DEMONSTRATION")
    print("=" * 70)
    
    d_model = 64
    seq_len = 20
    pe = PositionalEncoding(d_model)
    
    # Analyze positional encoding properties
    encoding_matrix = pe.pe[:seq_len, :]
    
    print("Positional Encoding Properties:")
    print(f"Shape: {encoding_matrix.shape}")
    print(f"Value range: [{encoding_matrix.min():.3f}, {encoding_matrix.max():.3f}]")
    print(f"Mean: {encoding_matrix.mean():.6f} (should be ~0)")
    print(f"Std: {encoding_matrix.std():.6f} (should be ~{1/np.sqrt(2):.3f})")
    
    # Check orthogonality-like properties
    dot_products = []
    for i in range(seq_len - 1):
        for j in range(i + 1, seq_len):
            dot = np.dot(encoding_matrix[i], encoding_matrix[j])
            dot_products.append(dot)
    
    print(f"Inter-position dot products - Mean: {np.mean(dot_products):.6f}, Std: {np.std(dot_products):.6f}")
    
    # Visualize
    pe.visualize(seq_len=seq_len)
    
    return pe


if __name__ == "__main__":
    print("Exercise 21: Transformer Feed-Forward & Positional Encodings")
    print("Core Concept: Position encodings, layer norm, residuals, and encoder blocks")
    print("=" * 70)
    
    # Demonstrate positional encodings
    pe = demonstrate_positional_encodings()
    
    # Verify shape consistency and analyze components
    encoder_block, x_pe, output = verify_shape_consistency()
    
    # Analyze internal representations
    analyze_internal_representations(encoder_block, x_pe, output)
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("1. POSITIONAL ENCODINGS: Fixed sinusoidal patterns provide order information")
    print("2. LAYER NORM: Stabilizes training by normalizing activations per sequence element") 
    print("3. RESIDUAL CONNECTIONS: Preserve gradient flow and enable deep networks")
    print("4. FEED-FORWARD NETWORKS: Provide non-linear transformation capacity")
    print("5. SHAPE CONSISTENCY: All components maintain (batch_size, seq_len, d_model)")
    print("6. GRADIENT PRESERVATION: Residual connections prevent vanishing gradients")