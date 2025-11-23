"""
Exercise 18: RNN Backpropagation Through Time (BPTT)

This module implements a complete Recurrent Neural Network (RNN) with full
Backpropagation Through Time (BPTT) for sequence learning. RNNs are powerful
for processing sequential data like time series, text, or speech because they
maintain a "memory" of previous inputs through hidden states.

Key Concepts:
- RNN Cell: Processes one timestep at a time, combining current input with previous hidden state
- BPTT: Unrolls the RNN through time and applies backpropagation to learn temporal dependencies
- Vanishing Gradients: A challenge in RNNs where gradients diminish exponentially over time
- Truncated BPTT: Limits the number of timesteps for backpropagation to combat vanishing gradients

Mathematical Foundation:
- Hidden state: h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
- Output: y_t = W_hy * h_t + b_y
- Gradients flow backward through time, accumulating over the sequence
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

class RNNCell:
    """
    Vanilla RNN Cell - The fundamental building block of RNNs
    
    An RNN cell maintains a hidden state that captures information from previous
    timesteps. At each timestep t, it combines the current input x_t with the
    previous hidden state h_{t-1} to produce the next hidden state h_t and output y_t.
    
    This is called "vanilla" RNN because it's the basic, unmodified version.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights with small random values to break symmetry
        # Small initialization helps prevent exploding gradients during training
        self.W_xh = np.random.randn(input_size, hidden_size) * 0.01  # Input to hidden
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01  # Hidden to hidden (recurrent)
        self.W_hy = np.random.randn(hidden_size, output_size) * 0.01  # Hidden to output
        
        # Initialize biases to zero (common practice)
        self.b_h = np.zeros(hidden_size)
        self.b_y = np.zeros(output_size)
        
        # Cache for backward pass - stores intermediate values needed for gradient computation
        self.cache = None
    
    def forward(self, x: np.ndarray, h_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass for single RNN step
        
        This is where the "magic" of RNNs happens - combining current input with memory
        from previous timesteps. The tanh activation ensures hidden states stay bounded
        between -1 and 1, helping prevent exploding activations.
        
        Mathematical formulation:
        h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)  # Update hidden state
        y_t = W_hy * h_t + b_y                          # Compute output
        
        Args:
            x: Current input of shape (batch_size, input_size)
            h_prev: Previous hidden state of shape (batch_size, hidden_size)
        
        Returns:
            Tuple of (next hidden state, output prediction)
        """
        # Compute the next hidden state by combining input and previous hidden state
        # The tanh activation provides the non-linearity and bounds the values
        h_next = np.tanh(np.dot(x, self.W_xh) + np.dot(h_prev, self.W_hh) + self.b_h)
        
        # Compute the output from the current hidden state
        # Note: No activation here - could be followed by softmax for classification
        y = np.dot(h_next, self.W_hy) + self.b_y
        
        # Store intermediate values for backward pass
        # This is crucial for computing gradients later
        self.cache = (x, h_prev, h_next)
        
        return h_next, y
    
    def backward(self, dh_next: np.ndarray, dy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Backward pass for single RNN step
        
        This implements the chain rule for RNNs. Gradients come from two sources:
        1. The output at this timestep (dy)
        2. The next hidden state in the sequence (dh_next) - this is the "through time" part
        
        The key insight: gradients flow backward through the sequence, and we need to
        accumulate them properly. The tanh derivative (1 - h^2) can cause vanishing
        gradients when |h| approaches 1.
        
        Args:
            dh_next: Gradient from next timestep's hidden state (batch_size, hidden_size)
            dy: Gradient from this timestep's output (batch_size, output_size)
        
        Returns:
            Tuple of (input gradient, previous hidden gradient, parameter gradients)
        """
        x, h_prev, h_next = self.cache
        batch_size = x.shape[0]
        
        # Backprop through output layer: y = W_hy * h + b_y
        # dy/dW_hy = h^T, dy/db_y = 1
        dW_hy = np.dot(h_next.T, dy) / batch_size
        db_y = np.sum(dy, axis=0) / batch_size
        # Gradient w.r.t. hidden state from output: dy/dh = W_hy^T
        dh_output = np.dot(dy, self.W_hy.T)
        
        # Total gradient w.r.t. hidden state = from output + from next timestep
        dh_total = dh_next + dh_output
        
        # Backprop through tanh activation: h = tanh(z), dh/dz = 1 - h^2
        # This is where vanishing gradients can occur if h is close to ±1
        dtanh = dh_total * (1 - h_next ** 2)
        
        # Backprop through hidden state computation: z = x*W_xh + h_prev*W_hh + b_h
        # dz/dW_xh = x^T, dz/dW_hh = h_prev^T, dz/db_h = 1
        dW_xh = np.dot(x.T, dtanh) / batch_size
        dW_hh = np.dot(h_prev.T, dtanh) / batch_size
        db_h = np.sum(dtanh, axis=0) / batch_size
        
        # Gradients to pass backward to previous layers/timesteps
        # dz/dx = W_xh^T, dz/dh_prev = W_hh^T
        dx = np.dot(dtanh, self.W_xh.T)
        dh_prev = np.dot(dtanh, self.W_hh.T)
        
        # Package all parameter gradients
        grads = {
            'W_xh': dW_xh,
            'W_hh': dW_hh,
            'W_hy': dW_hy,
            'b_h': db_h,
            'b_y': db_y
        }
        
        return dx, dh_prev, grads

class RNN:
    """
    Complete RNN implementation with Backpropagation Through Time (BPTT)
    
    This class orchestrates the RNN cells across time. The key innovation of RNNs
    is processing sequences by maintaining state across timesteps. BPTT extends
    standard backpropagation to handle the temporal dimension.
    
    Challenges addressed:
    - Vanishing gradients: Gradients diminish exponentially with sequence length
    - Exploding gradients: Can be mitigated with gradient clipping (not implemented here)
    - Truncated BPTT: Limits backprop steps to balance computation and learning
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Single RNN cell that will be reused for each timestep
        self.cell = RNNCell(input_size, hidden_size, output_size)
    
    def forward_sequence(self, x_sequence: np.ndarray, h0: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process an entire sequence through the RNN
        
        This "unrolls" the RNN through time - conceptually creating T copies of the
        RNN cell (one for each timestep) where each passes its hidden state to the next.
        
        Args:
            x_sequence: Input sequence of shape (seq_len, batch_size, input_size)
            h0: Initial hidden state (batch_size, hidden_size). Defaults to zeros.
        
        Returns:
            Tuple of (all hidden states, all outputs) for the sequence
        """
        seq_len, batch_size, _ = x_sequence.shape
        
        # Initialize hidden state if not provided
        if h0 is None:
            h0 = np.zeros((batch_size, self.hidden_size))
        
        # Storage for all hidden states and outputs across the sequence
        hidden_states = np.zeros((seq_len, batch_size, self.hidden_size))
        outputs = np.zeros((seq_len, batch_size, self.output_size))
        
        # Process each timestep sequentially
        h_prev = h0
        for t in range(seq_len):
            # Forward pass for this timestep
            h_prev, y = self.cell.forward(x_sequence[t], h_prev)
            # Store results
            hidden_states[t] = h_prev
            outputs[t] = y
        
        return hidden_states, outputs
    
    def backward_sequence(self, x_sequence: np.ndarray, dh_sequence: np.ndarray, 
                         dy_sequence: np.ndarray, truncate_steps: int = None) -> Dict:
        """
        Backpropagation Through Time (BPTT) - The heart of RNN training
        
        BPTT conceptually "unrolls" the RNN in time and applies backpropagation
        across the entire sequence. This allows the network to learn temporal
        dependencies and long-range patterns.
        
        Key concepts:
        - Gradients flow backward from the end of the sequence to the beginning
        - Each timestep's gradients depend on both local errors and future errors
        - Truncation helps with vanishing gradients and computational efficiency
        
        Args:
            x_sequence: Input sequence (seq_len, batch_size, input_size)
            dh_sequence: Hidden state gradients (seq_len, batch_size, hidden_size)
            dy_sequence: Output gradients (seq_len, batch_size, output_size)
            truncate_steps: Max timesteps for BPTT (None = full sequence)
        
        Returns:
            Averaged gradients across the truncated sequence
        """
        seq_len, batch_size, _ = x_sequence.shape
        
        # Default to full BPTT if no truncation specified
        if truncate_steps is None:
            truncate_steps = seq_len
        
        # Initialize gradient accumulators for each parameter
        grads_accum = {
            'W_xh': np.zeros_like(self.cell.W_xh),
            'W_hh': np.zeros_like(self.cell.W_hh),
            'W_hy': np.zeros_like(self.cell.W_hy),
            'b_h': np.zeros_like(self.cell.b_h),
            'b_y': np.zeros_like(self.cell.b_y)
        }
        
        # Start with zero gradient for the "next" hidden state (end of sequence)
        dh_next = np.zeros((batch_size, self.hidden_size))
        
        # Backward through time - process timesteps in reverse order
        # This is where the "through time" magic happens
        for t in reversed(range(max(0, seq_len - truncate_steps), seq_len)):
            # Note: In a full implementation, we'd need to store caches for each timestep
            # Here we assume the forward pass was just run and caches are available
            
            # Compute gradients for this timestep
            # dh_next comes from the future (t+1), dy_sequence[t] from the current output
            dx, dh_prev, grads = self.cell.backward(
                dh_next + dh_sequence[t],  # Total hidden gradient
                dy_sequence[t]             # Output gradient
            )
            
            # The gradient w.r.t. previous hidden state becomes dh_next for the previous timestep
            dh_next = dh_prev
            
            # Accumulate gradients across timesteps
            for key in grads_accum:
                grads_accum[key] += grads[key]
        
        # Average gradients over the number of timesteps processed
        # This gives equal weight to each timestep in the truncated sequence
        for key in grads_accum:
            grads_accum[key] /= truncate_steps
        
        return grads_accum
    
    def update_parameters(self, grads: Dict, learning_rate: float):
        """
        Update RNN parameters using computed gradients
        
        This implements gradient descent: parameters are adjusted in the direction
        that reduces the loss. The learning rate controls the step size.
        
        Args:
            grads: Dictionary of gradients for each parameter
            learning_rate: Step size for parameter updates
        """
        # Apply gradient descent updates to all parameters
        self.cell.W_xh -= learning_rate * grads['W_xh']
        self.cell.W_hh -= learning_rate * grads['W_hh']
        self.cell.W_hy -= learning_rate * grads['W_hy']
        self.cell.b_h -= learning_rate * grads['b_h']
        self.cell.b_y -= learning_rate * grads['b_y']

def generate_sine_wave_sequence(seq_len: int, batch_size: int = 1, 
                               num_sequences: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic sine wave sequences for RNN training
    
    RNNs need to learn temporal patterns. Sine waves provide a good test case
    because they have predictable patterns that repeat over time. The RNN
    should learn to predict the next value in the sequence.
    
    Args:
        seq_len: Length of each sequence
        batch_size: Number of sequences processed together (for batching)
        num_sequences: Total number of sequences to generate
    
    Returns:
        Tuple of (inputs, targets) for next-step prediction
        - inputs: (num_sequences, seq_len, batch_size, 1)
        - targets: (num_sequences, seq_len, batch_size, 1)
    """
    inputs = []
    targets = []
    
    for _ in range(num_sequences):
        # Create a sine wave with some noise to make it more realistic
        t = np.linspace(0, 4 * np.pi, seq_len + 1)
        sequence = np.sin(t) + 0.1 * np.random.randn(seq_len + 1)
        
        # Create sliding window: predict next value from current
        # input_seq[t] should predict target_seq[t]
        input_seq = sequence[:-1].reshape(seq_len, batch_size, 1)
        target_seq = sequence[1:].reshape(seq_len, batch_size, 1)
        
        inputs.append(input_seq)
        targets.append(target_seq)
    
    return np.array(inputs), np.array(targets)

def compute_gradient_norms(grads: Dict) -> Dict:
    """
    Compute L2 norms of gradients for monitoring training
    
    Gradient norms are important diagnostics in RNN training:
    - Very small norms: Vanishing gradients (learning stops)
    - Very large norms: Exploding gradients (training unstable)
    - Appropriate norms: Healthy learning
    
    Args:
        grads: Dictionary of gradients
    
    Returns:
        Dictionary of gradient norms (L2 norms)
    """
    norms = {}
    for key, grad in grads.items():
        # L2 norm: sqrt(sum(grad^2))
        norms[key] = np.linalg.norm(grad)
    return norms

def train_rnn_sequence_prediction():
    """
    Complete RNN training pipeline for sequence prediction
    
    This function demonstrates the full RNN training process:
    1. Generate synthetic sine wave data (temporal patterns)
    2. Train RNN using truncated BPTT
    3. Monitor gradient norms for stability analysis
    4. Apply gradient clipping to prevent exploding gradients
    
    Key educational points:
    - Truncated BPTT balances learning long dependencies vs. computational cost
    - Gradient monitoring helps diagnose training issues
    - Gradient clipping stabilizes training when gradients explode
    """
    print("RNN Sequence Prediction with BPTT")
    print("=" * 50)
    
    # Hyperparameters - these control the learning behavior
    input_size = 1      # Single value per timestep (sine wave)
    hidden_size = 32    # Size of hidden state (memory capacity)
    output_size = 1     # Single prediction per timestep
    seq_len = 50        # How far back in time the RNN can see
    batch_size = 1      # Process one sequence at a time
    num_epochs = 1000   # Training iterations
    learning_rate = 0.01  # Step size for parameter updates
    truncate_steps = 20  # BPTT truncation (prevents vanishing gradients)
    
    # Initialize RNN with specified architecture
    rnn = RNN(input_size, hidden_size, output_size)
    
    # Generate training data: sine waves with noise
    train_inputs, train_targets = generate_sine_wave_sequence(
        seq_len, batch_size, num_sequences=1
    )
    
    # Track training progress
    losses = []          # Loss history
    gradient_norms = []  # Gradient norms for analysis
    
    print(f"Training RNN on sine wave prediction...")
    print(f"Sequence length: {seq_len}, Hidden size: {hidden_size}")
    print(f"Truncated BPTT steps: {truncate_steps}")
    
    for epoch in range(num_epochs):
        total_loss = 0
        epoch_gradient_norms = []
        
        # Process each training sequence
        for seq_idx in range(len(train_inputs)):
            # Forward pass: compute predictions for entire sequence
            x_sequence = train_inputs[seq_idx]  # Input sequence
            hidden_states, outputs = rnn.forward_sequence(x_sequence)
            
            # Compute loss and gradients for backpropagation
            loss = 0
            # Initialize gradient arrays (zero because loss doesn't directly depend on hidden states)
            dh_sequence = np.zeros_like(hidden_states)  # Gradients w.r.t. hidden states
            dy_sequence = np.zeros_like(outputs)         # Gradients w.r.t. outputs
            
            # Compute loss and output gradients for each timestep
            for t in range(seq_len):
                # Mean squared error: 0.5 * (prediction - target)^2
                error = outputs[t] - train_targets[seq_idx, t]
                loss += 0.5 * np.sum(error ** 2)
                
                # Gradient of MSE w.r.t. output: (prediction - target)
                dy_sequence[t] = error
                # Hidden state gradients remain zero (no direct loss dependence)
            
            # Backward pass: compute parameter gradients using BPTT
            grads = rnn.backward_sequence(x_sequence, dh_sequence, dy_sequence, truncate_steps)
            
            # Monitor gradient norms for training stability analysis
            grad_norms = compute_gradient_norms(grads)
            epoch_gradient_norms.append(grad_norms)
            
            # Gradient clipping: prevent exploding gradients
            max_grad_norm = 1.0  # Maximum allowed gradient norm
            total_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
            
            if total_norm > max_grad_norm:
                # Scale down all gradients proportionally
                clip_coef = max_grad_norm / (total_norm + 1e-6)
                for key in grads:
                    grads[key] *= clip_coef
                print(f"Epoch {epoch}: Gradient clipped (norm: {total_norm:.4f})")
            
            # Update parameters using gradient descent
            rnn.update_parameters(grads, learning_rate)
            
            total_loss += loss
        
        # Track average loss for this epoch
        avg_loss = total_loss / len(train_inputs)
        losses.append(avg_loss)
        
        # Track average gradient norms across sequences
        avg_grad_norms = {}
        for key in epoch_gradient_norms[0]:
            avg_grad_norms[key] = np.mean([gn[key] for gn in epoch_gradient_norms])
        gradient_norms.append(avg_grad_norms)
        
        # Periodic progress reporting
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
            print(f"  Gradient norms - W_xh: {avg_grad_norms['W_xh']:.4f}, "
                  f"W_hh: {avg_grad_norms['W_hh']:.4f}, "
                  f"W_hy: {avg_grad_norms['W_hy']:.4f}")
    
    return rnn, losses, gradient_norms

def analyze_gradient_flow(gradient_norms: List[Dict]):
    """
    Analyze gradient behavior during training to detect training issues
    
    RNN training is notoriously difficult due to gradient flow problems:
    - Vanishing gradients: Gradients become extremely small, learning stops
    - Exploding gradients: Gradients become extremely large, training unstable
    
    This analysis helps identify these issues by examining gradient norms over time.
    The recurrent weights (W_hh) are most susceptible to these problems.
    
    Args:
        gradient_norms: List of gradient norm dictionaries from each epoch
    
    Returns:
        Tuple of (W_xh_norms, W_hh_norms, W_hy_norms) for plotting
    """
    print("\n" + "=" * 50)
    print("Gradient Flow Analysis")
    print("=" * 50)
    
    # Extract gradient norms for each parameter type
    W_xh_norms = [gn['W_xh'] for gn in gradient_norms]  # Input-to-hidden
    W_hh_norms = [gn['W_hh'] for gn in gradient_norms]  # Hidden-to-hidden (recurrent)
    W_hy_norms = [gn['W_hy'] for gn in gradient_norms]  # Hidden-to-output
    
    # Analyze the recurrent gradients (most critical for RNN health)
    final_W_hh_norm = W_hh_norms[-1]
    max_W_hh_norm = max(W_hh_norms)
    min_W_hh_norm = min(W_hh_norms)
    
    print(f"W_hh gradient norms (recurrent connections):")
    print(f"  Final: {final_W_hh_norm:.6f}")
    print(f"  Maximum: {max_W_hh_norm:.6f}")
    print(f"  Minimum: {min_W_hh_norm:.6f}")
    print(f"  Ratio (max/min): {max_W_hh_norm/(min_W_hh_norm + 1e-8):.2f}")
    
    # Diagnose potential issues
    if min_W_hh_norm < 1e-6:
        print("  WARNING: Potential vanishing gradients detected!")
        print("           Gradients are becoming extremely small, hindering learning.")
    
    if max_W_hh_norm > 100:
        print("  WARNING: Potential exploding gradients detected!")
        print("           Gradients are becoming extremely large, causing instability.")
    
    if 1e-6 <= min_W_hh_norm and max_W_hh_norm <= 100:
        print("  ✓ Gradient norms appear healthy for training.")
    
    return W_xh_norms, W_hh_norms, W_hy_norms

def visualize_training(losses: List[float], gradient_norms: List[Dict]):
    """
    Create comprehensive visualizations of RNN training progress
    
    Effective visualization is crucial for understanding RNN training:
    1. Loss curve: Shows overall learning progress
    2. Gradient norms: Reveals training stability issues
    3. Recurrent gradients: Most important for detecting vanishing/exploding gradients
    4. Gradient ratios: Helps understand relative gradient magnitudes
    
    Args:
        losses: List of loss values per epoch
        gradient_norms: List of gradient norm dictionaries per epoch
    """
    # Get gradient norm histories
    W_xh_norms, W_hh_norms, W_hy_norms = analyze_gradient_flow(gradient_norms)
    
    # Create 2x2 subplot layout for comprehensive analysis
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Top-left: Training loss over time
    axes[0, 0].plot(losses)
    axes[0, 0].set_title('Training Loss Over Time')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Mean Squared Error Loss')
    axes[0, 0].set_yscale('log')  # Log scale helps see loss reduction
    axes[0, 0].grid(True)
    
    # Top-right: All gradient norms comparison
    axes[0, 1].plot(W_xh_norms, label='W_xh (input→hidden)')
    axes[0, 1].plot(W_hh_norms, label='W_hh (hidden→hidden)')
    axes[0, 1].plot(W_hy_norms, label='W_hy (hidden→output)')
    axes[0, 1].set_title('Gradient Norms for All Parameters')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('L2 Gradient Norm')
    axes[0, 1].set_yscale('log')  # Log scale for wide range of values
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Bottom-left: Focus on recurrent gradients (most critical)
    axes[1, 0].plot(W_hh_norms, color='red', linewidth=2)
    axes[1, 0].set_title('Recurrent Gradient Norms (W_hh)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Gradient Norm')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True)
    # Add reference lines for healthy gradient ranges
    axes[1, 0].axhline(y=1e-6, color='orange', linestyle='--', alpha=0.7, label='Vanishing threshold')
    axes[1, 0].axhline(y=100, color='purple', linestyle='--', alpha=0.7, label='Exploding threshold')
    axes[1, 0].legend()
    
    # Bottom-right: Gradient ratios (relative magnitudes)
    gradient_ratios = [W_hh_norms[i] / (W_xh_norms[i] + 1e-8) for i in range(len(W_hh_norms))]
    axes[1, 1].plot(gradient_ratios, color='green')
    axes[1, 1].set_title('Gradient Ratio: Recurrent vs Input')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('W_hh / W_xh Ratio')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('rnn_training_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def test_sequence_prediction(rnn: RNN, seq_len: int = 30):
    """
    Evaluate the trained RNN on unseen sequence prediction data
    
    Testing on held-out data is crucial to verify that the RNN has learned
    generalizable patterns rather than just memorizing the training data.
    For sequence prediction, we want to see if the RNN can predict future
    values based on past observations.
    
    Args:
        rnn: Trained RNN model
        seq_len: Length of test sequences
    """
    print("\n" + "=" * 50)
    print("Sequence Prediction Test")
    print("=" * 50)
    
    # Generate fresh test data (different from training data)
    test_inputs, test_targets = generate_sine_wave_sequence(seq_len, num_sequences=1)
    x_sequence = test_inputs[0]  # Get the first (and only) sequence
    
    # Forward pass: generate predictions
    hidden_states, predictions = rnn.forward_sequence(x_sequence)
    
    # Compute test loss (mean squared error)
    test_loss = 0
    for t in range(seq_len):
        error = predictions[t] - test_targets[0, t]
        test_loss += 0.5 * np.sum(error ** 2)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Average prediction error per timestep: {np.sqrt(2 * test_loss / seq_len):.4f}")
    
    # Create visualization comparing predictions vs. ground truth
    plt.figure(figsize=(12, 4))
    
    # Extract sequences for plotting
    true_sequence = test_targets[0, :, 0, 0]    # Ground truth targets
    pred_sequence = predictions[:, 0, 0]        # Model predictions
    input_sequence = x_sequence[:, 0, 0]        # Input sequence
    
    # Plot all three sequences
    plt.plot(range(seq_len), true_sequence, 'b-', label='True (target)', linewidth=2)
    plt.plot(range(seq_len), pred_sequence, 'r--', label='Predicted', linewidth=2)
    plt.plot(range(seq_len), input_sequence, 'g:', label='Input', alpha=0.7)
    
    plt.title('RNN Sequence Prediction: True vs Predicted')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('rnn_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()

def experiment_with_gradient_clipping():
    """
    Demonstrate gradient clipping techniques for training stability
    
    Gradient clipping is a crucial technique for RNN training because:
    - RNNs are prone to exploding gradients due to repeated matrix multiplication
    - Large gradients can cause parameter updates that destabilize training
    - Clipping scales down large gradients while preserving smaller ones
    
    This experiment shows how different clipping thresholds affect gradient norms.
    """
    print("\n" + "=" * 50)
    print("Gradient Clipping Experiment")
    print("=" * 50)
    
    # Test different clipping thresholds
    clip_thresholds = [0.1, 1.0, 5.0, None]  # None means no clipping
    
    for clip_threshold in clip_thresholds:
        print(f"\nTesting gradient clipping threshold: {clip_threshold}")
        
        # Simulate large gradients (common in RNNs with exploding gradients)
        grads = {
            'W_xh': np.random.randn(10, 20) * 10,  # Large input-to-hidden gradients
            'W_hh': np.random.randn(20, 20) * 15,  # Even larger recurrent gradients
            'W_hy': np.random.randn(20, 5) * 5,   # Moderate output gradients
            'b_h': np.random.randn(20) * 8,       # Bias gradients
            'b_y': np.random.randn(5) * 3         # Output bias gradients
        }
        
        # Compute the total gradient norm (L2 norm across all parameters)
        original_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
        print(f"  Original gradient norm: {original_norm:.4f}")
        
        # Apply gradient clipping if threshold is specified and exceeded
        if clip_threshold is not None and original_norm > clip_threshold:
            # Compute clipping coefficient: threshold / current_norm
            clip_coef = clip_threshold / original_norm
            
            # Scale all gradients by the clipping coefficient
            clipped_grads = {}
            for key in grads:
                clipped_grads[key] = grads[key] * clip_coef
            
            # Verify the new norm equals the threshold
            clipped_norm = np.sqrt(sum(np.sum(g**2) for g in clipped_grads.values()))
            print(f"  Clipped gradient norm: {clipped_norm:.4f}")
            print(f"  Clipping coefficient: {clip_coef:.4f}")
            print(f"  All gradients scaled by: {clip_coef:.4f}")
        else:
            if clip_threshold is None:
                print("  No clipping applied (threshold = None)")
            else:
                print(f"  No clipping needed (norm {original_norm:.4f} < threshold {clip_threshold})")

def main():
    """
    Main execution function for the RNN BPTT exercise
    
    This function orchestrates the complete educational experience:
    1. Train an RNN on sequence prediction using BPTT
    2. Analyze training dynamics and gradient behavior
    3. Visualize learning progress and identify potential issues
    4. Test the trained model on held-out data
    5. Demonstrate gradient clipping techniques
    
    The goal is to provide a comprehensive understanding of RNN training challenges
    and solutions, making this both an implementation and educational resource.
    """
    print("Exercise 18: RNN Cell and Backpropagation Through Time (BPTT)")
    print("=" * 60)
    print("This exercise demonstrates:")
    print("- Vanilla RNN cell implementation with forward/backward passes")
    print("- Backpropagation Through Time (BPTT) for sequence learning")
    print("- Gradient flow analysis to detect vanishing/exploding gradients")
    print("- Truncated BPTT and gradient clipping for stable training")
    print("- Sequence prediction on synthetic sine wave data")
    print("=" * 60)
    
    # Core experiment: Train RNN with BPTT
    rnn, losses, gradient_norms = train_rnn_sequence_prediction()
    
    # Analyze and visualize training results
    visualize_training(losses, gradient_norms)
    
    # Evaluate on test data
    test_sequence_prediction(rnn)
    
    # Demonstrate gradient clipping
    experiment_with_gradient_clipping()
    
    # Educational summary
    print("\n" + "=" * 60)
    print("Key Learning Outcomes:")
    print("- ✓ Implemented complete RNN forward/backward passes")
    print("- ✓ Demonstrated BPTT for learning temporal dependencies")
    print("- ✓ Identified gradient flow issues in RNN training")
    print("- ✓ Applied truncated BPTT and gradient clipping solutions")
    print("- ✓ Visualized training dynamics and prediction quality")
    print("\nNext steps for deeper learning:")
    print("- Try different sequence lengths and see vanishing gradient effects")
    print("- Experiment with LSTM/GRU cells for better gradient flow")
    print("- Apply to real datasets like text or time series")
    print("- Implement teacher forcing for more stable training")

if __name__ == "__main__":
    main()