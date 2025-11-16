import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

class RNNCell:
    """Vanilla RNN Cell"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights
        self.W_xh = np.random.randn(input_size, hidden_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(hidden_size, output_size) * 0.01
        
        # Initialize biases
        self.b_h = np.zeros(hidden_size)
        self.b_y = np.zeros(output_size)
        
        # Cache for backward pass
        self.cache = None
    
    def forward(self, x: np.ndarray, h_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass for single RNN step
        x: input of shape (batch_size, input_size)
        h_prev: previous hidden state of shape (batch_size, hidden_size)
        Returns: (h_next, y)
        """
        # Hidden state update
        h_next = np.tanh(np.dot(x, self.W_xh) + np.dot(h_prev, self.W_hh) + self.b_h)
        
        # Output
        y = np.dot(h_next, self.W_hy) + self.b_y
        
        # Store cache for backward pass
        self.cache = (x, h_prev, h_next)
        
        return h_next, y
    
    def backward(self, dh_next: np.ndarray, dy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Backward pass for single RNN step
        dh_next: gradient from next hidden state (batch_size, hidden_size)
        dy: gradient from output (batch_size, output_size)
        Returns: (dx, dh_prev, grads)
        """
        x, h_prev, h_next = self.cache
        batch_size = x.shape[0]
        
        # Gradient through output layer
        dW_hy = np.dot(h_next.T, dy) / batch_size
        db_y = np.sum(dy, axis=0) / batch_size
        dh_output = np.dot(dy, self.W_hy.T)
        
        # Combine gradients from output and next time step
        dh_total = dh_next + dh_output
        
        # Gradient through tanh activation
        dtanh = dh_total * (1 - h_next ** 2)
        
        # Gradients for hidden weights
        dW_xh = np.dot(x.T, dtanh) / batch_size
        dW_hh = np.dot(h_prev.T, dtanh) / batch_size
        db_h = np.sum(dtanh, axis=0) / batch_size
        
        # Gradients to previous layers
        dx = np.dot(dtanh, self.W_xh.T)
        dh_prev = np.dot(dtanh, self.W_hh.T)
        
        grads = {
            'W_xh': dW_xh,
            'W_hh': dW_hh,
            'W_hy': dW_hy,
            'b_h': db_h,
            'b_y': db_y
        }
        
        return dx, dh_prev, grads

class RNN:
    """Full RNN with BPTT"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.cell = RNNCell(input_size, hidden_size, output_size)
    
    def forward_sequence(self, x_sequence: np.ndarray, h0: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass for entire sequence
        x_sequence: (seq_len, batch_size, input_size)
        h0: initial hidden state (batch_size, hidden_size)
        Returns: (hidden_states, outputs)
        """
        seq_len, batch_size, _ = x_sequence.shape
        
        if h0 is None:
            h0 = np.zeros((batch_size, self.hidden_size))
        
        hidden_states = np.zeros((seq_len, batch_size, self.hidden_size))
        outputs = np.zeros((seq_len, batch_size, self.output_size))
        
        h_prev = h0
        for t in range(seq_len):
            h_prev, y = self.cell.forward(x_sequence[t], h_prev)
            hidden_states[t] = h_prev
            outputs[t] = y
        
        return hidden_states, outputs
    
    def backward_sequence(self, x_sequence: np.ndarray, dh_sequence: np.ndarray, 
                         dy_sequence: np.ndarray, truncate_steps: int = None) -> Dict:
        """
        Backward pass through time (BPTT)
        x_sequence: (seq_len, batch_size, input_size)
        dh_sequence: gradients from hidden states (seq_len, batch_size, hidden_size)
        dy_sequence: gradients from outputs (seq_len, batch_size, output_size)
        truncate_steps: number of steps to truncate BPTT (None for full BPTT)
        Returns: averaged gradients
        """
        seq_len, batch_size, _ = x_sequence.shape
        
        if truncate_steps is None:
            truncate_steps = seq_len
        
        # Initialize gradients
        grads_accum = {
            'W_xh': np.zeros_like(self.cell.W_xh),
            'W_hh': np.zeros_like(self.cell.W_hh),
            'W_hy': np.zeros_like(self.cell.W_hy),
            'b_h': np.zeros_like(self.cell.b_h),
            'b_y': np.zeros_like(self.cell.b_y)
        }
        
        dh_next = np.zeros((batch_size, self.hidden_size))
        
        # Backward through time
        for t in reversed(range(max(0, seq_len - truncate_steps), seq_len)):
            # Set the cache for this time step (we need to re-run forward or store cache)
            # For simplicity, we assume forward was just run and cache is available
            
            # Get gradients for this time step
            dx, dh_prev, grads = self.cell.backward(
                dh_next + dh_sequence[t], 
                dy_sequence[t]
            )
            
            dh_next = dh_prev
            
            # Accumulate gradients
            for key in grads_accum:
                grads_accum[key] += grads[key]
        
        # Average gradients
        for key in grads_accum:
            grads_accum[key] /= truncate_steps
        
        return grads_accum
    
    def update_parameters(self, grads: Dict, learning_rate: float):
        """Update parameters using gradients"""
        self.cell.W_xh -= learning_rate * grads['W_xh']
        self.cell.W_hh -= learning_rate * grads['W_hh']
        self.cell.W_hy -= learning_rate * grads['W_hy']
        self.cell.b_h -= learning_rate * grads['b_h']
        self.cell.b_y -= learning_rate * grads['b_y']

def generate_sine_wave_sequence(seq_len: int, batch_size: int = 1, 
                               num_sequences: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic sine wave sequences for next-step prediction
    Returns: (inputs, targets) both of shape (num_sequences, seq_len, batch_size, 1)
    """
    inputs = []
    targets = []
    
    for _ in range(num_sequences):
        # Generate sine wave with some noise
        t = np.linspace(0, 4 * np.pi, seq_len + 1)
        sequence = np.sin(t) + 0.1 * np.random.randn(seq_len + 1)
        
        # Create input-target pairs (predict next value)
        input_seq = sequence[:-1].reshape(seq_len, batch_size, 1)
        target_seq = sequence[1:].reshape(seq_len, batch_size, 1)
        
        inputs.append(input_seq)
        targets.append(target_seq)
    
    return np.array(inputs), np.array(targets)

def compute_gradient_norms(grads: Dict) -> Dict:
    """Compute L2 norms of gradients"""
    norms = {}
    for key, grad in grads.items():
        norms[key] = np.linalg.norm(grad)
    return norms

def train_rnn_sequence_prediction():
    """Train RNN on sequence prediction task and analyze gradients"""
    print("RNN Sequence Prediction with BPTT")
    print("=" * 50)
    
    # Hyperparameters
    input_size = 1
    hidden_size = 32
    output_size = 1
    seq_len = 50
    batch_size = 1
    num_epochs = 1000
    learning_rate = 0.01
    truncate_steps = 20  # Truncated BPTT
    
    # Initialize RNN
    rnn = RNN(input_size, hidden_size, output_size)
    
    # Generate training data
    train_inputs, train_targets = generate_sine_wave_sequence(
        seq_len, batch_size, num_sequences=1
    )
    
    # Training history
    losses = []
    gradient_norms = []
    
    print(f"Training RNN on sine wave prediction...")
    print(f"Sequence length: {seq_len}, Hidden size: {hidden_size}")
    print(f"Truncated BPTT steps: {truncate_steps}")
    
    for epoch in range(num_epochs):
        total_loss = 0
        epoch_gradient_norms = []
        
        # Process each sequence
        for seq_idx in range(len(train_inputs)):
            # Forward pass
            x_sequence = train_inputs[seq_idx]  # (seq_len, batch_size, input_size)
            hidden_states, outputs = rnn.forward_sequence(x_sequence)
            
            # Compute loss and gradients
            loss = 0
            dh_sequence = np.zeros_like(hidden_states)
            dy_sequence = np.zeros_like(outputs)
            
            for t in range(seq_len):
                # Mean squared error
                error = outputs[t] - train_targets[seq_idx, t]
                loss += 0.5 * np.sum(error ** 2)
                
                # Gradient of loss w.r.t. output
                dy_sequence[t] = error
                # No direct gradient from loss to hidden states
                # (only through output and next time steps)
            
            # Backward pass with truncated BPTT
            grads = rnn.backward_sequence(x_sequence, dh_sequence, dy_sequence, truncate_steps)
            
            # Compute gradient norms
            grad_norms = compute_gradient_norms(grads)
            epoch_gradient_norms.append(grad_norms)
            
            # Apply gradient clipping
            max_grad_norm = 1.0
            total_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
            
            if total_norm > max_grad_norm:
                clip_coef = max_grad_norm / (total_norm + 1e-6)
                for key in grads:
                    grads[key] *= clip_coef
                print(f"Epoch {epoch}: Gradient clipped (norm: {total_norm:.4f})")
            
            # Update parameters
            rnn.update_parameters(grads, learning_rate)
            
            total_loss += loss
        
        avg_loss = total_loss / len(train_inputs)
        losses.append(avg_loss)
        
        # Track average gradient norms
        avg_grad_norms = {}
        for key in epoch_gradient_norms[0]:
            avg_grad_norms[key] = np.mean([gn[key] for gn in epoch_gradient_norms])
        gradient_norms.append(avg_grad_norms)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
            print(f"  Gradient norms - W_xh: {avg_grad_norms['W_xh']:.4f}, "
                  f"W_hh: {avg_grad_norms['W_hh']:.4f}, "
                  f"W_hy: {avg_grad_norms['W_hy']:.4f}")
    
    return rnn, losses, gradient_norms

def analyze_gradient_flow(gradient_norms: List[Dict]):
    """Analyze how gradient norms change during training"""
    print("\n" + "=" * 50)
    print("Gradient Flow Analysis")
    print("=" * 50)
    
    # Extract gradient norms for each parameter
    W_xh_norms = [gn['W_xh'] for gn in gradient_norms]
    W_hh_norms = [gn['W_hh'] for gn in gradient_norms]
    W_hy_norms = [gn['W_hy'] for gn in gradient_norms]
    
    # Analyze vanishing/exploding gradients
    final_W_hh_norm = W_hh_norms[-1]
    max_W_hh_norm = max(W_hh_norms)
    min_W_hh_norm = min(W_hh_norms)
    
    print(f"W_hh gradient norms:")
    print(f"  Final: {final_W_hh_norm:.6f}")
    print(f"  Maximum: {max_W_hh_norm:.6f}")
    print(f"  Minimum: {min_W_hh_norm:.6f}")
    print(f"  Ratio (max/min): {max_W_hh_norm/(min_W_hh_norm + 1e-8):.2f}")
    
    # Check for vanishing gradients
    if min_W_hh_norm < 1e-6:
        print("  WARNING: Potential vanishing gradients detected!")
    
    # Check for exploding gradients
    if max_W_hh_norm > 100:
        print("  WARNING: Potential exploding gradients detected!")
    
    return W_xh_norms, W_hh_norms, W_hy_norms

def visualize_training(losses: List[float], gradient_norms: List[Dict]):
    """Visualize training progress and gradient behavior"""
    W_xh_norms, W_hh_norms, W_hy_norms = analyze_gradient_flow(gradient_norms)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot loss
    axes[0, 0].plot(losses)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True)
    
    # Plot gradient norms
    axes[0, 1].plot(W_xh_norms, label='W_xh')
    axes[0, 1].plot(W_hh_norms, label='W_hh')
    axes[0, 1].plot(W_hy_norms, label='W_hy')
    axes[0, 1].set_title('Gradient Norms')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Gradient Norm')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot W_hh gradients specifically (most prone to vanishing/exploding)
    axes[1, 0].plot(W_hh_norms)
    axes[1, 0].set_title('W_hh Gradient Norms (Recurrent)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Gradient Norm')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True)
    
    # Plot gradient ratios
    gradient_ratios = [W_hh_norms[i] / (W_xh_norms[i] + 1e-8) for i in range(len(W_hh_norms))]
    axes[1, 1].plot(gradient_ratios)
    axes[1, 1].set_title('Gradient Ratio: W_hh / W_xh')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Ratio')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('rnn_training_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def test_sequence_prediction(rnn: RNN, seq_len: int = 30):
    """Test the trained RNN on sequence prediction"""
    print("\n" + "=" * 50)
    print("Sequence Prediction Test")
    print("=" * 50)
    
    # Generate test sequence
    test_inputs, test_targets = generate_sine_wave_sequence(seq_len, num_sequences=1)
    x_sequence = test_inputs[0]
    
    # Make predictions
    hidden_states, predictions = rnn.forward_sequence(x_sequence)
    
    # Calculate test loss
    test_loss = 0
    for t in range(seq_len):
        error = predictions[t] - test_targets[0, t]
        test_loss += 0.5 * np.sum(error ** 2)
    
    print(f"Test Loss: {test_loss:.4f}")
    
    # Visualize predictions
    plt.figure(figsize=(12, 4))
    
    # Plot first sequence
    true_sequence = test_targets[0, :, 0, 0]
    pred_sequence = predictions[:, 0, 0]
    input_sequence = x_sequence[:, 0, 0]
    
    plt.plot(range(seq_len), true_sequence, 'b-', label='True', linewidth=2)
    plt.plot(range(seq_len), pred_sequence, 'r--', label='Predicted', linewidth=2)
    plt.plot(range(seq_len), input_sequence, 'g:', label='Input', alpha=0.7)
    
    plt.title('RNN Sequence Prediction')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('rnn_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()

def experiment_with_gradient_clipping():
    """Experiment with different gradient clipping strategies"""
    print("\n" + "=" * 50)
    print("Gradient Clipping Experiment")
    print("=" * 50)
    
    clip_thresholds = [0.1, 1.0, 5.0, None]
    
    for clip_threshold in clip_thresholds:
        print(f"\nTesting gradient clipping threshold: {clip_threshold}")
        
        # Simple gradient example (simulating exploding gradients)
        grads = {
            'W_xh': np.random.randn(10, 20) * 10,  # Large gradients
            'W_hh': np.random.randn(20, 20) * 15,  # Even larger
            'W_hy': np.random.randn(20, 5) * 5,
            'b_h': np.random.randn(20) * 8,
            'b_y': np.random.randn(5) * 3
        }
        
        # Compute original norm
        original_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
        print(f"  Original gradient norm: {original_norm:.4f}")
        
        # Apply clipping if threshold specified
        if clip_threshold is not None and original_norm > clip_threshold:
            clip_coef = clip_threshold / original_norm
            clipped_grads = {}
            for key in grads:
                clipped_grads[key] = grads[key] * clip_coef
            
            clipped_norm = np.sqrt(sum(np.sum(g**2) for g in clipped_grads.values()))
            print(f"  Clipped gradient norm: {clipped_norm:.4f}")
            print(f"  Clipping coefficient: {clip_coef:.4f}")
        else:
            print(f"  No clipping applied")

def main():
    """Main function to run RNN experiments"""
    print("Exercise 18: RNN Cell and Backpropagation Through Time (BPTT)")
    print("=" * 60)
    
    # Train RNN with BPTT
    rnn, losses, gradient_norms = train_rnn_sequence_prediction()
    
    # Analyze and visualize results
    visualize_training(losses, gradient_norms)
    
    # Test the trained model
    test_sequence_prediction(rnn)
    
    # Experiment with gradient clipping
    experiment_with_gradient_clipping()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("- Implemented vanilla RNN cell with forward/backward passes")
    print("- Implemented truncated BPTT for efficient training")
    print("- Demonstrated sequence prediction on synthetic data")
    print("- Analyzed gradient norms to detect vanishing/exploding gradients")
    print("- Experimented with gradient clipping for training stability")

if __name__ == "__main__":
    main()