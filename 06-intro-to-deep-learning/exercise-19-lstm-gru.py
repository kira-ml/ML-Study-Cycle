"""
Exercise 19 — LSTM & GRU Internals
Filename: exercise-19-lstm-gru.py

Core Concept: Understanding gated architectures (forget/update gates) and how they mitigate vanishing gradients.

This implementation demonstrates:
1. LSTM and GRU cell forward/backward passes (vectorized)
2. Why gated architectures help with vanishing gradients
3. Training stability comparison with vanilla RNN

Key Insights:
- Gates control information flow through time
- Forget gates decide what to remember/forget
- Update gates control how much new info to incorporate
- Gates create paths where gradients don't vanish as quickly
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import seaborn as sns

class LSTMCell:
    """
    Long Short-Term Memory Cell with complete vectorized implementation.
    
    Key components:
    - Input gate: Controls what new information to store
    - Forget gate: Controls what old information to discard  
    - Output gate: Controls what information to output
    - Cell state: Long-term memory that flows through time with minimal transformations
    
    Why LSTM mitigates vanishing gradients:
    The cell state creates a "highway" where gradients can flow with minimal attenuation.
    Forget gates can learn to keep values close to 1, allowing gradients to propagate.
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        # Initialize all gates and transformations simultaneously
        # Shape: [4 * hidden_size, input_size + hidden_size]
        # Order: [input_gate, forget_gate, output_gate, candidate]
        self.W = np.random.randn(4 * hidden_size, input_size + hidden_size) * 0.01
        self.b = np.zeros((4 * hidden_size, 1))
        
        self.hidden_size = hidden_size
        self.cache = None
        
    def forward(self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass for LSTM cell.
        
        Args:
            x: Input of shape (input_size, batch_size)
            h_prev: Previous hidden state (hidden_size, batch_size)
            c_prev: Previous cell state (hidden_size, batch_size)
            
        Returns:
            h_next: Next hidden state
            c_next: Next cell state
        """
        batch_size = x.shape[1]
        
        # Concatenate input and previous hidden state
        concat = np.vstack((h_prev, x))  # Shape: (hidden_size + input_size, batch_size)
        
        # Compute all gates and candidate simultaneously
        gates = self.W @ concat + self.b  # Shape: (4 * hidden_size, batch_size)
        
        # Split into individual components
        split_size = self.hidden_size
        input_gate = sigmoid(gates[0:split_size])
        forget_gate = sigmoid(gates[split_size:2*split_size])
        output_gate = sigmoid(gates[2*split_size:3*split_size])
        candidate = np.tanh(gates[3*split_size:4*split_size])
        
        # Update cell state: c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
        # This is the key equation - forget gate controls gradient flow
        c_next = forget_gate * c_prev + input_gate * candidate
        
        # Compute next hidden state
        h_next = output_gate * np.tanh(c_next)
        
        # Store intermediate values for backward pass
        self.cache = (x, h_prev, c_prev, input_gate, forget_gate, output_gate, candidate, c_next, concat)
        
        return h_next, c_next
    
    def backward(self, dh_next: np.ndarray, dc_next: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass for LSTM cell.
        
        Args:
            dh_next: Gradient from next hidden state
            dc_next: Gradient from next cell state
            
        Returns:
            dx: Gradient w.r.t input
            dh_prev: Gradient w.r.t previous hidden state  
            dc_prev: Gradient w.r.t previous cell state
        """
        # Retrieve cached values
        x, h_prev, c_prev, i_gate, f_gate, o_gate, candidate, c_next, concat = self.cache
        
        # Gradient through output gate and tanh
        dtanh_c = dh_next * o_gate
        dc_next += dtanh_c * (1 - np.tanh(c_next) ** 2)
        
        # Gradient through cell state update
        dc_prev = dc_next * f_gate
        df_gate = dc_next * c_prev
        di_gate = dc_next * candidate
        dcandidate = dc_next * i_gate
        
        # Gate gradients through sigmoid
        di_gate_raw = di_gate * i_gate * (1 - i_gate)
        df_gate_raw = df_gate * f_gate * (1 - f_gate)
        do_gate_raw = dh_next * np.tanh(c_next) * o_gate * (1 - o_gate)
        
        # Candidate gradient through tanh
        dcandidate_raw = dcandidate * (1 - candidate ** 2)
        
        # Concatenate all gate gradients
        dgates = np.vstack((di_gate_raw, df_gate_raw, do_gate_raw, dcandidate_raw))
        
        # Gradients for weights and biases
        self.dW = dgates @ concat.T
        self.db = np.sum(dgates, axis=1, keepdims=True)
        
        # Gradient through concatenation
        dconcat = self.W.T @ dgates
        
        # Split into hidden and input gradients
        dh_prev = dconcat[:self.hidden_size]
        dx = dconcat[self.hidden_size:]
        
        return dx, dh_prev, dc_prev


class GRUCell:
    """
    Gated Recurrent Unit Cell - simplified gated architecture.
    
    Key components:
    - Update gate: Controls how much to update hidden state (replaces LSTM's input/forget gates)
    - Reset gate: Controls how much past information to use for new candidate
    - Candidate activation: Proposed new hidden state value
    
    Why GRU mitigates vanishing gradients:
    Update gate creates adaptive paths - when z_t ≈ 1, gradients flow directly through time.
    Simpler than LSTM but often performs similarly.
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        # Update gate, reset gate, and candidate weights
        self.W_z = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.W_r = np.random.randn(hidden_size, input_size + hidden_size) * 0.01  
        self.W_h = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        
        self.b_z = np.zeros((hidden_size, 1))
        self.b_r = np.zeros((hidden_size, 1))
        self.b_h = np.zeros((hidden_size, 1))
        
        self.hidden_size = hidden_size
        self.cache = None
        
    def forward(self, x: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        """
        Forward pass for GRU cell.
        
        Args:
            x: Input of shape (input_size, batch_size)
            h_prev: Previous hidden state (hidden_size, batch_size)
            
        Returns:
            h_next: Next hidden state
        """
        batch_size = x.shape[1]
        concat = np.vstack((h_prev, x))
        
        # Update gate: how much to update hidden state
        z = sigmoid(self.W_z @ concat + self.b_z)
        
        # Reset gate: how much past information to use
        r = sigmoid(self.W_r @ concat + self.b_r)
        
        # Candidate hidden state using reset gate
        concat_reset = np.vstack((r * h_prev, x))
        h_candidate = np.tanh(self.W_h @ concat_reset + self.b_h)
        
        # Final hidden state: interpolate between previous and candidate
        h_next = (1 - z) * h_prev + z * h_candidate
        
        self.cache = (x, h_prev, z, r, h_candidate, concat, concat_reset)
        return h_next
    
    def backward(self, dh_next: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward pass for GRU cell.
        
        Args:
            dh_next: Gradient from next hidden state
            
        Returns:
            dx: Gradient w.r.t input
            dh_prev: Gradient w.r.t previous hidden state
        """
        x, h_prev, z, r, h_candidate, concat, concat_reset = self.cache
        
        # Gradient through final interpolation
        dh_prev1 = dh_next * (1 - z)  # Direct path through h_prev
        dz = dh_next * (h_candidate - h_prev)  # Through update gate
        
        dh_candidate = dh_next * z  # Through candidate
        
        # Gradient through candidate tanh
        dh_candidate_raw = dh_candidate * (1 - h_candidate ** 2)
        
        # Gradients for candidate weights
        self.dW_h = dh_candidate_raw @ concat_reset.T
        self.db_h = np.sum(dh_candidate_raw, axis=1, keepdims=True)
        
        # Gradient through reset concatenation
        dconcat_reset = self.W_h.T @ dh_candidate_raw
        
        # Split into reset-hidden and input parts
        dr_h_prev = dconcat_reset[:self.hidden_size]
        dx1 = dconcat_reset[self.hidden_size:]
        
        # Gradient through reset gate
        dr = dr_h_prev * h_prev
        dh_prev2 = dr_h_prev * r
        
        # Gradients for reset gate
        dr_raw = dr * r * (1 - r)
        self.dW_r = dr_raw @ concat.T
        self.db_r = np.sum(dr_raw, axis=1, keepdims=True)
        
        # Gradients for update gate  
        dz_raw = dz * z * (1 - z)
        self.dW_z = dz_raw @ concat.T
        self.db_z = np.sum(dz_raw, axis=1, keepdims=True)
        
        # Gradient through initial concatenation for update/reset gates
        dconcat_zr = self.W_z.T @ dz_raw + self.W_r.T @ dr_raw
        
        dh_prev3 = dconcat_zr[:self.hidden_size]
        dx2 = dconcat_zr[self.hidden_size:]
        
        # Combine all gradients
        dh_prev = dh_prev1 + dh_prev2 + dh_prev3
        dx = dx1 + dx2
        
        return dx, dh_prev


class VanillaRNNCell:
    """Vanilla RNN cell for comparison - suffers from vanishing gradients."""
    
    def __init__(self, input_size: int, hidden_size: int):
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        
        self.hidden_size = hidden_size
        self.cache = None
        
    def forward(self, x: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        h_next = np.tanh(self.W_hh @ h_prev + self.W_xh @ x + self.b_h)
        self.cache = (x, h_prev, h_next)
        return h_next
    
    def backward(self, dh_next: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x, h_prev, h_next = self.cache
        
        # Gradient through tanh - this is where vanishing gradients occur!
        dh_raw = dh_next * (1 - h_next ** 2)
        
        self.dW_hh = dh_raw @ h_prev.T
        self.dW_xh = dh_raw @ x.T
        self.db_h = np.sum(dh_raw, axis=1, keepdims=True)
        
        dx = self.W_xh.T @ dh_raw
        dh_prev = self.W_hh.T @ dh_raw
        
        return dx, dh_prev


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))


class AddingTask:
    """
    Adding Problem: A classic benchmark for testing sequence learning.
    
    The network must learn to add two specific numbers in a long sequence.
    This tests both short-term memory and the ability to ignore irrelevant information.
    """
    
    def __init__(self, seq_len: int = 50, batch_size: int = 32):
        self.seq_len = seq_len
        self.batch_size = batch_size
        
    def generate_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate batch for adding task.
        
        Returns:
            x: Input sequence of shape (seq_len, batch_size, 2)
            y: Target output of shape (batch_size, 1)
            markers: Positions of the numbers to add
        """
        x = np.random.rand(self.seq_len, self.batch_size, 2) * 0.5  # Random numbers in [0, 0.5]
        markers = np.zeros((self.seq_len, self.batch_size, 1))
        
        # Set markers for which numbers to add
        for b in range(self.batch_size):
            idx1, idx2 = np.random.choice(self.seq_len, 2, replace=False)
            markers[idx1, b, 0] = 1
            markers[idx2, b, 0] = 1
            
        # Combine numbers and markers
        x = np.concatenate([x, markers], axis=2)
        
        # Targets: sum of the two marked numbers
        y = np.zeros((self.batch_size, 1))
        for b in range(self.batch_size):
            marked_numbers = []
            for t in range(self.seq_len):
                if markers[t, b, 0] > 0.5:
                    marked_numbers.append(x[t, b, 0])
            y[b, 0] = sum(marked_numbers)
            
        return x, y, markers


def train_comparison():
    """
    Compare training stability of LSTM, GRU, and Vanilla RNN on adding task.
    
    This demonstrates why gated architectures are more stable for learning
    long-range dependencies compared to vanilla RNNs.
    """
    # Hyperparameters
    seq_len = 25
    hidden_size = 32
    input_size = 3  # two numbers + marker
    output_size = 1
    learning_rate = 0.01
    epochs = 500
    batch_size = 16
    
    # Initialize models
    lstm_cell = LSTMCell(input_size, hidden_size)
    gru_cell = GRUCell(input_size, hidden_size)
    rnn_cell = VanillaRNNCell(input_size, hidden_size)
    
    # Output layer (shared)
    W_out = np.random.randn(output_size, hidden_size) * 0.01
    b_out = np.zeros((output_size, 1))
    
    # Training history
    losses = {'LSTM': [], 'GRU': [], 'RNN': []}
    task = AddingTask(seq_len, batch_size)
    
    for epoch in range(epochs):
        # Generate batch
        x, y_true, _ = task.generate_batch()
        
        for model_name, cell in [('LSTM', lstm_cell), ('GRU', gru_cell), ('RNN', rnn_cell)]:
            # Forward pass through time
            if model_name == 'LSTM':
                h = np.zeros((hidden_size, batch_size))
                c = np.zeros((hidden_size, batch_size))
                for t in range(seq_len):
                    h, c = cell.forward(x[t].T, h, c)
            else:
                h = np.zeros((hidden_size, batch_size))
                for t in range(seq_len):
                    h = cell.forward(x[t].T, h)
            
            # Output layer
            y_pred = W_out @ h + b_out
            
            # Loss and gradient
            loss = np.mean((y_pred.T - y_true) ** 2)
            dy_pred = 2 * (y_pred.T - y_true).T / batch_size
            
            # Backward pass
            dW_out = dy_pred @ h.T
            db_out = np.sum(dy_pred, axis=1, keepdims=True)
            dh = W_out.T @ dy_pred
            
            # Backward through time
            if model_name == 'LSTM':
                dc = np.zeros_like(c)
                for t in reversed(range(seq_len)):
                    dx, dh, dc = cell.backward(dh, dc)
            else:
                for t in reversed(range(seq_len)):
                    dx, dh = cell.backward(dh)
            
            # Update weights (simple SGD)
            if model_name == 'LSTM':
                cell.W -= learning_rate * cell.dW
                cell.b -= learning_rate * cell.db
            elif model_name == 'GRU':
                cell.W_z -= learning_rate * cell.dW_z
                cell.W_r -= learning_rate * cell.dW_r
                cell.W_h -= learning_rate * cell.dW_h
                cell.b_z -= learning_rate * cell.db_z
                cell.b_r -= learning_rate * cell.db_r
                cell.b_h -= learning_rate * cell.db_h
            else:  # RNN
                cell.W_hh -= learning_rate * cell.dW_hh
                cell.W_xh -= learning_rate * cell.dW_xh
                cell.b_h -= learning_rate * cell.db_h
            
            W_out -= learning_rate * dW_out
            b_out -= learning_rate * db_out
            
            losses[model_name].append(loss)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: LSTM={losses['LSTM'][-1]:.4f}, "
                  f"GRU={losses['GRU'][-1]:.4f}, RNN={losses['RNN'][-1]:.4f}")
    
    return losses


def plot_results(losses: Dict[str, List[float]]):
    """Plot training curves to compare stability."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    for model_name, loss_history in losses.items():
        plt.plot(loss_history, label=model_name, alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Stability Comparison')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Plot last 100 epochs to see convergence details
    for model_name, loss_history in losses.items():
        plt.plot(loss_history[-100:], label=model_name, alpha=0.8)
    plt.xlabel('Epoch (last 100)')
    plt.ylabel('MSE Loss')
    plt.title('Final Convergence')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('exercise-19-training-comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def gradient_flow_analysis():
    """
    Analyze gradient flow through time to demonstrate why gates help.
    
    This shows how gradients propagate differently in each architecture,
    explaining why LSTM/GRU are better at learning long-range dependencies.
    """
    print("\n" + "="*60)
    print("GRADIENT FLOW ANALYSIS")
    print("="*60)
    
    # Simple demonstration of gradient magnitudes
    T = 10  # sequence length
    hidden_size = 8
    
    # Initialize cells
    lstm = LSTMCell(4, hidden_size)
    gru = GRUCell(4, hidden_size)
    rnn = VanillaRNNCell(4, hidden_size)
    
    # Forward pass (dummy data)
    x = np.random.randn(4, 1) * 0.1
    h_lstm = np.zeros((hidden_size, 1))
    c_lstm = np.zeros((hidden_size, 1))
    h_gru = np.zeros((hidden_size, 1))
    h_rnn = np.zeros((hidden_size, 1))
    
    lstm_states, gru_states, rnn_states = [], [], []
    
    for t in range(T):
        h_lstm, c_lstm = lstm.forward(x, h_lstm, c_lstm)
        h_gru = gru.forward(x, h_gru)
        h_rnn = rnn.forward(x, h_rnn)
        
        lstm_states.append(h_lstm.copy())
        gru_states.append(h_gru.copy())
        rnn_states.append(h_rnn.copy())
    
    # Backward pass - track gradient norms
    dh = np.ones_like(h_lstm)  # unit gradient from end
    
    lstm_grads, gru_grads, rnn_grads = [], [], []
    
    for t in reversed(range(T)):
        if t == T-1:
            dc = np.zeros_like(c_lstm)
            dx_lstm, dh_lstm, dc = lstm.backward(dh, dc)
            dx_gru, dh_gru = gru.backward(dh)
            dx_rnn, dh_rnn = rnn.backward(dh)
        else:
            dx_lstm, dh_lstm, dc = lstm.backward(dh_lstm, dc)
            dx_gru, dh_gru = gru.backward(dh_gru)
            dx_rnn, dh_rnn = rnn.backward(dh_rnn)
        
        lstm_grads.append(np.linalg.norm(dh_lstm))
        gru_grads.append(np.linalg.norm(dh_gru))
        rnn_grads.append(np.linalg.norm(dh_rnn))
    
    # Reverse to show propagation from start to end
    lstm_grads.reverse()
    gru_grads.reverse()
    rnn_grads.reverse()
    
    print("\nGradient norms through time (higher is better):")
    print("Time\tLSTM\t\tGRU\t\tRNN")
    for t in range(T):
        print(f"{t}\t{lstm_grads[t]:.6f}\t{gru_grads[t]:.6f}\t{rnn_grads[t]:.6f}")
    
    print(f"\nKey Insight: RNN gradients vanish much faster!")
    print(f"Final/Initial gradient ratios:")
    print(f"LSTM: {lstm_grads[-1]/lstm_grads[0]:.4f}")
    print(f"GRU:  {gru_grads[-1]/gru_grads[0]:.4f}") 
    print(f"RNN:  {rnn_grads[-1]/rnn_grads[0]:.4f}")


if __name__ == "__main__":
    print("Exercise 19: LSTM & GRU Internals")
    print("Core Concept: Gated architectures and vanishing gradients mitigation")
    print("=" * 70)
    
    # Run gradient analysis first to build intuition
    gradient_flow_analysis()
    
    print("\n" + "="*70)
    print("TRAINING COMPARISON ON ADDING TASK")
    print("="*70)
    print("This will take a moment...")
    
    # Train and compare all architectures
    losses = train_comparison()
    
    # Plot results
    plot_results(losses)
    
    print("\n" + "="*70)
    print("KEY TAKEAWAYS:")
    print("="*70)
    print("1. LSTM/GRU maintain much more stable gradient flow through time")
    print("2. Forget/update gates create adaptive paths for gradient propagation") 
    print("3. When gates ≈ 1, gradients can flow with minimal attenuation")
    print("4. Vanilla RNN suffers from exponential gradient decay (vanishing gradients)")
    print("5. This enables LSTM/GRU to learn longer-range dependencies effectively")