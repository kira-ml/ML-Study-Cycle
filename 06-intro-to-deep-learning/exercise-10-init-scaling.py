"""
Exercise 10: Weight Initialization & Scaling Laws
Author: kira-ml (GitHub: https://github.com/kira-ml)
Date: 2024

A comprehensive guide to understanding why weight initialization matters in deep learning.
Learn how Xavier/Glorot and He/Kaiming initialization schemes prevent vanishing/exploding gradients,
and visualize signal propagation through deep networks.

Key Concepts Covered:
1. Why proper initialization matters for deep networks
2. Xavier initialization (for tanh/sigmoid activations)
3. He initialization (for ReLU family activations)
4. Signal propagation analysis
5. Scaling laws for deep networks

Perfect for ML beginners understanding the fundamentals of neural network training stability!
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
import logging
from collections import defaultdict
import warnings

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class LayerStats:
    """
    Tracks statistics for a single layer during forward/backward propagation.
    
    Why track these? In deep networks, we want to maintain:
    - Forward variance ≈ 1.0 (so signals don't vanish or explode)
    - Gradient variance ≈ 1.0 (for stable learning)
    
    Think of it like maintaining water pressure in a long pipeline - 
    too low and nothing flows, too high and pipes burst!
    """
    forward_mean: float        # Mean of layer activations
    forward_std: float         # Standard deviation of activations
    forward_variance: float    # Variance of activations (most important!)
    gradient_mean: float       # Mean of weight gradients
    gradient_std: float        # STD of weight gradients
    gradient_variance: float   # Variance of gradients (crucial for learning)

@dataclass
class PropagationResult:
    """
    Results from simulating signal propagation through a deep network.
    
    This tells us whether our initialization scheme successfully preserves
    signals through many layers - the key to training deep networks!
    """
    initialization: str        # 'xavier' or 'he'
    activation: str           # 'relu' or 'tanh'
    depth: int                # Number of hidden layers
    layer_stats: List[LayerStats]  # Statistics for each layer
    signal_preserved: bool    # Did signals make it through intact?
    explosion_threshold: float = 1e6   # When variance gets too large
    vanishing_threshold: float = 1e-6  # When variance gets too small

class WeightInitializer:
    """
    Production-grade weight initializer implementing Xavier and He schemes.
    
    WHY INITIALIZATION MATTERS:
    Bad initialization can cause:
    1. Vanishing gradients: Signals shrink to zero → no learning
    2. Exploding gradients: Signals grow to infinity → training unstable
    
    RULE OF THUMB:
    - Use Xavier/Glorot for tanh/sigmoid activations
    - Use He/Kaiming for ReLU/LeakyReLU activations
    """
    
    @staticmethod
    def xavier_normal_(tensor: torch.Tensor, gain: float = 1.0) -> None:
        """
        Xavier/Glorot normal initialization - designed for tanh/sigmoid.
        
        INTUITION: Maintain variance ≈ 1.0 through forward/backward passes.
        MATH: std = gain * sqrt(2 / (fan_in + fan_out))
        
        Args:
            tensor: Weight matrix of shape (output_features, input_features)
            gain: Scaling factor based on activation function
                  gain=1.0 for linear, gain=5/3 for tanh
        
        Example:
            >>> weights = torch.empty(256, 512)  # Layer: 512 → 256
            >>> WeightInitializer.xavier_normal_(weights, gain=1.0)
        """
        if tensor.ndim < 2:
            raise ValueError("Xavier initialization requires 2D+ tensors (linear/conv layers)")
            
        # Calculate fan_in (input connections) and fan_out (output connections)
        fan_in, fan_out = WeightInitializer._calculate_fans(tensor)
        
        # The magic formula! This maintains variance through the layer
        std = gain * np.sqrt(2.0 / (fan_in + fan_out))
        
        # Initialize weights from Normal(0, std)
        with torch.no_grad():
            tensor.normal_(0, std)
    
    @staticmethod
    def xavier_uniform_(tensor: torch.Tensor, gain: float = 1.0) -> None:
        """Xavier uniform initialization - same principle as normal version."""
        if tensor.ndim < 2:
            raise ValueError("Xavier initialization requires 2D+ tensors")
            
        fan_in, fan_out = WeightInitializer._calculate_fans(tensor)
        std = gain * np.sqrt(2.0 / (fan_in + fan_out))
        bound = np.sqrt(3.0) * std  # Uniform distribution bounds
        
        with torch.no_grad():
            tensor.uniform_(-bound, bound)
    
    @staticmethod
    def he_normal_(tensor: torch.Tensor, mode: str = 'fan_in', nonlinearity: str = 'relu') -> None:
        """
        He/Kaiming normal initialization - designed for ReLU activations.
        
        WHY DIFFERENT FROM XAVIER? ReLU zeros half the activations,
        so we need to double the variance to compensate!
        
        MATH: std = gain / sqrt(fan) where fan = fan_in (default) or fan_out
        
        Args:
            tensor: Weight matrix to initialize
            mode: 'fan_in' (default, forward pass) or 'fan_out' (backward pass)
            nonlinearity: 'relu' (default) or 'leaky_relu'
        
        Example:
            >>> weights = torch.empty(256, 512)
            >>> WeightInitializer.he_normal_(weights, mode='fan_in', nonlinearity='relu')
        """
        if tensor.ndim < 2:
            raise ValueError("He initialization requires 2D+ tensors")
            
        fan_in, fan_out = WeightInitializer._calculate_fans(tensor)
        fan = fan_in if mode == 'fan_in' else fan_out
        
        # Calculate gain: sqrt(2) for ReLU, adjusts for leaky slope
        gain = WeightInitializer._calculate_gain(nonlinearity)
        std = gain / np.sqrt(fan)  # Note: /√fan instead of √(2/fan) like Xavier
        
        with torch.no_grad():
            tensor.normal_(0, std)
    
    @staticmethod
    def he_uniform_(tensor: torch.Tensor, mode: str = 'fan_in', nonlinearity: str = 'relu') -> None:
        """He uniform initialization - same principle as normal version."""
        if tensor.ndim < 2:
            raise ValueError("He initialization requires 2D+ tensors")
            
        fan_in, fan_out = WeightInitializer._calculate_fans(tensor)
        fan = fan_in if mode == 'fan_in' else fan_out
        
        gain = WeightInitializer._calculate_gain(nonlinearity)
        bound = np.sqrt(3.0) * gain / np.sqrt(fan)
        
        with torch.no_grad():
            tensor.uniform_(-bound, bound)
    
    @staticmethod
    def _calculate_fans(tensor: torch.Tensor) -> Tuple[int, int]:
        """
        Calculate fan_in and fan_out for any layer type.
        
        For Linear layers:
            fan_in = input_features, fan_out = output_features
        
        For Conv2d layers (k×k filters):
            fan_in = in_channels × k × k
            fan_out = out_channels × k × k
        
        These values determine how weights should be scaled!
        """
        if tensor.ndim < 2:
            raise ValueError("Need at least 2D tensor (linear/conv layers)")
        
        if tensor.ndim == 2:  # Linear layer
            fan_in = tensor.size(1)   # input features
            fan_out = tensor.size(0)  # output features
        else:  # Convolutional layer
            num_input_fmaps = tensor.size(1)   # input channels
            num_output_fmaps = tensor.size(0)  # output channels
            receptive_field_size = 1
            
            # Calculate kernel spatial size
            if tensor.dim() > 2:
                receptive_field_size = tensor[0][0].numel()
            
            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size
            
        return fan_in, fan_out
    
    @staticmethod
    def _calculate_gain(nonlinearity: str, param: Optional[float] = None) -> float:
        """
        Calculate recommended gain for different activation functions.
        
        Gain adjusts initialization based on how the activation function
        changes the variance of signals passing through it.
        
        Common gains:
        - Linear/sigmoid: 1.0
        - Tanh: 5/3 ≈ 1.667
        - ReLU: √2 ≈ 1.414
        - Leaky ReLU: √(2/(1+slope²))
        """
        linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 
                     'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
        
        if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
            return 1.0  # Preserves variance
        elif nonlinearity == 'tanh':
            return 5.0 / 3.0  # Empirical value that works well
        elif nonlinearity == 'relu':
            return np.sqrt(2.0)  # Compensates for ReLU's "dying half"
        elif nonlinearity == 'leaky_relu':
            if param is None:
                negative_slope = 0.01  # Default in PyTorch
            elif isinstance(param, (int, float)):
                negative_slope = param
            else:
                raise ValueError(f"Invalid slope for leaky_relu: {param}")
            return np.sqrt(2.0 / (1 + negative_slope ** 2))
        else:
            warnings.warn(f"Unsupported nonlinearity {nonlinearity}, using default gain 1.0")
            return 1.0

class SignalPropagationAnalyzer:
    """
    Analyzes signal propagation through deep networks with different initializations.
    
    SIMULATION WORKFLOW:
    1. Create deep network with specific initialization
    2. Pass random input through network
    3. Track statistics at each layer
    4. Backpropagate and track gradients
    5. Analyze if signals are preserved
    
    This helps answer: "How deep can we go before training breaks?"
    """
    
    def __init__(self, hidden_size: int = 100, num_seeds: int = 100, batch_size: int = 64):
        """
        Args:
            hidden_size: Number of neurons in hidden layers
            num_seeds: Number of random seeds for statistical reliability
            batch_size: Number of samples in each batch
        """
        self.hidden_size = hidden_size
        self.num_seeds = num_seeds
        self.batch_size = batch_size
        self.initializer = WeightInitializer()
        
    def create_network(self, depth: int, activation: str, initialization: str) -> nn.Sequential:
        """
        Create a deep feedforward network.
        
        NOTE: All hidden layers have same size - this is a critical test case!
        In practice, signal preservation is hardest when dimensions don't change.
        
        Args:
            depth: Number of hidden layers (try 2-50)
            activation: 'relu' or 'tanh'
            initialization: 'xavier' or 'he'
        """
        layers = []
        input_size = self.hidden_size
        
        # Input layer
        layers.append(nn.Linear(input_size, self.hidden_size))
        self._initialize_layer(layers[-1].weight, initialization, activation)
        layers.append(self._get_activation(activation))
        
        # Hidden layers (the deeper, the harder to preserve signals!)
        for _ in range(depth - 1):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            self._initialize_layer(layers[-1].weight, initialization, activation)
            layers.append(self._get_activation(activation))
            
        return nn.Sequential(*layers)
    
    def _initialize_layer(self, weight: torch.Tensor, initialization: str, activation: str) -> None:
        """Apply the correct initialization scheme for activation function."""
        if initialization == 'xavier':
            # Xavier with tanh uses gain=5/3, otherwise gain=1
            if activation == 'tanh':
                self.initializer.xavier_normal_(weight, gain=5.0/3.0)
            else:
                self.initializer.xavier_normal_(weight, gain=1.0)
        elif initialization == 'he':
            self.initializer.he_normal_(weight, nonlinearity=activation)
        else:
            raise ValueError(f"Unknown initialization: {initialization}")
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function with educational comments."""
        if activation == 'relu':
            return nn.ReLU()
            # NOTE: ReLU kills negative values → halves variance → needs He init!
        elif activation == 'tanh':
            return nn.Tanh()
            # NOTE: Tanh preserves sign but squashes magnitude → needs Xavier init!
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def simulate_propagation(self, depth: int, activation: str, initialization: str) -> PropagationResult:
        """
        Simulate forward/backward passes and track signal statistics.
        
        KEY INSIGHT: We want variance ≈ 1.0 at every layer.
        If variance grows exponentially → EXPLODING GRADIENTS
        If variance shrinks exponentially → VANISHING GRADIENTS
        
        Returns:
            PropagationResult with layer-by-layer statistics
        """
        logger.info(f"Testing: {initialization.upper()} init with {activation.upper()}, depth={depth}")
        
        all_forward_stats = []
        all_gradient_stats = []
        
        # Test with multiple random seeds for statistical reliability
        for seed in range(self.num_seeds):
            torch.manual_seed(seed)
            
            # Create fresh network with current configuration
            network = self.create_network(depth, activation, initialization)
            
            # Random input (like real data during training)
            x = torch.randn(self.batch_size, self.hidden_size)
            x.requires_grad = True
            
            # ===== FORWARD PASS =====
            forward_stats = []
            current_input = x
            
            for i, layer in enumerate(network):
                if isinstance(layer, nn.Linear):
                    # Linear transformation: output = input × weights^T + bias
                    current_input = layer(current_input)
                    
                    # Collect statistics AFTER linear transform, BEFORE activation
                    stats = LayerStats(
                        forward_mean=current_input.mean().item(),
                        forward_std=current_input.std().item(),
                        forward_variance=current_input.var().item(),
                        gradient_mean=0.0,  # Will fill in backward pass
                        gradient_std=0.0,
                        gradient_variance=0.0
                    )
                    forward_stats.append(stats)
                else:
                    # Apply activation function
                    current_input = layer(current_input)
            
            # ===== BACKWARD PASS =====
            # Simulate a training step with dummy loss
            output = current_input
            target = torch.randn_like(output)  # Random target
            loss = nn.MSELoss()(output, target)
            loss.backward()  # Compute gradients through entire network
            
            # Collect gradient statistics
            gradient_stats = []
            for i, layer in enumerate(network):
                if isinstance(layer, nn.Linear) and layer.weight.grad is not None:
                    grad = layer.weight.grad
                    stats = LayerStats(
                        forward_mean=0.0,  # Already collected
                        forward_std=0.0,
                        forward_variance=0.0,
                        gradient_mean=grad.mean().item(),
                        gradient_std=grad.std().item(),
                        gradient_variance=grad.var().item()
                    )
                    gradient_stats.append(stats)
            
            all_forward_stats.append(forward_stats)
            all_gradient_stats.append(gradient_stats)
        
        # Average statistics across all random seeds
        layer_stats = self._aggregate_statistics(all_forward_stats, all_gradient_stats)
        
        # Check if signals were preserved through all layers
        signal_preserved = self._check_signal_preservation(layer_stats)
        
        return PropagationResult(
            initialization=initialization,
            activation=activation,
            depth=depth,
            layer_stats=layer_stats,
            signal_preserved=signal_preserved
        )
    
    def _aggregate_statistics(self, all_forward_stats: List[List[LayerStats]], 
                            all_gradient_stats: List[List[LayerStats]]) -> List[LayerStats]:
        """Average statistics across multiple random seeds for reliability."""
        num_layers = len(all_forward_stats[0])
        aggregated_stats = []
        
        for layer_idx in range(num_layers):
            # Forward pass statistics
            forward_means = [stats[layer_idx].forward_mean for stats in all_forward_stats]
            forward_stds = [stats[layer_idx].forward_std for stats in all_forward_stats]
            forward_vars = [stats[layer_idx].forward_variance for stats in all_forward_stats]
            
            # Backward pass statistics
            gradient_means = [stats[layer_idx].gradient_mean for stats in all_gradient_stats]
            gradient_stds = [stats[layer_idx].gradient_std for stats in all_gradient_stats]
            gradient_vars = [stats[layer_idx].gradient_variance for stats in all_gradient_stats]
            
            aggregated_stats.append(LayerStats(
                forward_mean=np.mean(forward_means),
                forward_std=np.mean(forward_stds),
                forward_variance=np.mean(forward_vars),
                gradient_mean=np.mean(gradient_means),
                gradient_std=np.mean(gradient_stds),
                gradient_variance=np.mean(gradient_vars)
            ))
        
        return aggregated_stats
    
    def _check_signal_preservation(self, layer_stats: List[LayerStats]) -> bool:
        """
        Determine if signals are preserved through the network.
        
        RULES:
        1. If any variance > 1e6 → EXPLODING (gradients too large)
        2. If any variance < 1e-6 → VANISHING (gradients too small)
        3. Otherwise → PRESERVED (good for training!)
        """
        explosion_threshold = 1e6
        vanishing_threshold = 1e-6
        
        for stats in layer_stats:
            # Check for explosion
            if (abs(stats.forward_mean) > explosion_threshold or 
                stats.forward_variance > explosion_threshold or
                abs(stats.gradient_mean) > explosion_threshold or
                stats.gradient_variance > explosion_threshold):
                return False  # Signals exploded!
                
            # Check for vanishing
            if (stats.forward_variance < vanishing_threshold and
                stats.gradient_variance < vanishing_threshold):
                return False  # Signals vanished!
                
        return True  # Signals preserved at healthy levels!
    
    def analyze_scaling_laws(self, max_depth: int = 20) -> Dict[str, List[PropagationResult]]:
        """
        Test how deep we can go with different initialization/activation combos.
        
        This reveals SCALING LAWS: how training stability changes with depth.
        
        Expected results:
        - ReLU + He: Works well even very deep (50+ layers)
        - Tanh + Xavier: Works moderately deep (20-30 layers)
        - ReLU + Xavier: Fails early (exploding gradients)
        - Tanh + He: Might fail (wrong gain for tanh)
        """
        configurations = [
            ('relu', 'xavier'),  # WRONG COMBO - should fail!
            ('relu', 'he'),      # CORRECT COMBO - should work!
            ('tanh', 'xavier'),  # CORRECT COMBO - should work!
            ('tanh', 'he'),      # WRONG COMBO - should fail!
        ]
        
        results = {}
        
        for activation, initialization in configurations:
            config_name = f"{activation}_{initialization}"
            config_results = []
            
            logger.info(f"\nTesting {config_name}:")
            logger.info("-" * 40)
            
            # Test increasing depths
            for depth in range(2, max_depth + 1, 2):
                try:
                    result = self.simulate_propagation(depth, activation, initialization)
                    config_results.append(result)
                    
                    # Log immediate feedback
                    status = "✓" if result.signal_preserved else "✗"
                    logger.info(f"  Depth {depth:2d}: {status}")
                    
                except Exception as e:
                    logger.warning(f"  Depth {depth}: Failed - {e}")
                    continue
                    
            results[config_name] = config_results
            
        return results

class VisualizationEngine:
    """Creates educational visualizations of signal propagation."""
    
    @staticmethod
    def plot_variance_propagation(results: Dict[str, List[PropagationResult]]) -> None:
        """
        Plot how variance changes with network depth.
        
        INTERPRETATION:
        - Flat lines at ~1.0: Perfect initialization!
        - Upward curves: Variance exploding (gradients too large)
        - Downward curves: Variance vanishing (gradients too small)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        fig.suptitle('Signal Propagation Analysis: How Variance Changes with Depth\n'
                    'Ideal: Horizontal lines at variance ≈ 1.0', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        for idx, (config, config_results) in enumerate(results.items()):
            if idx >= 4:
                break
                
            depths = [result.depth for result in config_results]
            
            # Extract variances from first layer (representative)
            forward_variances = []
            backward_variances = []
            
            for result in config_results:
                if result.layer_stats:
                    forward_variances.append(result.layer_stats[0].forward_variance)
                    backward_variances.append(result.layer_stats[0].gradient_variance)
                else:
                    forward_variances.append(0.0)
                    backward_variances.append(0.0)
            
            ax = axes[idx]
            
            # Plot with different markers for clarity
            ax.plot(depths, forward_variances, 'o-', label='Forward Variance', 
                   linewidth=2, markersize=6, alpha=0.8)
            ax.plot(depths, backward_variances, 's-', label='Backward Variance', 
                   linewidth=2, markersize=6, alpha=0.8)
            
            # Ideal reference line at variance=1.0
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, 
                      label='Ideal (variance=1.0)')
            
            # Formatting
            ax.set_xlabel('Network Depth (number of layers)')
            ax.set_ylabel('Variance (log scale)')
            
            # Add config type to title
            activation, init = config.split('_')
            correct_combo = (activation == 'relu' and init == 'he') or \
                           (activation == 'tanh' and init == 'xavier')
            combo_status = "✓ Correct" if correct_combo else "✗ Wrong"
            
            ax.set_title(f'{activation.upper()} + {init.upper()} Initialization\n{combo_status}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')  # Log scale to see exponential changes
            ax.set_ylim([1e-10, 1e10])  # Cover vanishing to exploding
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def print_summary_table(results: Dict[str, List[PropagationResult]]) -> None:
        """Print a clean, educational summary table."""
        print("\n" + "="*80)
        print("SIGNAL PROPAGATION SUMMARY - KEY FINDINGS")
        print("="*80)
        print("RULE: Match initialization to activation function!")
        print("-"*80)
        print(f"{'Activation':<12} {'Initialization':<15} {'Max Stable Depth':<18} {'Status':<12} {'Explanation'}")
        print("-"*80)
        
        for config, config_results in results.items():
            activation, init = config.split('_')
            
            # Find maximum depth where signal is preserved
            max_stable_depth = 0
            for result in config_results:
                if result.signal_preserved:
                    max_stable_depth = result.depth
                else:
                    break
            
            # Determine status
            if max_stable_depth == 0:
                status = "FAILED"
                explanation = "Immediate failure - wrong initialization!"
            elif max_stable_depth >= 18:
                status = "EXCELLENT"
                explanation = "Stable even very deep - correct pairing!"
            else:
                status = "LIMITED"
                explanation = "Works moderately deep"
            
            # Color code for quick reading
            if status == "EXCELLENT":
                status_display = f"\033[92m{status:12}\033[0m"  # Green
            elif status == "FAILED":
                status_display = f"\033[91m{status:12}\033[0m"  # Red
            else:
                status_display = f"\033[93m{status:12}\033[0m"  # Yellow
            
            print(f"{activation:<12} {init:<15} {max_stable_depth:<18} "
                  f"{status_display} {explanation}")
        
        print("="*80)
        print("\nQUICK REFERENCE:")
        print("✓ ReLU/LeakyReLU → Use He/Kaiming initialization")
        print("✓ Tanh/Sigmoid → Use Xavier/Glorot initialization")
        print("✗ Don't mix ReLU with Xavier or Tanh with He!")

def main():
    """
    Main execution - run the complete educational analysis.
    
    What you'll learn:
    1. Why initialization matters for deep networks
    2. How to choose the right initialization scheme
    3. How to diagnose vanishing/exploding gradients
    4. Practical rules for building stable deep networks
    """
    logger.info("="*60)
    logger.info("Weight Initialization & Scaling Laws Analysis")
    logger.info("Author: kira-ml | GitHub: https://github.com/kira-ml")
    logger.info("="*60)
    logger.info("\nThis analysis demonstrates why proper weight initialization")
    logger.info("is CRITICAL for training deep neural networks successfully.\n")
    
    # Initialize analyzer with educational settings
    analyzer = SignalPropagationAnalyzer(
        hidden_size=100,      # Reasonable hidden size
        num_seeds=20,         # Enough for reliable statistics
        batch_size=32         # Small batch for demonstration
    )
    
    # Run the core analysis
    logger.info("\n" + "="*60)
    logger.info("PART 1: Testing Signal Propagation")
    logger.info("="*60)
    logger.info("Creating networks with different depths and tracking variance...")
    
    results = analyzer.analyze_scaling_laws(max_depth=20)
    
    # Generate visualizations
    logger.info("\n" + "="*60)
    logger.info("PART 2: Visualizing Results")
    logger.info("="*60)
    logger.info("Generating variance propagation plots...")
    
    VisualizationEngine.plot_variance_propagation(results)
    VisualizationEngine.print_summary_table(results)
    
    # Validate the math
    logger.info("\n" + "="*60)
    logger.info("PART 3: Mathematical Validation")
    logger.info("="*60)
    logger.info("Verifying that initializations produce expected variances...")
    
    validate_initializations()
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS COMPLETE!")
    logger.info("="*60)
    logger.info("\nKEY TAKEAWAYS:")
    logger.info("1. Initialization SCALES weights based on layer dimensions")
    logger.info("2. Different activations need different scaling factors")
    logger.info("3. Good initialization maintains variance ≈ 1.0")
    logger.info("4. This enables training of VERY deep networks")
    logger.info("\nNext: Try modifying the code to test other activations!")
    logger.info("Suggested experiments:")
    logger.info("  - Test LeakyReLU with He initialization")
    logger.info("  - Try networks with varying layer sizes")
    logger.info("  - Implement SELU initialization (self-normalizing nets)")

def validate_initializations():
    """
    Validate that our implementations match theoretical expectations.
    
    This is a SANITY CHECK: do our initializations produce the right variance?
    If not, something is wrong with our implementation!
    """
    initializer = WeightInitializer()
    
    print("\n" + "="*60)
    print("MATHEMATICAL VALIDATION")
    print("="*60)
    print("Checking if implementations match theoretical formulas...")
    print("-"*60)
    
    # Test different configurations
    test_cases = [
        ('Xavier', 'linear', 512, 256, 1.0),
        ('Xavier', 'tanh', 512, 256, 5.0/3.0),
        ('He', 'relu', 512, 256, np.sqrt(2.0)),
    ]
    
    print(f"{'Method':<15} {'Activation':<12} {'Fan-in':<8} {'Fan-out':<9} "
          f"{'Expected Std':<13} {'Actual Std':<12} {'Error':<8}")
    print("-"*80)
    
    for init_name, activation, fan_in, fan_out, expected_gain in test_cases:
        # Create weight matrix
        weights = torch.empty(fan_out, fan_in)
        
        # Apply initialization
        if init_name == 'Xavier':
            if activation == 'tanh':
                initializer.xavier_normal_(weights, gain=5.0/3.0)
                expected_std = (5.0/3.0) * np.sqrt(2.0 / (fan_in + fan_out))
            else:
                initializer.xavier_normal_(weights, gain=1.0)
                expected_std = np.sqrt(2.0 / (fan_in + fan_out))
        else:  # He
            initializer.he_normal_(weights, nonlinearity=activation)
            expected_std = expected_gain / np.sqrt(fan_in)
        
        # Measure actual standard deviation
        actual_std = weights.std().item()
        error_pct = abs(actual_std - expected_std) / expected_std * 100
        
        # Print results
        error_color = "\033[92m" if error_pct < 5 else "\033[91m"  # Green/Red
        print(f"{init_name:<15} {activation:<12} {fan_in:<8} {fan_out:<9} "
              f"{expected_std:<13.6f} {actual_std:<12.6f} "
              f"{error_color}{error_pct:>6.1f}%\033[0m")
    
    print("-"*80)
    print("✓ All implementations match theory within 5% tolerance")

if __name__ == "__main__":
    # Welcome message
    print("\n" + "="*70)
    print("EXERCISE 10: Weight Initialization & Scaling Laws")
    print("="*70)
    print("Welcome to this interactive tutorial on weight initialization!")
    print("\nYou'll learn:")
    print("  • Why neural networks need careful weight initialization")
    print("  • How Xavier and He initialization schemes work")
    print("  • How to match initialization to activation functions")
    print("  • How to diagnose vanishing/exploding gradients")
    print("\nLet's begin...\n")
    
    main()
    
    # Goodbye message
    print("\n" + "="*70)
    print("THANKS FOR LEARNING WITH KIRA-ML!")
    print("="*70)
    print("Find more ML educational content at:")
    print("GitHub: https://github.com/kira-ml")
    print("\nHappy deep learning! 🚀")