"""
Exercise 10: Weight Initialization & Scaling Laws
Implementation of Xavier/He initialization and signal propagation analysis.
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LayerStats:
    """Statistics for a single layer during propagation"""
    forward_mean: float
    forward_std: float
    forward_variance: float
    gradient_mean: float
    gradient_std: float
    gradient_variance: float

@dataclass
class PropagationResult:
    """Results from signal propagation simulation"""
    initialization: str
    activation: str
    depth: int
    layer_stats: List[LayerStats]
    signal_preserved: bool
    explosion_threshold: float = 1e6
    vanishing_threshold: float = 1e-6

class WeightInitializer:
    """
    Production-grade weight initializer implementing Xavier and He schemes.
    """
    
    @staticmethod
    def xavier_normal_(tensor: torch.Tensor, gain: float = 1.0) -> None:
        """
        Xavier/Glorot normal initialization.
        
        Args:
            tensor: Weight tensor to initialize
            gain: Scaling factor for the distribution
        """
        if tensor.ndim < 2:
            raise ValueError("Xavier initialization requires 2D+ tensors")
            
        fan_in, fan_out = WeightInitializer._calculate_fans(tensor)
        std = gain * np.sqrt(2.0 / (fan_in + fan_out))
        
        with torch.no_grad():
            tensor.normal_(0, std)
    
    @staticmethod
    def xavier_uniform_(tensor: torch.Tensor, gain: float = 1.0) -> None:
        """Xavier/Glorot uniform initialization"""
        if tensor.ndim < 2:
            raise ValueError("Xavier initialization requires 2D+ tensors")
            
        fan_in, fan_out = WeightInitializer._calculate_fans(tensor)
        std = gain * np.sqrt(2.0 / (fan_in + fan_out))
        bound = np.sqrt(3.0) * std
        
        with torch.no_grad():
            tensor.uniform_(-bound, bound)
    
    @staticmethod
    def he_normal_(tensor: torch.Tensor, mode: str = 'fan_in', nonlinearity: str = 'relu') -> None:
        """
        He/Kaiming normal initialization.
        
        Args:
            tensor: Weight tensor to initialize
            mode: 'fan_in' (default) or 'fan_out'
            nonlinearity: 'relu' (default) or 'leaky_relu'
        """
        if tensor.ndim < 2:
            raise ValueError("He initialization requires 2D+ tensors")
            
        fan_in, fan_out = WeightInitializer._calculate_fans(tensor)
        fan = fan_in if mode == 'fan_in' else fan_out
        
        gain = WeightInitializer._calculate_gain(nonlinearity)
        std = gain / np.sqrt(fan)
        
        with torch.no_grad():
            tensor.normal_(0, std)
    
    @staticmethod
    def he_uniform_(tensor: torch.Tensor, mode: str = 'fan_in', nonlinearity: str = 'relu') -> None:
        """He/Kaiming uniform initialization"""
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
        """Calculate fan_in and fan_out for a tensor"""
        if tensor.ndim < 2:
            raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
        
        if tensor.ndim == 2:  # Linear
            fan_in = tensor.size(1)
            fan_out = tensor.size(0)
        else:
            num_input_fmaps = tensor.size(1)
            num_output_fmaps = tensor.size(0)
            receptive_field_size = 1
            if tensor.dim() > 2:
                receptive_field_size = tensor[0][0].numel()
            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size
            
        return fan_in, fan_out
    
    @staticmethod
    def _calculate_gain(nonlinearity: str, param: Optional[float] = None) -> float:
        """
        Calculate gain for different nonlinearities.
        
        Args:
            nonlinearity: Name of nonlinearity function
            param: Optional parameter for leaky ReLU
            
        Returns:
            Recommended gain value
        """
        linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 
                     'conv_transpose2d', 'conv_transpose3d']
        
        if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
            return 1
        elif nonlinearity == 'tanh':
            return 5.0 / 3.0
        elif nonlinearity == 'relu':
            return np.sqrt(2.0)
        elif nonlinearity == 'leaky_relu':
            if param is None:
                negative_slope = 0.01
            elif not isinstance(param, bool) and isinstance(param, (int, float)):
                negative_slope = param
            else:
                raise ValueError("negative_slope {} not a valid number".format(param))
            return np.sqrt(2.0 / (1 + negative_slope ** 2))
        else:
            warnings.warn(f"Unsupported nonlinearity {nonlinearity}, using default gain 1.0")
            return 1.0

class SignalPropagationAnalyzer:
    """
    Analyzes signal propagation through deep networks with different initializations.
    """
    
    def __init__(self, hidden_size: int = 100, num_seeds: int = 100, batch_size: int = 64):
        self.hidden_size = hidden_size
        self.num_seeds = num_seeds
        self.batch_size = batch_size
        self.initializer = WeightInitializer()
        
    def create_network(self, depth: int, activation: str, initialization: str) -> nn.Sequential:
        """
        Create a deep network with specified configuration.
        
        Args:
            depth: Number of hidden layers
            activation: 'relu' or 'tanh'
            initialization: 'xavier' or 'he'
            
        Returns:
            Sequential neural network
        """
        layers = []
        input_size = self.hidden_size
        
        # Input layer
        layers.append(nn.Linear(input_size, self.hidden_size))
        self._initialize_layer(layers[-1].weight, initialization, activation)
        layers.append(self._get_activation(activation))
        
        # Hidden layers
        for _ in range(depth - 1):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            self._initialize_layer(layers[-1].weight, initialization, activation)
            layers.append(self._get_activation(activation))
            
        return nn.Sequential(*layers)
    
    def _initialize_layer(self, weight: torch.Tensor, initialization: str, activation: str) -> None:
        """Initialize a single layer's weights"""
        if initialization == 'xavier':
            if activation == 'tanh':
                self.initializer.xavier_normal_(weight, gain=5.0/3.0)
            else:
                self.initializer.xavier_normal_(weight, gain=1.0)
        elif initialization == 'he':
            self.initializer.he_normal_(weight, nonlinearity=activation)
        else:
            raise ValueError(f"Unsupported initialization: {initialization}")
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function module"""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def simulate_propagation(self, depth: int, activation: str, initialization: str) -> PropagationResult:
        """
        Simulate signal propagation through a deep network.
        
        Args:
            depth: Network depth
            activation: Activation function
            initialization: Weight initialization scheme
            
        Returns:
            Propagation results with statistics
        """
        logger.info(f"Simulating {initialization} init with {activation} activation, depth {depth}")
        
        all_forward_stats = []
        all_gradient_stats = []
        
        # Vectorized simulation across multiple seeds
        for seed in range(self.num_seeds):
            torch.manual_seed(seed)
            
            # Create network and input
            network = self.create_network(depth, activation, initialization)
            x = torch.randn(self.batch_size, self.hidden_size)
            x.requires_grad = True
            
            # Forward pass with statistics collection
            forward_stats = []
            current_input = x
            
            for i, layer in enumerate(network):
                if isinstance(layer, nn.Linear):
                    current_input = layer(current_input)
                    
                    # Collect forward statistics
                    stats = LayerStats(
                        forward_mean=current_input.mean().item(),
                        forward_std=current_input.std().item(),
                        forward_variance=current_input.var().item(),
                        gradient_mean=0.0,
                        gradient_std=0.0,
                        gradient_variance=0.0
                    )
                    forward_stats.append(stats)
                else:
                    current_input = layer(current_input)
            
            # Backward pass
            output = current_input
            target = torch.randn_like(output)
            loss = nn.MSELoss()(output, target)
            loss.backward()
            
            # Collect gradient statistics
            gradient_stats = []
            for i, layer in enumerate(network):
                if isinstance(layer, nn.Linear) and layer.weight.grad is not None:
                    grad = layer.weight.grad
                    stats = LayerStats(
                        forward_mean=0.0,
                        forward_std=0.0,
                        forward_variance=0.0,
                        gradient_mean=grad.mean().item(),
                        gradient_std=grad.std().item(),
                        gradient_variance=grad.var().item()
                    )
                    gradient_stats.append(stats)
            
            all_forward_stats.append(forward_stats)
            all_gradient_stats.append(gradient_stats)
        
        # Aggregate statistics across seeds
        layer_stats = self._aggregate_statistics(all_forward_stats, all_gradient_stats)
        
        # Check for signal preservation
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
        """Aggregate statistics across multiple random seeds"""
        num_layers = len(all_forward_stats[0])
        aggregated_stats = []
        
        for layer_idx in range(num_layers):
            # Aggregate forward stats
            forward_means = [stats[layer_idx].forward_mean for stats in all_forward_stats]
            forward_stds = [stats[layer_idx].forward_std for stats in all_forward_stats]
            forward_vars = [stats[layer_idx].forward_variance for stats in all_forward_stats]
            
            # Aggregate gradient stats
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
        """Check if signal is preserved through the network"""
        explosion_threshold = 1e6
        vanishing_threshold = 1e-6
        
        for stats in layer_stats:
            if (abs(stats.forward_mean) > explosion_threshold or 
                stats.forward_variance > explosion_threshold or
                abs(stats.gradient_mean) > explosion_threshold or
                stats.gradient_variance > explosion_threshold):
                return False  # Signal exploded
                
            if (stats.forward_variance < vanishing_threshold and
                stats.gradient_variance < vanishing_threshold):
                return False  # Signal vanished
                
        return True
    
    def analyze_scaling_laws(self, max_depth: int = 20) -> Dict[str, List[PropagationResult]]:
        """
        Analyze scaling laws for different configurations.
        
        Args:
            max_depth: Maximum network depth to test
            
        Returns:
            Dictionary of propagation results for each configuration
        """
        configurations = [
            ('relu', 'xavier'),
            ('relu', 'he'),
            ('tanh', 'xavier'),
            ('tanh', 'he')
        ]
        
        results = {}
        
        for activation, initialization in configurations:
            config_results = []
            for depth in range(2, max_depth + 1, 2):
                try:
                    result = self.simulate_propagation(depth, activation, initialization)
                    config_results.append(result)
                except Exception as e:
                    logger.warning(f"Failed for {activation}-{initialization} at depth {depth}: {e}")
                    continue
                    
            results[f"{activation}_{initialization}"] = config_results
            
        return results

class VisualizationEngine:
    """Handles visualization of signal propagation results."""
    
    @staticmethod
    def plot_variance_propagation(results: Dict[str, List[PropagationResult]]) -> None:
        """Plot forward and backward variance propagation"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, (config, config_results) in enumerate(results.items()):
            if idx >= 4:
                break
                
            depths = [result.depth for result in config_results]
            
            # Forward variance
            forward_variances = []
            for result in config_results:
                # Use variance from first layer as reference
                if result.layer_stats:
                    forward_variances.append(result.layer_stats[0].forward_variance)
                else:
                    forward_variances.append(0.0)
            
            # Backward variance
            backward_variances = []
            for result in config_results:
                if result.layer_stats:
                    backward_variances.append(result.layer_stats[0].gradient_variance)
                else:
                    backward_variances.append(0.0)
            
            ax = axes[idx]
            ax.plot(depths, forward_variances, 'o-', label='Forward Variance', linewidth=2)
            ax.plot(depths, backward_variances, 's-', label='Backward Variance', linewidth=2)
            ax.set_xlabel('Network Depth')
            ax.set_ylabel('Variance')
            ax.set_title(f'Signal Propagation: {config}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def print_summary_table(results: Dict[str, List[PropagationResult]]) -> None:
        """Print summary table of signal preservation"""
        print("\n" + "="*80)
        print("SIGNAL PROPAGATION SUMMARY")
        print("="*80)
        print(f"{'Configuration':<20} {'Max Stable Depth':<18} {'Signal Status':<15} {'Notes'}")
        print("-"*80)
        
        for config, config_results in results.items():
            max_stable_depth = 0
            signal_status = "PRESERVED"
            notes = []
            
            for result in config_results:
                if result.signal_preserved:
                    max_stable_depth = result.depth
                else:
                    if not notes:
                        # Analyze why signal failed
                        if any(stats.forward_variance > 1e6 for stats in result.layer_stats):
                            notes.append("Explosion")
                        elif any(stats.forward_variance < 1e-6 for stats in result.layer_stats):
                            notes.append("Vanishing")
                        else:
                            notes.append("Unstable")
                    break
            
            if max_stable_depth == 0:
                signal_status = "FAILED"
                notes_str = ", ".join(notes) if notes else "Immediate failure"
            else:
                notes_str = f"Stable up to {max_stable_depth} layers"
                
            print(f"{config:<20} {max_stable_depth:<18} {signal_status:<15} {notes_str}")
        
        print("="*80)

def main():
    """Main execution function"""
    logger.info("Starting Weight Initialization & Scaling Laws Analysis")
    
    # Initialize analyzer
    analyzer = SignalPropagationAnalyzer(
        hidden_size=100,
        num_seeds=50,  # Reduced for demonstration
        batch_size=64
    )
    
    # Run analysis
    logger.info("Running signal propagation analysis...")
    results = analyzer.analyze_scaling_laws(max_depth=20)
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    VisualizationEngine.plot_variance_propagation(results)
    VisualizationEngine.print_summary_table(results)
    
    # Validate initialization correctness
    logger.info("Validating initialization schemes...")
    validate_initializations()
    
    logger.info("Analysis complete!")

def validate_initializations():
    """Validate that initializations produce expected variances"""
    initializer = WeightInitializer()
    
    print("\n" + "="*50)
    print("INITIALIZATION VALIDATION")
    print("="*50)
    
    # Test configurations
    test_cases = [
        ('xavier', 'linear', 100, 100),
        ('xavier', 'tanh', 100, 100),
        ('he', 'relu', 100, 100),
    ]
    
    for init_type, activation, fan_in, fan_out in test_cases:
        weights = torch.empty(fan_out, fan_in)
        
        if init_type == 'xavier':
            if activation == 'tanh':
                initializer.xavier_normal_(weights, gain=5.0/3.0)
                expected_std = (5.0/3.0) * np.sqrt(2.0 / (fan_in + fan_out))
            else:
                initializer.xavier_normal_(weights, gain=1.0)
                expected_std = np.sqrt(2.0 / (fan_in + fan_out))
        else:  # he
            initializer.he_normal_(weights, nonlinearity=activation)
            gain = initializer._calculate_gain(activation)
            expected_std = gain / np.sqrt(fan_in)
        
        actual_std = weights.std().item()
        error_pct = abs(actual_std - expected_std) / expected_std * 100
        
        print(f"{init_type.upper()} ({activation}): Expected std={expected_std:.4f}, "
              f"Actual std={actual_std:.4f}, Error={error_pct:.1f}%")

if __name__ == "__main__":
    main()