import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Callable
import seaborn as sns




plt.style.use('seaborn-v0_8')

np.random.seed(42)





def xavier_initialization(fan_in: int, fan_out: int) -> np.ndarray:

    scale = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.normal(0, scale, size=(fan_in, fan_out))



def he_initialization(fan_in: int, fan_out: int) -> np.ndarray:



    scale = np.sqrt(2.0 / fan_in)

    return np.random.normal(0, scale, size=(fan_in, fan_out))



def uniform_xavier(fan_in: int, fan_out: int) -> np.ndarray:



    limit =  np.sqrt(6.0 / (fan_in + fan_out))



    return np.random.uniform(-limit, limit, size=(fan_in, fan_out))




def simulate_forward_pass(layer_sizes: List[int], activation: str, initialization: str, 
                         n_samples: int = 1000) -> Dict[str, np.ndarray]:
    """Simulate forward signal propagation through network"""
    activations = {}
    current_activation = np.random.randn(n_samples, layer_sizes[0])
    activations['input'] = current_activation
    
    for i in range(1, len(layer_sizes)):
        fan_in, fan_out = layer_sizes[i-1], layer_sizes[i]
        
        # Choose initialization
        if initialization.lower() == 'xavier':
            W = xavier_initialization(fan_in, fan_out)
        elif initialization.lower() == 'he':
            W = he_initialization(fan_in, fan_out)
        else:
            W = np.random.randn(fan_in, fan_out) * 0.1  # Naive initialization
        
        # Linear transformation
        z = current_activation @ W
        
        # Apply activation
        if activation.lower() == 'relu':
            current_activation = np.maximum(0, z)
        elif activation.lower() == 'tanh':
            current_activation = np.tanh(z)
        elif activation.lower() == 'sigmoid':
            current_activation = 1 / (1 + np.exp(-z))
        else:
            current_activation = z  # Linear
        
        activations[f'layer_{i}'] = current_activation
    
    return activations

def calculate_layer_statistics(activations: Dict[str, np.ndarray]) -> Dict[str, Tuple[float, float]]:
    """Calculate mean and variance for each layer's activations"""
    stats = {}
    for layer_name, activation in activations.items():
        stats[layer_name] = (np.mean(activation), np.var(activation))
    return stats 