import numpy as np
from typing import Tuple



def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1/ (1 + np.exp(-z))


def initialize_layer(input_dim: int, output_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    weights = np.random.randn(output_dim, input_dim) * 0.01
    biases = np.zeros((output_dim, 1))
    return weights, biases


def linear_forward(inputs: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> np.ndarray:
    return np.dot(weights, inputs) + biases


def layer_forward(inputs: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> np.ndarray:

    z = linear_forward(inputs, weights, biases)
    return sigmoid(z)


def initialize_network(layer_dims: Tuple[int, int, int, int]) -> dict:
    parameters = {
        'W1': None, 'b1': None,
        'W2': None, 'b2': None,
        'W3': None, 'b3': None
    }
    parameters['W1'], parameters['b1'] = initialize_layer(layer_dims[0], layer_dims[1])
    parameters['W2'], parameters['b2'] = initialize_layer(layer_dims[1], layer_dims[2])
    parameters['W3'], parameters['b3'] = initialize_layer(layer_dims[2], layer_dims[3])
    
    return parameters


def forward_propagation(X: np.ndarray, parameters: dict) -> np.ndarray:

    A1 = layer_forward(X, parameters['W1'], parameters['b1'])

    A2 = layer_forward(A1, parameters['W2'], parameters['b2'])

    A3 = layer_forward(A2, parameters['W3'], parameters['b3'])

    return A3



if __name__ == "__main__":
    layer_dims = (2, 3, 3, 1)
    parameters = initialize_network(layer_dims)


    X = np.random.randn(2, 5)

    output = forward_propagation(X, parameters)
    print(f"Output shape: {output.shape}")
    