import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# --- Neural Net Functions ---
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))

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
    return A3, (A1, A2, A3)

# --- Visualization Helpers ---
def plot_sigmoid():
    z = np.linspace(-10, 10, 200)
    s = sigmoid(z)
    plt.figure(figsize=(6, 4))
    plt.plot(z, s, label="sigmoid(z)")
    plt.axhline(0.5, color="gray", linestyle="--", alpha=0.7)
    plt.title("Sigmoid Activation Function")
    plt.xlabel("z")
    plt.ylabel("sigmoid(z)")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_network_architecture(layer_dims):
    plt.figure(figsize=(8, 6))
    n_layers = len(layer_dims)
    layer_x = np.linspace(0, n_layers - 1, n_layers)

    for i, layer_size in enumerate(layer_dims):
        y_positions = np.linspace(0, 1, layer_size)
        plt.scatter([layer_x[i]] * layer_size, y_positions, s=500, label=f"Layer {i} ({layer_size} neurons)")
        if i > 0:
            prev_y_positions = np.linspace(0, 1, layer_dims[i-1])
            for y1 in prev_y_positions:
                for y2 in y_positions:
                    plt.plot([layer_x[i-1], layer_x[i]], [y1, y2], "k-", alpha=0.2)

    plt.title("Neural Network Architecture")
    plt.axis("off")
    plt.legend()
    plt.show()

def plot_forward_flow(X, activations):
    A1, A2, A3 = activations
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    matrices = {"Hidden Layer 1 Output": A1, "Hidden Layer 2 Output": A2, "Final Output": A3}
    
    for ax, (title, mat) in zip(axes, matrices.items()):
        im = ax.imshow(mat, cmap="viridis", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Samples")
        ax.set_ylabel("Neurons")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Forward Propagation Activations")
    plt.tight_layout()
    plt.show()

# --- Run Example ---
if __name__ == "__main__":
    layer_dims = (2, 3, 3, 1)  # input → hidden1 → hidden2 → output
    parameters = initialize_network(layer_dims)

    # Random input (2 features, 5 samples)
    X = np.random.randn(2, 5)
    output, activations = forward_propagation(X, parameters)
    print(f"Output shape: {output.shape}")

    # Visuals
    plot_sigmoid()
    plot_network_architecture(layer_dims)
    plot_forward_flow(X, activations)
