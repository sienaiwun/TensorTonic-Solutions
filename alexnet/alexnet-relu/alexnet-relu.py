import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation: f(x) = max(0, x)"""
    # YOUR CODE HERE
    return np.maximum(x,np.zeros(x.shape[0]))
    pass