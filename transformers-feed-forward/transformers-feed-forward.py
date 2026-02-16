import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    """
    x = x @ W1 + b1
    x = np.maximum(0, x)
    x = x @ W2 + b2
    return x
    # Your code here
    pass