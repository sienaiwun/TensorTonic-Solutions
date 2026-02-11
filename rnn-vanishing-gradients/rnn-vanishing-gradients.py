import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """
    Simulate gradient norm decay over T time steps.
    Returns list of gradient norms.
    """
    spectral_normal = np.linalg.norm(W_hh, ord=2)
    arr = []
    for i in range(T):
        arr.append(spectral_normal ** i)
    # YOUR CODE HERE
    return arr