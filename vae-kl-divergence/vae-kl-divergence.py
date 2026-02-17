import numpy as np

def kl_divergence(mu: np.ndarray, log_var: np.ndarray) -> float:
    """
    Compute KL divergence between q(z|x) and N(0, I).
    """
    var = np.exp(log_var)
    element = 1 + log_var - mu ** 2 - var
    return -np.sum(element) /2
    # Your implementation here
    pass