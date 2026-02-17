import numpy as np

def reparameterize(mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
    """
    Sample from latent distribution using reparameterization trick.
    """
    epsilon = np.random.randn(*mu.shape)
    return mu + np.exp(log_var*0.5) * epsilon
    # Your implementation here
    pass