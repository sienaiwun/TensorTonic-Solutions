import numpy as np

def get_alpha_bar(betas: np.ndarray) -> np.ndarray:
    """
    Compute the cumulative product of (1 - beta).
    """
    return np.cumprod(1-betas)
    # YOUR CODE HERE
    pass

def forward_diffusion(
    x_0: np.ndarray,
    t: int,
    betas: np.ndarray
) -> tuple:
    alpha_bar = get_alpha_bar(betas)
    alpha_t = alpha_bar[t]
    noise = np.random.randn(*x_0.shape)
    return np.sqrt(alpha_t) * x_0 + np.sqrt(1 - alpha_t) * noise , noise
    """
    Sample x_t from q(x_t | x_0).
    """
    # YOUR CODE HERE
    pass
