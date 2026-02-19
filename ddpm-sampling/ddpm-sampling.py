import numpy as np

def ddpm_sample(
    model_predict: callable,
    shape: tuple,
    betas: np.ndarray,
    T: int
) -> np.ndarray:
    """
    Generate a sample using DDPM reverse process.

    Args:
        model_predict: function (x_t, t) -> predicted noise
        shape: output shape (B, C, H, W)
        betas: noise schedule, shape (T,)
        T: total diffusion steps

    Returns:
        Generated sample x_0
    """

    # Precompute constants
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)

    # Start from pure noise
    x_t = np.random.randn(*shape)

    B = shape[0]

    for t in reversed(range(T)):

        t_batch = np.full((B,), t)

        alpha_t = alphas[t]
        beta_t = betas[t]
        alpha_bar_t = alpha_bars[t]

        # Predict noise
        epsilon_theta = model_predict(x_t, t_batch)

        # Compute mean
        coef1 = 1.0 / np.sqrt(alpha_t)
        coef2 = beta_t / np.sqrt(1.0 - alpha_bar_t)

        mean = coef1 * (x_t - coef2 * epsilon_theta)

        if t > 0:
            noise = np.random.randn(*shape)
            sigma = np.sqrt(beta_t)
            x_t = mean + sigma * noise
        else:
            x_t = mean  # last step, no noise

    return x_t