import numpy as np

def compute_ddpm_loss(
    model_predict: callable,
    x_0: np.ndarray,
    betas: np.ndarray,
    T: int
) -> float:
    """
    Compute DDPM training loss for a batch of images.

    Args:
        model_predict: function (x_t, t) -> predicted noise
        x_0: clean images, shape (B, ...)
        betas: noise schedule, shape (T,)
        T: total diffusion steps

    Returns:
        Scalar MSE loss
    """

    B = x_0.shape[0]

    # 1. Precompute alphas
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)  # shape (T,)

    # 2. Sample random timestep for each sample in batch
    t = np.random.randint(0, T, size=B)

    # 3. Sample noise
    epsilon = np.random.randn(*x_0.shape)

    # 4. Get corresponding alpha_bar_t
    alpha_bar_t = alpha_bars[t]  # shape (B,)

    # Reshape for broadcasting
    while len(alpha_bar_t.shape) < len(x_0.shape):
        alpha_bar_t = alpha_bar_t[:, None]

    # 5. Forward diffusion (q sample)
    x_t = (
        np.sqrt(alpha_bar_t) * x_0 +
        np.sqrt(1.0 - alpha_bar_t) * epsilon
    )

    # 6. Predict noise
    epsilon_pred = model_predict(x_t, t)

    # 7. Compute MSE loss
    loss = np.mean((epsilon - epsilon_pred) ** 2)

    return loss