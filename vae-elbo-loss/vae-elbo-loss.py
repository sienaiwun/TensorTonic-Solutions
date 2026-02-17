import numpy as np


def kl_divergence(mu: np.ndarray, log_var: np.ndarray) -> float:
    """
    Compute KL divergence between q(z|x) and N(0, I).
    """
    var = np.exp(log_var)
    element = 1 + log_var - mu**2 - var
    return -0.5 * np.sum(element)


def vae_loss(x: np.ndarray, x_recon: np.ndarray, 
             mu: np.ndarray, log_var: np.ndarray) -> dict:
    """
    Compute VAE ELBO loss.
    """
    # Reconstruction loss (MSE)
    recon_loss = np.sum((x - x_recon) ** 2)

    # KL divergence
    kl_loss = kl_divergence(mu, log_var)

    # Total ELBO loss
    total_loss =  recon_loss + kl_loss
    return {
        "total": total_loss,
        "recon": recon_loss,
        "kl": kl_loss
    }