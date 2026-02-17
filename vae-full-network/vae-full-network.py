import numpy as np

class VAE:
    def __init__(self, input_dim: int, latent_dim: int):
        """
        Initialize VAE.
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = 32

        # Encoder weights
        self.W_l = 0.01 * np.random.randn(input_dim, self.hidden_dim)
        self.B_l = np.zeros(self.hidden_dim)

        self.W_mean = 0.01 * np.random.randn(self.hidden_dim, latent_dim)
        self.B_mean = np.zeros(latent_dim)

        self.W_var = 0.01 * np.random.randn(self.hidden_dim, latent_dim)
        self.B_var = np.zeros(latent_dim)

        # Decoder weights
        self.W_1 = 0.01 * np.random.randn(latent_dim, self.hidden_dim)
        self.B_1 = np.zeros(self.hidden_dim)

        self.W_2 = 0.01 * np.random.randn(self.hidden_dim, input_dim)
        self.B_2 = np.zeros(input_dim)

    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def vae_encoder(self, x: np.ndarray) -> tuple:
        h = self.relu(x @ self.W_l + self.B_l)
        mean = h @ self.W_mean + self.B_mean
        log_var = h @ self.W_var + self.B_var
        return mean, log_var

    def reparameterize(self, mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
        epsilon = np.random.randn(*mu.shape)
        return mu + np.exp(log_var * 0.5) * epsilon

    def vae_decoder(self, z: np.ndarray) -> np.ndarray:
        h = self.relu(z @ self.W_1 + self.B_1)
        x_recon = self.relu(h @ self.W_2 + self.B_2)
        return x_recon

    def kl_divergence(self, mu: np.ndarray, log_var: np.ndarray) -> float:
        var = np.exp(log_var)
        element = 1 + log_var - mu ** 2 - var
        return -0.5 * np.sum(element)

    def vae_loss(self, x: np.ndarray, x_recon: np.ndarray,
                 mu: np.ndarray, log_var: np.ndarray) -> dict:
        recon_loss = np.sum((x - x_recon) ** 2)
        kl_loss = self.kl_divergence(mu, log_var)
        total_loss = recon_loss + kl_loss
        return {
            "total": total_loss,
            "recon": recon_loss,
            "kl": kl_loss
        }

    def forward(self, x: np.ndarray) -> tuple:
        """
        Full forward pass through VAE.
        Returns reconstructed x, mean, log_var
        """
        mu, log_var = self.vae_encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.vae_decoder(z)
        return x_recon, mu, log_var

    def generate(self, n_samples: int) -> np.ndarray:
        """
        Generate new samples from standard normal prior.
        """
        z = np.random.randn(n_samples, self.latent_dim)
        x_gen = self.vae_decoder(z)
        return x_gen