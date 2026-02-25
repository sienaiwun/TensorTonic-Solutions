import numpy as np

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))

def _bce_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)))

class GAN:
    """
    Minimal end-to-end GAN container (Generator + Discriminator) using simple MLPs.
    - Generator: z -> x_hat
    - Discriminator: x -> score in [0,1]
    Implements: generate, discriminate, train_step with SGD.
    """
    def __init__(self, data_dim: int, noise_dim: int):
        self.data_dim = int(data_dim)
        self.noise_dim = int(noise_dim)

        # Small hidden sizes (can be tuned)
        self.g_hidden = 128
        self.d_hidden = 128

        # Learning rates
        self.lr_g = 1e-3
        self.lr_d = 1e-3

        rng = np.random.default_rng()

        # --- Initialize generator params (z -> h -> x_hat) ---
        # Xavier/Glorot init
        def xavier(in_dim, out_dim):
            limit = np.sqrt(6.0 / (in_dim + out_dim))
            return rng.uniform(-limit, limit, size=(in_dim, out_dim))

        self.G_W1 = xavier(self.noise_dim, self.g_hidden)
        self.G_b1 = np.zeros((1, self.g_hidden))
        self.G_W2 = xavier(self.g_hidden, self.data_dim)
        self.G_b2 = np.zeros((1, self.data_dim))

        # --- Initialize discriminator params (x -> h -> logit) ---
        self.D_W1 = xavier(self.data_dim, self.d_hidden)
        self.D_b1 = np.zeros((1, self.d_hidden))
        self.D_W2 = xavier(self.d_hidden, 1)
        self.D_b2 = np.zeros((1, 1))

    # ----- Generator forward -----
    def _G_forward(self, z: np.ndarray):
        h1 = np.tanh(z @ self.G_W1 + self.G_b1)          # (N, g_hidden)
        x_hat = np.tanh(h1 @ self.G_W2 + self.G_b2)      # (N, data_dim)
        cache = (z, h1, x_hat)
        return x_hat, cache

    # ----- Discriminator forward -----
    def _D_forward(self, x: np.ndarray):
        h1 = np.tanh(x @ self.D_W1 + self.D_b1)          # (N, d_hidden)
        logit = h1 @ self.D_W2 + self.D_b2               # (N, 1)
        prob = _sigmoid(logit)                           # (N, 1)
        cache = (x, h1, logit, prob)
        return prob, cache

    def generate(self, n_samples: int) -> np.ndarray:
        """Generate fake samples: returns shape (n_samples, data_dim)."""
        n = int(n_samples)
        z = np.random.randn(n, self.noise_dim).astype(np.float64)
        x_hat, _ = self._G_forward(z)
        return x_hat

    def discriminate(self, x: np.ndarray) -> np.ndarray:
        """Classify samples as real/fake: returns probabilities shape (N, 1)."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.data_dim:
            raise ValueError(f"x must have shape (N, {self.data_dim})")
        prob, _ = self._D_forward(x)
        return prob

    def train_step(self, real_data: np.ndarray) -> dict:
        """
        Perform one GAN training iteration:
        1) Update Discriminator on real + fake (detach fake)
        2) Update Generator to fool Discriminator
        Returns dict with losses for monitoring.
        """
        real = np.asarray(real_data, dtype=np.float64)
        if real.ndim != 2 or real.shape[1] != self.data_dim:
            raise ValueError(f"real_data must have shape (N, {self.data_dim})")

        N = real.shape[0]

        # -----------------------
        # (A) Discriminator step
        # -----------------------
        z = np.random.randn(N, self.noise_dim).astype(np.float64)
        fake, _ = self._G_forward(z)  # treat as constant for D update

        # Forward D on real and fake
        d_real, cache_real = self._D_forward(real)   # (N,1)
        d_fake, cache_fake = self._D_forward(fake)   # (N,1)

        y_real = np.ones((N, 1), dtype=np.float64)
        y_fake = np.zeros((N, 1), dtype=np.float64)

        d_loss_real = _bce_loss(y_real, d_real)
        d_loss_fake = _bce_loss(y_fake, d_fake)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        # Gradients for D (BCE with sigmoid): dL/dlogit = (p - y)/N
        # Real
        x_r, h_r, logit_r, p_r = cache_real
        dlogit_r = (p_r - y_real) / N  # (N,1)

        # Fake
        x_f, h_f, logit_f, p_f = cache_fake
        dlogit_f = (p_f - y_fake) / N  # (N,1)

        # Accumulate grads from real+fake
        # Layer 2
        dD_W2 = h_r.T @ dlogit_r + h_f.T @ dlogit_f        # (d_hidden,1)
        dD_b2 = np.sum(dlogit_r, axis=0, keepdims=True) + np.sum(dlogit_f, axis=0, keepdims=True)

        # Backprop to h1 through tanh
        dh_r = dlogit_r @ self.D_W2.T                      # (N,d_hidden)
        dh_f = dlogit_f @ self.D_W2.T
        dpre_r = dh_r * (1.0 - h_r**2)                     # tanh'
        dpre_f = dh_f * (1.0 - h_f**2)

        # Layer 1
        dD_W1 = x_r.T @ dpre_r + x_f.T @ dpre_f            # (data_dim, d_hidden)
        dD_b1 = np.sum(dpre_r, axis=0, keepdims=True) + np.sum(dpre_f, axis=0, keepdims=True)

        # SGD update D
        self.D_W1 -= self.lr_d * dD_W1
        self.D_b1 -= self.lr_d * dD_b1
        self.D_W2 -= self.lr_d * dD_W2
        self.D_b2 -= self.lr_d * dD_b2

        # -------------------
        # (B) Generator step
        # -------------------
        z = np.random.randn(N, self.noise_dim).astype(np.float64)
        fake, cache_g = self._G_forward(z)

        d_fake_new, cache_df = self._D_forward(fake)
        # Non-saturating generator loss: -log(D(G(z))) == BCE(ones, D(fake))
        y_g = np.ones((N, 1), dtype=np.float64)
        g_loss = _bce_loss(y_g, d_fake_new)

        # Backprop through D to get dL/dx_fake
        x_f2, h_f2, logit_f2, p_f2 = cache_df
        dlogit = (p_f2 - y_g) / N                           # (N,1)
        dh = dlogit @ self.D_W2.T                           # (N,d_hidden)
        dpre = dh * (1.0 - h_f2**2)                         # (N,d_hidden)
        dx = dpre @ self.D_W1.T                             # (N,data_dim)  gradient wrt D input (fake)

        # Backprop through G
        z0, h1, x_hat = cache_g
        # x_hat = tanh(a2)
        da2 = dx * (1.0 - x_hat**2)                         # (N,data_dim)
        dG_W2 = h1.T @ da2                                  # (g_hidden,data_dim)
        dG_b2 = np.sum(da2, axis=0, keepdims=True)

        dh1 = da2 @ self.G_W2.T                             # (N,g_hidden)
        da1 = dh1 * (1.0 - h1**2)                           # tanh'
        dG_W1 = z0.T @ da1                                  # (noise_dim,g_hidden)
        dG_b1 = np.sum(da1, axis=0, keepdims=True)

        # SGD update G
        self.G_W1 -= self.lr_g * dG_W1
        self.G_b1 -= self.lr_g * dG_b1
        self.G_W2 -= self.lr_g * dG_W2
        self.G_b2 -= self.lr_g * dG_b2

        return {"d_loss": float(d_loss), "g_loss": float(g_loss)}