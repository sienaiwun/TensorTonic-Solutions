import numpy as np

class VanillaRNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim

        # Xavier initialization
        self.W_xh = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / (2 * hidden_dim))
        self.W_hy = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_h = np.zeros(hidden_dim)
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray, h_0: np.ndarray = None) -> tuple:
        """
        Forward pass through entire sequence.
        Returns (y_seq, h_final).
        """

        B,T,D = X.shape
        H = self.hidden_dim
        O = self.W_hy.shape[0]
        if h_0 is None:
            h_prev = np.zeros((B,H),dtype = X.dtype)
        else:
            h_prev = h_0
        y_seq =  np.empty((B,T,O),dtype = X.dtype)
        for t in range(T):
            X_t = X[:,t,:] # B, D
            w_t =  X_t @ self.W_xh.T + h_prev @   self.W_hh .T + self.b_h  # B, H
            h_prev = np.tanh(w_t)
            y_t = h_prev  @ self.W_hy.T + self.b_y
            y_seq[:,t,:] = y_t
        return y_seq, h_prev
        