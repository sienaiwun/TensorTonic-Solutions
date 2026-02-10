import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    B,T,D = X.shape
    H = b_h.shape[0]
    h_all = np.zeros((B, T, H))
    h = h_0
    for t in range(T):
      weight =   h@ W_hh+  X[:,t,:]@W_xh.T+ b_h  # (B, H)
      h = np.tanh(weight) # (B, H)
      h_all[:,t,:] = h
    return (h_all,h)