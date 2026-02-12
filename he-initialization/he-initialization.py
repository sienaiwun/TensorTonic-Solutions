import numpy as np
def he_initialization(W, fan_in):
    W = np.array(W, dtype=float)
    L = np.sqrt(6/fan_in)
    return W*2*L - L
    """
    Scale raw weights to He uniform initialization.
    """
    # Write code here