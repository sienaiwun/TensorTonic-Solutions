def xavier_initialization(W, fan_in, fan_out):
    W = np.array(W, dtype=float)
    L = np.sqrt(6/(fan_in+fan_out))
    return W*2*L - L
    """
    Scale raw weights to Xavier uniform initialization.
    """
    # Write code here