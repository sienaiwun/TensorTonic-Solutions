import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    x = np.asarray(x)
    pos_exp = np.exp(x)
    neg_exp = np.exp(-x)
    return (pos_exp - neg_exp) / (pos_exp + neg_exp)
    # Write code here
    pass