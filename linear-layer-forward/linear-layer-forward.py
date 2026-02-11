import numpy as np
def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    np_X = np.asarray(X)
    np_W = np.asarray(W)
    np_b = np.asarray(b)
    np_result = np_X@np_W + np_b
    return np_result.tolist()