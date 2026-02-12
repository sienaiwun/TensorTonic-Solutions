import numpy as np
def maxpool_forward(X, pool_size, stride):
   

    X = np.array(X)
    H, W = X.shape  # height, width
    
    Hout = (H - pool_size) // stride + 1
    Wout = (W - pool_size) // stride + 1
    
    out = np.empty((Hout, Wout), dtype=X.dtype)
    
    for h in range(Hout):
        for w in range(Wout):
            h_start = h * stride
            w_start = w * stride
            window = X[h_start:h_start+pool_size, w_start:w_start+pool_size]
            out[h, w] = np.max(window)
    
    return out.tolist()
    """
    Compute the forward pass of 2D max pooling.
    """
    # Write code here