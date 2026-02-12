import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    x = np.array(x, dtype=float)
    if x.ndim == 3:
        C,H,W = x.shape
        
        output = np.empty((C),dtype = x.dtype)
        for c in range(C):
            window = x[c,:,:]
            x_view = window.reshape(1, H*W)
            output[c] = np.mean(x_view)
    elif  x.ndim == 4:
        N,C,H,W = x.shape
        output = np.empty((N,C),dtype = x.dtype)
        for n in range(N):
            for c in range(C):
                window = x[n,c,:,:]
                x_view = window.reshape(1, H*W)
                output[n, c] = np.mean(x_view)
    # Write code here
    else:
        raise ValueError("Input must be (C,H,W) or (N,C,H,W)")
    return output