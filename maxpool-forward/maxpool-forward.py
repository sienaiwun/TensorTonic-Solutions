import numpy as np
def maxpool_forward(X, pool_size, stride):
    X = np.array(X)
    H,W  = X.shape
    
    Wout = int(np.floor((W - pool_size) / stride)) + 1
    Hout = int(np.floor((H - pool_size) / stride)) + 1
    out = np.empty((Hout,Wout),dtype = X.dtype)
    for w in range(Wout):
        for h in range(Hout):
            h_start = h * stride
            w_start = w * stride
            out[h,w] = np.max(X[h_start:h_start+pool_size,w_start:w_start+pool_size])
    return out.tolist()
