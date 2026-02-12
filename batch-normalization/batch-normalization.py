import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    npx = np.array(x)
    gamma = np.array(gamma)
    beta = np.array(beta)
    if npx.ndim == 2:
        pu = np.mean(x, axis=0) #2D features
        delta_squear = np.mean((x -pu)**2, axis = 0, keepdims=True)
        x_hat = (x - pu)/np.sqrt(delta_squear + eps)
        return gamma* x_hat + beta
    else:
        N,C,H,W = npx.shape
        pu = np.mean(x,axis = (0,2,3), keepdims=True) # 1 C 1 1 
        delta_squear = np.mean((x - pu)**2)
        x_hat = (x - pu)/np.sqrt(delta_squear + eps)
        return gamma.reshape(1,-1,1,1)* x_hat + beta.reshape(1,-1,1,1)
    # Write code here
    pass