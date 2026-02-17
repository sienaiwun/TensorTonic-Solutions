import numpy as np

def vae_encoder(x: np.ndarray, latent_dim: int) -> tuple:


    B,input_dim = x.shape
    hiden_layer = 32

    W_l = 0.01 * np.random.randn(input_dim,hiden_layer).astype(x.dtype)
    B_l = np.zeros(hiden_layer, dtype = x.dtype)

    W_mean = 0.01 * np.random.randn(hiden_layer,latent_dim).astype(x.dtype)
    B_mean = np.zeros(latent_dim, dtype = x.dtype)

    W_var = 0.01 * np.random.randn(hiden_layer,latent_dim).astype(x.dtype)
    B_var = np.zeros(latent_dim, dtype = x.dtype)

    h = x@W_l + B_l
    h = np.maximum(0, h) 
    mean = h@W_mean + B_mean
    var = h@ W_var + B_var
    
    return mean, var