import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)
    
def vae_decoder(z: np.ndarray, output_dim: int) -> np.ndarray:
    """
    Decode latent vectors to reconstructed data.
    """
    B,input_dim = z.shape
    hiden_layer = 32

    W_1 = 0.01 * np.random.randn(input_dim,hiden_layer).astype(z.dtype)
    B_1= np.zeros(hiden_layer, dtype = z.dtype)

    W_2 = 0.01 * np.random.randn(hiden_layer,output_dim).astype(z.dtype)
    B_2 = np.zeros(output_dim, dtype = z.dtype)

    h = relu(z @ W_1 + B_1 )
    return relu(h@ W_2 + B_2)

    # Your implementation here
    pass