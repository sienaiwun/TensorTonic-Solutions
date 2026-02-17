import numpy as np


import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    LeakyReLU activation function
    x: input numpy array
    alpha: negative slope coefficient
    """
    return np.where(x > 0, x, alpha * x)




    
def generator(z: np.ndarray, output_dim: int) -> np.ndarray:
    """
    Generate fake data from noise vectors.
    """
    layer1_dim = 256
    layer2_dim = 512
    x= z
    W_1 = 0.01 * np.random.randn(x.shape[-1],layer1_dim).astype(x.dtype)
    B_1 = np.zeros(layer1_dim, dtype = x.dtype)

    W_2 = 0.01 * np.random.randn(layer1_dim,layer2_dim).astype(x.dtype)
    B_2 = np.zeros(layer2_dim, dtype = x.dtype)

    W_3 = 0.01 * np.random.randn(layer2_dim,output_dim).astype(x.dtype)
    B_3 = np.zeros(output_dim, dtype = x.dtype)

    x = x@W_1 + B_1
    x = leaky_relu(x)
    x = x@W_2 + B_2
    x = leaky_relu(x)
    x = x@W_3 + B_3
    return np.tanh(x)
    # Your implementation here
    pass