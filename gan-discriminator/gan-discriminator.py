import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    LeakyReLU activation function
    x: input numpy array
    alpha: negative slope coefficient
    """
    return np.where(x > 0, x, alpha * x)


def sigmoid(x):
    """
    Sigmoid activation function
    x: input numpy array
    """
    return 1 / (1 + np.exp(-x))

    
def discriminator(x: np.ndarray) -> np.ndarray:
    """
    Classify inputs as real or fake.
    """
    layer1_dim = 512
    layer2_dim = 256
    layer3_dim = 1
    W_1 = 0.01 * np.random.randn(x.shape[-1],layer1_dim).astype(x.dtype)
    B_1 = np.zeros(layer1_dim, dtype = x.dtype)

    W_2 = 0.01 * np.random.randn(layer1_dim,layer2_dim).astype(x.dtype)
    B_2 = np.zeros(layer2_dim, dtype = x.dtype)

    W_3 = 0.01 * np.random.randn(layer2_dim,layer3_dim).astype(x.dtype)
    B_3 = np.zeros(layer3_dim, dtype = x.dtype)

    x = x@W_1 + B_1
    x = leaky_relu(x)
    x = x@W_2 + B_2
    x = leaky_relu(x)
    x = x@W_3 + B_3
    x = leaky_relu(x)
    return sigmoid(x)

    
    
    # Your implementation here
    pass