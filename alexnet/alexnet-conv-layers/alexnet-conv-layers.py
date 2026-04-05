import numpy as np

def alexnet_conv1(image: np.ndarray) -> np.ndarray:
    """AlexNet first conv layer: 11x11, stride 4, 96 filters (shape simulation)."""
    # YOUR CODE HERE
    N, H, W, C = image.shape

    kernel = 11
    stride = 4

    H_out = int(np.floor((H - 7) / stride) + 1)
    W_out = int(np.floor((W - 7) / stride) + 1)

    return np.zeros((N, H_out, W_out, 96))