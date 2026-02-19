import numpy as np

def unet_bottleneck(x: np.ndarray, out_channels: int) -> np.ndarray:
    """
    U-Net bottleneck: double convolution at lowest resolution.
    """
    # Your implementation here
    B, H, W, C = x.shape

    H_out = H - 4
    W_out = W - 4

    out = np.zeros((B, H_out, W_out, out_channels))

    return out