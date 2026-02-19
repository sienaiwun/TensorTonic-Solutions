import numpy as np

def unet_output(features: np.ndarray, num_classes: int) -> np.ndarray:
    """
    U-Net output layer: 1x1 conv for pixel-wise classification.
    """
    B, H, W, C_feat = features.shape
    out = np.zeros((B, H, W, num_classes))
    return out
    # Your implementation here
    pass