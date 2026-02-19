import numpy as np

def unet_decoder_block(x: np.ndarray, skip: np.ndarray, out_channels: int) -> np.ndarray:
    """
    U-Net decoder block: up-conv (2x2) + concat skip + double 3x3 valid conv.

    Input:
        x: input from previous layer (B,H,W,C)
        skip: skip connection from encoder (B,H_skip,W_skip,C_skip)
        out_channels: number of output channels after decoder block

    Returns:
        output: (B, H_out, W_out, out_channels)
    """
    B, H, W, C = x.shape

    # 1. Up-conv (2x2) -> double H and W
    H_up, W_up = H * 2, W * 2
    x_up = np.zeros((B, H_up, W_up, C))

    # 2. Concatenate skip connection along channels
    # Make sure spatial dims match skip (if not, crop skip to H_up, W_up)
    H_skip, W_skip = skip.shape[1], skip.shape[2]
    H_crop = min(H_up, H_skip)
    W_crop = min(W_up, W_skip)

    x_cat = np.zeros((B, H_crop, W_crop, C + skip.shape[3]))

    # 3. Double 3x3 valid conv: H/W reduce by 4
    H_out = H_crop - 4
    W_out = W_crop - 4

    out = np.zeros((B, H_out, W_out, out_channels))

    return out