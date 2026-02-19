import numpy as np

def unet_encoder_block_shape(x_shape, out_channels):
    B, H, W, C = x_shape
    H2 = H - 4
    W2 = W - 4
    skip_shape = (B, H2, W2, out_channels)
    pool_shape = (B, H2 // 2, W2 // 2, out_channels)
    return pool_shape, skip_shape

def unet_bottleneck_shape(x_shape, out_channels):
    B, H, W, C = x_shape
    H_out = H - 4
    W_out = W - 4
    return (B, H_out, W_out, out_channels)

def unet_decoder_block_shape(x_shape, skip_shape, out_channels):
    B, H, W, C = x_shape
    # upsample H/W *2
    H_up = H * 2
    W_up = W * 2
    # crop skip to match H_up/W_up
    H_crop = min(H_up, skip_shape[1])
    W_crop = min(W_up, skip_shape[2])
    # two 3x3 conv → H/W -4
    H_out = H_crop - 4
    W_out = W_crop - 4
    return (B, H_out, W_out, out_channels)

def unet_output_layer_shape(x_shape, num_classes):
    B, H, W, C = x_shape
    return (B, H, W, num_classes)

def unet(x: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """
    Complete U-Net forward pass (shape simulation).
    Input: x (B,H,W,C)
    Output: (B,H_out,W_out,num_classes)
    """
    B, H, W, C = x.shape

    # Encoder channel sizes
    enc_channels = [64, 128, 256, 512]

    # Encoder blocks
    skips = []
    x_shape = x.shape
    for out_ch in enc_channels:
        pool_shape, skip_shape = unet_encoder_block_shape(x_shape, out_ch)
        skips.append(skip_shape)
        x_shape = pool_shape

    # Bottleneck
    bottleneck_ch = 1024
    x_shape = unet_bottleneck_shape(x_shape, bottleneck_ch)

    # Decoder channel sizes (mirror of encoder)
    dec_channels = [512, 256, 128, 64]
    for i, out_ch in enumerate(dec_channels):
        skip_shape = skips[-(i+1)]
        x_shape = unet_decoder_block_shape(x_shape, skip_shape, out_ch)

    # Output layer
    out_shape = unet_output_layer_shape(x_shape, num_classes)

    # Return dummy array with correct shape
    return np.zeros(out_shape)