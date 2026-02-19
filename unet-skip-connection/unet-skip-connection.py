import numpy as np

def crop_and_concat(encoder_features: np.ndarray, decoder_features: np.ndarray) -> np.ndarray:
    """
    Center crop encoder features to match decoder features and concatenate along channels.
    
    Args:
        encoder_features: (B, H_enc, W_enc, C_enc)
        decoder_features: (B, H_dec, W_dec, C_dec)
    
    Returns:
        output: (B, H_dec, W_dec, C_enc + C_dec)
    """
    B, H_enc, W_enc, C_enc = encoder_features.shape
    _, H_dec, W_dec, C_dec = decoder_features.shape

    # Compute top-left corner for center crop
    top = (H_enc - H_dec) // 2
    left = (W_enc - W_dec) // 2

    # Crop encoder features
    cropped_enc = encoder_features[:, top:top+H_dec, left:left+W_dec, :]

    # Concatenate along channels
    output = np.concatenate([cropped_enc, decoder_features], axis=-1)

    return output