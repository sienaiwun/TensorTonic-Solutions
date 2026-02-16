import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """

    out = np.empty((seq_length,d_model),float)
    for model_index in range(d_model):
        for pos in range(seq_length):
            if model_index%2 == 0:
                i = model_index /2
                denominator = 10000 ** (2 * i / d_model)
                out[pos,model_index] = np.sin(pos / denominator)
            else:
                i = (model_index -1) /2
                denominator = 10000 ** (2 * i / d_model)
                out[pos,model_index] = np.cos(pos / denominator)
    return out
    # Your code here
    pass