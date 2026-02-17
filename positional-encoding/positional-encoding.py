import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    out = np.empty((seq_len,d_model),float)
    for model_index in range(d_model):
        for pos in range(seq_len):
            if model_index%2 == 0:
                i = model_index /2
                denominator = base ** (2 * i / d_model)
                out[pos,model_index] = np.sin(pos / denominator)
            else:
                i = (model_index -1) /2
                denominator = base ** (2 * i / d_model)
                out[pos,model_index] = np.cos(pos / denominator)
    return out
    # Your code here
    # Write code here
    pass