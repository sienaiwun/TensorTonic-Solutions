import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def lstm_cell(x_t: np.ndarray, h_prev: np.ndarray, C_prev: np.ndarray,
              W_f: np.ndarray, W_i: np.ndarray, W_c: np.ndarray, W_o: np.ndarray,
              b_f: np.ndarray, b_i: np.ndarray, b_c: np.ndarray, b_o: np.ndarray) -> tuple:
    """Complete LSTM cell forward pass."""
    # YOUR CODE HERE

    source = np.concatenate((h_prev,x_t), axis = 1)
    ft = sigmoid(source@W_f.T + b_f)
    it = sigmoid(source@W_i.T + b_i)
    Ctbar = np.tanh(source@W_c.T + b_c)
    Ct = ft*C_prev + it*Ctbar
    ot = sigmoid(source@W_o.T + b_o)
    ht = ot*np.tanh(Ct)
    return ht, ot