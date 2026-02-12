import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class LSTM:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))

        self.W_f = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_i = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_c = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_o = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.b_f = np.zeros(hidden_dim)
        self.b_i = np.zeros(hidden_dim)
        self.b_c = np.zeros(hidden_dim)
        self.b_o = np.zeros(hidden_dim)

        self.W_y = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_y = np.zeros(output_dim)

    def lstm_cell(self, x_t: np.ndarray, h_prev: np.ndarray, C_prev: np.ndarray,
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
        return ht, Ct

    def forward(self, X: np.ndarray) -> tuple:
        """Forward pass. Returns (y, h_last, C_last)."""
        # YOUR CODE HERE

        N,T, D = X.shape
        ht = np.zeros((N, self.hidden_dim),dtype = X.dtype)
        Ct = np.zeros((N, self.hidden_dim),dtype = X.dtype)
        output = np.empty((N,T,self.b_y.shape[0]),dtype = X.dtype)
        for t in range(T):
            xt = X[:,t,:]
            ht, Ct = self.lstm_cell(xt,ht, Ct, 
            self.W_f, self.W_i, self.W_c,self.W_o,
            self.b_f, self.b_i, self.b_c,self.b_o)
            yt = ht @ self.W_y.T + self.b_y
            output[:,t,:] = yt
        return output,ht,Ct