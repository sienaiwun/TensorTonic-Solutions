import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    x = np.asarray(x)   # 🔥 关键修复
    return 1.0 / (1.0 + np.exp(-x))
    # Write code here
    pass