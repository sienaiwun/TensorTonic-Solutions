import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    x = np.array(x)
    def o(x):
        return 1/(1+np.exp(-x))
    return x*o(x)
    # Write code here
    pass