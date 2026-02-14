import numpy as np

def one_hot(y, num_classes=None):
    """
    Convert integer labels y ∈ {0,...,K-1} into one-hot matrix of shape (N, K).
    """
    if num_classes is None:
      num_classes = np.max(y) + 1
    output = np.zeros((len(y),num_classes),np.int32)
    for i in range(len(y)):
      output[i][y[i]] = 1
    # Write code here
    return output
    pass