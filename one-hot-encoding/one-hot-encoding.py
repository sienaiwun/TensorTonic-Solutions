import numpy as np

def one_hot(y, num_classes=None):
    """
    Convert integer labels y ∈ {0,...,K-1} into one-hot matrix of shape (N, K).
    """
    if num_classes is None:
      num_classes = np.max(y) + 1

    # Write code here
    return  np.eye(num_classes, dtype=np.int32)[y]
    pass