import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
  
    # 1. Number of samples
    n = len(y_true)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    probs = y_pred[np.arange(n), y_true]
    ce =  -np.log(probs)
    
    # 3. Compute the negative log of those probabilities
    
    # 4. Return the mean of these losses
    return np.mean(ce, axis = 0)