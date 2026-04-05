import numpy as np

def dropout(x: np.ndarray, p: float = 0.5, training: bool = True) -> np.ndarray:
    """Apply dropout to input."""
    if not training:
        return x
    m = np.random.binomial(1, 1-p, x.shape)
    return x*m/(1-p)
    
    # YOUR CODE HERE
    pass