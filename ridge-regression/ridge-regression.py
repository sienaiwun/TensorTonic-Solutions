import numpy as np
def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    x = np.array(X)
    y = np.array(y)
    xt = np.transpose(x)
    return np.linalg.inv(xt @ x + lam * np.eye(x.shape[-1])) @  xt @  y
    # Write code here