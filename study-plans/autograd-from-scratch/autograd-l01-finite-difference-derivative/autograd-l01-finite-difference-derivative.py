import numpy as np

def finite_difference_derivative(coefficients, x, h):
    """
    Returns: the polynomial value at x, the value at x plus h, and the forward-difference slope
    """
    p = coefficients[::-1]              # convert to descending for np.polyval
    fx = np.polyval(p, x)
    fx_p = np.polyval(p, x + h)
    return fx, fx_p, (fx_p - fx) / h