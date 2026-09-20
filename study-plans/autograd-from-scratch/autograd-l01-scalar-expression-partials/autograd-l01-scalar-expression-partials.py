import numpy as np

def scalar_expression_partials(a, b, c, h):
    """
    Returns: the expression value and its three numerical partial derivatives
    """
     # perturb each variable by h
    d = a * b + c
    d_a = (a + h) * b + c
    d_b = a * (b + h) + c
    d_c = a * b + (c + h)

    return (
        float(d),
        float((d_a - d) / h),
        float((d_b - d) / h),
        float((d_c - d) / h),
    )
