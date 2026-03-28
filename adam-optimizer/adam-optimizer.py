import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    # Convert inputs to numpy arrays (handles scalars + arrays uniformly)
    param = np.asarray(param)
    grad = np.asarray(grad)
    m = np.asarray(m)
    v = np.asarray(v)

    # 1. Update biased first moment estimate
    m_new = beta1 * m + (1 - beta1) * grad

    # 2. Update biased second raw moment estimate
    v_new = beta2 * v + (1 - beta2) * (grad ** 2)

    # 3. Bias correction
    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)

    # 4. Parameter update
    param_new = param - lr * m_hat / (np.sqrt(v_hat) + eps)

    return param_new, m_new, v_new
  