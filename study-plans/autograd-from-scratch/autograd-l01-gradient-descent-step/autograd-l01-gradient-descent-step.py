import numpy as np

def gradient_descent_step(values, gradients, learning_rate):
    """
    Returns: updated values and the predicted first-order objective change
    """

     # 1. 先转成 float64 的 NumPy 数组
    values = np.asarray(values, dtype=np.float64)
    gradients = np.asarray(gradients, dtype=np.float64)

    # 2. 再做向量运算：update = -η * g
    update = -learning_rate * gradients

    # 3. 更新参数：θ' = θ + update
    updated_values = values + update

    # 4. 一阶预测：ΔL_pred = g · update
    delta_pred = np.dot(gradients, update)

    return updated_values.tolist(), float(delta_pred)