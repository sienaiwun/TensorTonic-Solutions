import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    nor = (x - mean) / np.sqrt(var + eps)
    return nor * gamma + beta
    # Your code here
    pass

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    B, N, D = Q.shape
    d_k = D // num_heads

    # 1. 必须使用投影后的矩阵 (qProj, kProj, vProj) 而不是原始的 Q, K, V
    q_proj = Q @ W_q  # (B, N, D)
    k_proj = K @ W_k
    v_proj = V @ W_v

    def split_heads(x):
        # 错误 1: reshape 不是 inplace 操作，必须重新赋值给 x
        x = x.reshape(B, N, num_heads, d_k) 
        # 错误 2: 换轴后的形状是 (B, num_heads, N, d_k)
        return x.transpose(0, 2, 1, 3)

    # 2. 调用 split_heads 时传入投影后的矩阵
    q_heads = split_heads(q_proj)
    k_heads = split_heads(k_proj)
    v_heads = split_heads(v_proj)

    # 3. 这里的变量名要和上面定义的对齐 (q_heads vs qHead)
    # 计算 QK^T / sqrt(d_k)
    scores = (q_heads @ k_heads.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    
    # 4. 注意力权重乘以 Value
    context_heads = weights @ v_heads  # (B, num_heads, N, d_k)
    
    # 5. 还原形状: 先 transpose 把 N 换到前面，再 reshape 拼接所有头
    # 错误 3: 这里的 batch_size, seq_len 等变量名要对应 B, N, D
    context = context_heads.transpose(0, 2, 1, 3).reshape(B, N, D)
    
    # 6. 最后投影
    output = context @ W_o
    
    return output
    # Your code here
    pass

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    x = x @ W1 + b1
    x = np.maximum(0, x)
    x = x @ W2 + b2
    return x
    # Your code here
    pass

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    atten =  multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    x = layer_norm(x + atten, gamma1, beta1)
    ff_out = feed_forward(x, W1, b1, W2, b2)

    # 4️⃣ Residual + LayerNorm
    out = layer_norm(x + ff_out, gamma2, beta2)

    # Your code here
    return out