import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    dk = Q.size(-1)
    Kt = K.transpose(-2,-1)
    dot = torch.matmul(Q,Kt)
    return torch.matmul(F.softmax(dot/math.sqrt(dk), dim=-1) , V)
    pass