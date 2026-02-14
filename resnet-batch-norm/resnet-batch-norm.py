import numpy as np

class BatchNorm:
    """Batch Normalization layer."""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Apply batch normalization.
        """
        if training:
          mean = np.mean(x, axis = 0)
          var = np.var(x,axis = 0)
          self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
          self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
          mean = self.running_mean
          var = self.running_var
        x_bar = (x- mean)/np.sqrt(var + self.eps) * self.gamma + self.beta
        # YOUR CODE HERE
        return x_bar
      
def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)

def post_activation_block(x: np.ndarray, W1: np.ndarray, W2: np.ndarray, bn1: BatchNorm, bn2: BatchNorm) -> np.ndarray:
    """
    Post-activation ResNet block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    Uses x @ W for "convolution" (simplified as linear transform).
    """
    out = x @W1
    out = bn1.forward(out)
    out = relu(out)
    out = out@ W2
    out = bn2.forward(out)
    out = out + x
    out = relu(out)
    return out
    

def pre_activation_block(x: np.ndarray, W1: np.ndarray, W2: np.ndarray, bn1: BatchNorm, bn2: BatchNorm) -> np.ndarray:
    """
    Pre-activation ResNet block: BN -> ReLU -> Conv -> BN -> ReLU -> Conv
    This ordering often works better for very deep networks.
    """
    out = bn1.forward(x)
    out = relu(out)
    out = out @ W1
    out = bn2.forward(out)
    out = relu(out)
    out = out@ W2
    out = out + x
    return out

    out = relu(bn1.forward(x)) @ W1
    # 第二层：BN -> ReLU -> Conv
    out = relu(bn2.forward(out)) @ W2
    # 直接相加，不经过最后的 ReLU (ResNet-v2 的标志，保持 Identity 路径畅通)
    return out + x
    # YOUR CODE HERE
    pass
