import numpy as np

def detect_mode_collapse(generated_samples: np.ndarray, threshold: float = 0.1) -> dict:
    """
    Detect mode collapse in generated samples.

    Args:
        generated_samples: np.ndarray of shape (num_samples, feature_dim)
        threshold: float, minimum diversity threshold to avoid collapse

    Returns:
        dict with:
            - 'diversity_score': float, mean std of features across samples
            - 'is_collapsed': bool, True if diversity below threshold
    """
    # 计算每个特征维度上的标准差
    std_per_feature = np.std(generated_samples, axis=0)
    
    # 对标准差取平均，得到总体多样性评分
    diversity_score = np.mean(std_per_feature)
    
    # 判断是否低于阈值 → 模式崩溃
    is_collapsed = diversity_score < threshold
    
    return {
        "diversity_score": float(diversity_score),
        "is_collapsed": is_collapsed
    }