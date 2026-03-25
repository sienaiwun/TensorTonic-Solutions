import numpy as np

def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    """
    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float)
    
    # Compute norms
    norm_v = np.linalg.norm(v)
    norm_w = np.linalg.norm(w)
    
    # Handle zero vectors
    if norm_v == 0 or norm_w == 0:
        return np.nan
    
    # Dot product
    dot = np.dot(v, w)
    
    # Cosine of angle
    cos_theta = dot / (norm_v * norm_w)
    
    # Clamp to [-1, 1] to avoid numerical issues
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Angle in radians
    theta = np.arccos(cos_theta)
    
    return theta