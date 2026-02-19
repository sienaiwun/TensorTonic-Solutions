
import numpy as np

def linear_beta_schedule(T: int, beta_1: float = 0.0001, beta_T: float = 0.02) -> np.ndarray:
    """
    Linear noise schedule from beta_1 to beta_T.
    """
    return np.linspace(beta_1, beta_T, T, dtype=np.float64)

def cosine_alpha_bar_schedule(T: int, s: float = 0.008) -> np.ndarray:
    """
    Cosine schedule for alpha_bar (cumulative signal retention).
    Uses the function f(t) = cos^2(((t/T + s) / (1 + s)) * pi/2)
    """
    # Create an array of timesteps from 0 to T
    t = np.linspace(0, T, T + 1, dtype=np.float64)
    
    # Compute f(t) according to the cosine schedule formula
    f_t = np.cos(((t / T) + s) / (1 + s) * np.pi / 2.0) ** 2
    
    # alpha_bar_t is defined as f(t) / f(0)
    alpha_bars = f_t / f_t[0]
    
    # Return alpha_bars from t=1 to T (length T)
    return alpha_bars[1:]

def alpha_bar_to_betas(alpha_bars: np.ndarray, max_beta: float = 0.999) -> np.ndarray:
    """
    Convert alpha_bar schedule to beta schedule.
    Since alpha_bar_t = product_{i=1}^t (1 - beta_i), 
    we can find beta_t = 1 - (alpha_bar_t / alpha_bar_{t-1})
    """
    # Shift alpha_bars by one to get alpha_bar_{t-1}, prepending 1.0 for t=0
    alpha_bars_shifted = np.concatenate(([1.0], alpha_bars[:-1]))
    
    # Calculate betas
    betas = 1.0 - (alpha_bars / alpha_bars_shifted)
    
    # Clip betas to prevent singularities at the end of the diffusion process
    # (A common max_beta used in practice is 0.999)
    return np.clip(betas, 0.0, max_beta)