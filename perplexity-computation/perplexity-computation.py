import numpy as np
def perplexity(prob_distributions, actual_tokens):
    prob_distributions = np.array(prob_distributions)
    actual_tokens = np.array(actual_tokens)
    p = prob_distributions[np.arange(len(actual_tokens)), actual_tokens]
    H = -np.mean(np.log(p))
    return np.exp(H)
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    # Write code here