

def jaccard_similarity(set_a, set_b):
    set_a = set(set_a)
    set_b = set(set_b)
    a_and_b = set_a.intersection(set_b)
    a_or_b = set_a.union(set_b)
    if(len(a_or_b) == 0):
        return 0.0
    return len(a_and_b)/len(a_or_b)
    """
    Compute the Jaccard similarity between two item sets.
    """
    # Write code here