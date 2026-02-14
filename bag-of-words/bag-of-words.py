import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """

    out = np.zeros(len(vocab), dtype=int)
    word_to_index = {}
    for i in range(len(vocab)):
      word_to_index[vocab[i]] = i
    for token in tokens:
      if token in word_to_index:
        out[word_to_index[token]] += 1
    return out
  # Your code here
    pass