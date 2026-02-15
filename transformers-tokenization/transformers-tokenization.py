import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

        
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        i =0
        for special_token in [ self.pad_token,self.unk_token,self.bos_token,self.eos_token]:
            self.word_to_id[special_token] = i
            self.id_to_word[i] = special_token
            i+=1
        for text in texts:
            for word in text.split(): # 核心修正：分词后再存入
                if word not in self.word_to_id:
                    self.word_to_id[word] = i
                    self.id_to_word[i] = word
                    i += 1
        self.vocab_size = i
            
        
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        out = []
        for token in text.split():
            # 查不到就用 UNK (ID: 1)
            token_id = self.word_to_id.get(token, self.word_to_id[self.unk_token])
            out.append(token_id)
        return out
        
        # YOUR CODE HERE
        pass
    
    def decode(self, ids: List[int]) -> str:
        to_filter = {
            self.word_to_id[self.pad_token], 
            self.word_to_id[self.bos_token], 
            self.word_to_id[self.eos_token]
        }
        
        tokens = []
        for idx in ids:
            if idx in to_filter:
                continue
            # 保留 UNK 或者正常单词
            tokens.append(self.id_to_word.get(idx, self.unk_token))
            
        return " ".join(tokens)
        