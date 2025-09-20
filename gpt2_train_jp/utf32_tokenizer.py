import numpy as np

class Utf32Tokenizer:
    def __init__(self, text):
        self.build_vocab(text)        

    def build_vocab(self, text):
        b =text.encode('utf-32le')
        n = np.frombuffer(b, dtype='<u4')
        all_words = sorted(set(n))
        vocab = {token:integer for integer, token in enumerate(all_words)}
        for i, item in enumerate(vocab.items()):
            print(item)
            if i >= 10:
                break

        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
        self.n_vocab = len(vocab)
        return vocab

    def encode(self, text, allowed_special={"<|endoftext|>"}):
        b =text.encode('utf-32le')
        n = np.frombuffer(b, dtype='<u4')
        ids = [self.str_to_int[s] for s in n]
        return ids
        
    def decode(self, ids):
        s = [self.int_to_str[i] for i in ids]
        n = np.array(s, dtype=np.uint32)
        b = bytes(n)
        t = b.decode('utf-32le')
        return t
    

#tokenizer = Utf32Tokenizer()
#
#text = """"あ It's the last he painted, you know," 
#           Mrs. Gisburn said with pardonable pride."""
#print(text)
#ids = tokenizer.encode(text)
#print(ids)
#
#r = tokenizer.decode(ids)
#print(r)
