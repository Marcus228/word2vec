import numpy as np

class SimpleTokeniser:
    def tokeniseCorpus(self, corpus: np.ndarray) -> dict:
        tokens = {}
        counter = 0
        for sentence in corpus:
            for word in sentence.split():
                if word not in tokens:
                    tokens.update({word : counter + 1})
                    counter += 1
        return tokens