import numpy as np

class Word2VecHelper:
    def __init__(self,corpus: np.ndarray):
        self.tokens : dict[str, int]= {}
        unigram_frequency : dict[str, int] = {}
        token_id = 0
        for sentence in corpus:
            for word in sentence.split():
                if word not in self.tokens:
                    unigram_frequency[word] = 1
                    self.tokens.update({word : token_id + 1})
                    token_id += 1
                else:
                    unigram_frequency[word] += 1

        vocab_size = len(self.tokens)
        frequency_arr = np.zeros(vocab_size)

        # "map" the word ids to their frequencies
        for word, idx in self.tokens.items():
            frequency_arr[idx-1] = unigram_frequency[word]
        smoothed_frequency_arr = frequency_arr ** 0.75
        self.unigram_distribution = smoothed_frequency_arr / np.sum(smoothed_frequency_arr)

    def getNegativeSamples(self, number_of_samples: int) -> np.ndarray:
        return np.random.choice(len(self.tokens), number_of_samples, p=self.unigram_distribution)
    def getTokens(self) -> dict[str, int]: return self.tokens
    def getId(self, target_word: str) -> int:
        if self.tokens.get(target_word) is None:
            return -1
        return self.tokens[target_word]