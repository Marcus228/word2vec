import numpy as np

class Word2VecHelper:
    def __init__(self,corpus_path: str):
        self.tokens : dict[str, int]= {}
        # read the corpus file
        with open(corpus_path, "r") as f:
            corpus_list = f.read().split()

        unique_words, counts = np.unique(corpus_list, return_counts=True)
        self.corpus = np.array(corpus_list)
        self.tokens = {word: idx for idx, word in enumerate(unique_words)}

        smoothed_frequency_arr = counts ** 0.75
        self.unigram_distribution = smoothed_frequency_arr / np.sum(smoothed_frequency_arr)

    def getNegativeSamples(self, number_of_samples: int) -> np.ndarray:
        return np.random.choice(len(self.tokens), number_of_samples, p=self.unigram_distribution)