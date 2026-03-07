import numpy as np
import w2vHelperModule as helper_module
import BinaryClassifiers as binary_classifiers


class Word2Vec:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 100, embedding_dimension: int = 30, window_size: int = 5, number_of_negative_samples: int = 10, batch_size: int = 128, save_embeddings: bool = True):
        if (learning_rate <= 0) or (epochs <= 0) or (embedding_dimension <= 0) or (window_size <= 0) or (number_of_negative_samples <= 0):
            raise AttributeError(
                "Learning rate, epochs, dimension and window size must be positive"
            )
        if window_size > embedding_dimension:
            raise AttributeError(
                "Window size must be less than or equal to dimension"
            )
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._embedding_dimension = embedding_dimension
        self._window_size = window_size
        self._number_of_negative_samples = number_of_negative_samples
        self._batch_size = batch_size
        self._save_embeddings = save_embeddings
        self._helper_functions = None
        self._embeddings = None

    def train(self, corpus_path: str):
        # initialise the helper object
        self._helper_functions = helper_module.Word2VecHelper(corpus_path)
        token_map = self._helper_functions.tokens
        corpus = self._helper_functions.corpus

        # initialise the buffers, which will hold the word ids, context ids and labels
        target_buffer, context_buffer, label_buffer = [], [], []

        # initialise the word and context matrices
        np.random.seed(42)
        word_matrix = np.random.uniform(-0.5, 0.5, (len(token_map), self._embedding_dimension))
        context_matrix = np.random.uniform(-0.5, 0.5, (len(token_map), self._embedding_dimension))

        # now we train the model
        for epoch in range(self._epochs):
            # iterate over the corpus
            for target_index, target_word in enumerate(corpus):
                target_word_id = token_map[target_word]

                # get the positive and negative samples for this word
                positive_pairs_ids = self.__getPositivePairsIds(target_index, token_map, corpus)
                negative_pairs_ids = self.__getNegativeSamplesIds()

                # create the training data in the buffers
                target_buffer.append(np.repeat(target_word_id, positive_pairs_ids.size))
                context_buffer.append(positive_pairs_ids)
                label_buffer.append(np.ones(positive_pairs_ids.size, dtype=int))

                target_buffer.append(np.repeat(target_word_id, negative_pairs_ids.size))
                context_buffer.append(negative_pairs_ids)
                label_buffer.append(np.zeros(negative_pairs_ids.size, dtype=int))

                # check if the buffer is full, and if so, train the model on this batch
                if len(target_buffer) >= self._batch_size:
                    batch_target_ids = np.concatenate(target_buffer)
                    batch_context_ids = np.concatenate(context_buffer)
                    batch_label_ids = np.concatenate(label_buffer)
                    self.__train_step(batch_target_ids, batch_context_ids, batch_label_ids, word_matrix, context_matrix)
                    target_buffer, context_buffer, label_buffer = [], [], []

            # have to include the last batch if it is not empty
            if len(target_buffer) > 0:
                batch_target_ids = np.concatenate(target_buffer)
                batch_context_ids = np.concatenate(context_buffer)
                batch_label_ids = np.concatenate(label_buffer)
                self.__train_step(batch_target_ids, batch_context_ids, batch_label_ids, word_matrix, context_matrix)
                target_buffer, context_buffer, label_buffer = [], [], []
            print(f"Epoch {epoch} complete")
        print("Training complete")

        # at this point, the model is trained
        # now we can merge the word and context matrices to get a meaningful embedding matrix and optionally save it
        self._embeddings = np.add(word_matrix, context_matrix)
        if self._save_embeddings:
            np.save("word_embeddings.npy", self._embeddings)
            np.save("token_map.npy", token_map)

    def getEmbeddingOf(self, word : str) -> np.ndarray:
        token_map = self.__load_token_map()
        self.__load_embeddings()
        if self._embeddings is None or token_map is None:
            raise FileNotFoundError("No pre-computed embeddings found. Please train the model first.")
        if word not in token_map:
            raise ValueError(f"Word {word} not found in the vocabulary.")
        embedding_id = token_map[word]
        return self._embeddings[embedding_id]

    #private methods
    def __train_step(self, targets: np.ndarray, contexts: np.ndarray, labels: np.ndarray, word_matrix: np.ndarray, context_matrix: np.ndarray):
        word_batch = word_matrix[targets]
        context_batch = context_matrix[contexts]
        dots = np.sum(word_batch * context_batch, axis=1)
        sigmoid = binary_classifiers.sigmoid(dots)
        errors = (labels - sigmoid)[:, np.newaxis]
        target_gradients = self._learning_rate * errors * context_batch
        context_gradients = self._learning_rate * errors * word_batch
        np.add.at(word_matrix, targets, target_gradients)
        np.add.at(context_matrix, contexts, context_gradients)


    def __getPositivePairsIds(self, target_word_index : int, token_map: dict, corpus: np.ndarray) -> np.ndarray:
        lower_bound = max(0, target_word_index - self._window_size)
        upper_bound = min(len(corpus), target_word_index + self._window_size + 1)
        window_words = corpus[lower_bound:upper_bound]
        window_words = np.delete(window_words, (target_word_index - lower_bound))
        word_ids = np.array([token_map[word] for word in window_words])
        return word_ids

    def __getNegativeSamplesIds(self) -> np.ndarray:
        return self._helper_functions.getNegativeSamples(self._number_of_negative_samples)

    def __load_embeddings(self):
        try:
            self._embeddings = np.load("word_embeddings.npy")
        except FileNotFoundError:
            self._embeddings = None

    def __load_token_map(self) -> dict[str, int]:
        try:
            token_map = np.load("token_map.npy", allow_pickle=True).item()
        except FileNotFoundError:
            token_map = None
        return token_map