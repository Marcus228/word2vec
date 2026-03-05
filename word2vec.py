import numpy as np
import w2vHelperModule as helper_module
import BinaryClassifiers as binary_classifiers


class Word2Vec:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 100, embedding_dimension: int = 30, window_size: int = 5, negative_samples: int = 10):
        if (learning_rate <= 0) or (epochs <= 0) or (embedding_dimension <= 0) or (window_size <= 0) or (negative_samples <= 0):
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
        self._negative_samples = negative_samples
        self._helper_functions = None
        self.current_epoch = 0
        self._embeddings = None

    def train(self, corpus: np.ndarray):
        # initialise the helper object
        self._helper_functions = helper_module.Word2VecHelper(corpus)
        token_map = self._helper_functions.getTokens()

        # initialise the buffers, which will hold the word ids, context ids and labels
        batch_size = 128
        target_buffer, context_buffer, label_buffer = np.zeros(0), np.zeros(0), np.zeros(0)

        # initialise the word and context matrices
        word_matrix = np.random.uniform(-0.5, 0.5, (len(token_map), self._embedding_dimension))
        context_matrix = np.random.uniform(-0.5, 0.5, (len(token_map), self._embedding_dimension))

        # now we train the model
        for epoch in range(self._epochs):
            # iterate over the corpus
            for target_word in corpus:
                target_word_index = token_map[target_word]

                # get the positive and negative samples for this word
                positive_samples_ids = self.__getPositiveSamples(target_word_index, token_map, corpus)
                negative_samples_ids = self.__getNegativeSamples()

                # create the training data in the buffers
                for positive_sample_id in positive_samples_ids:
                    np.append(target_buffer, target_word_index)
                    np.append(context_buffer, positive_sample_id)
                    np.append(label_buffer, 1)
                    for negative_sample_id in negative_samples_ids:
                        np.append(target_buffer, target_word_index)
                        np.append(context_buffer, negative_sample_id)
                        np.append(label_buffer, 0)

                # check if the buffer is full, and if so, train the model on this batch
                if target_buffer.size >= batch_size:
                    self.__train_step(target_buffer, context_buffer, label_buffer, word_matrix, context_matrix)
                    target_buffer, context_buffer, label_buffer = [], [], []

            # have to include the last batch, if it is not empty
            if target_buffer.size > 0:
                self.__train_step(target_buffer, context_buffer, label_buffer, word_matrix, context_matrix)
                target_buffer, context_buffer, label_buffer = [], [], []
            print(f"Epoch {epoch} complete")
        print("Training complete")

        # at this point, the model is trained
        # now we can merge the word and context matrices to get a meaningful embedding matrix and save it
        word_embedding = np.add(word_matrix, context_matrix)
        np.save("word_embeddings.npy", word_embedding)

    def getEmbedding(self, word : str) -> np.ndarray:
        self.__load_embeddings()
        if self._embeddings is None or self._helper_functions is None:
            raise FileNotFoundError("No embedding found. Please train the model first")
        embedding_id = self._helper_functions.getId(word)
        return self._embeddings[embedding_id]

    #private methods
    def __train_step(self, targets: np.ndarray, contexts: np.ndarray, labels: np.ndarray, word_matrix: np.ndarray, context_matrix: np.ndarray):
        dots = np.array([np.dot(word_matrix[word_id], context_matrix[context_id]) for word_id, context_id in zip(targets, contexts)])
        sigmoid = map(binary_classifiers.sigmoid, dots)
        errors = labels - sigmoid
        target_gradients = self._learning_rate * errors * map(lambda context : word_matrix[context], contexts)
        context_gradients = self._learning_rate * errors * map(lambda word : word_matrix[word], targets)
        np.add.at(word_matrix, targets, target_gradients)
        np.add.at(context_matrix, contexts, context_gradients)
        self.current_epoch += 1
        print(f"Epoch {self.current_epoch} complete")

    def __getPositiveSamples(self, target_word_id : int, token_map: dict, corpus: np.ndarray) -> np.ndarray:
        accumulator = np.array(1)
        for corpus_target_word_id in range(max(target_word_id - self._window_size, 0), min(target_word_id + self._window_size, len(corpus))):
            context_id = token_map[corpus[corpus_target_word_id]]
            if context_id != target_word_id:
                accumulator = np.append(accumulator, context_id)
        return accumulator

    def __getNegativeSamples(self) -> np.ndarray:
        return self._helper_functions.getNegativeSamples(self._negative_samples)

    def __load_embeddings(self):
        self._embeddings = np.load("word_embeddings.npy")