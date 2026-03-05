import numpy as np
import w2vHelperModule as tk
import word2vec as w2v

corpus = np.loadtxt("testText", dtype=str)
helper = tk.Word2VecHelper(corpus)
tokens = helper.getTokens()

w2v = w2v.Word2Vec(learning_rate=0.01, epochs=40, embedding_dimension=20, window_size=5, negative_samples=10)

w2v.train(corpus)

print("Embedding of 'the' :", w2v.getEmbedding("the"))