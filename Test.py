import word2vec as w2v

w2v = w2v.Word2Vec(learning_rate=0.01, epochs=40, embedding_dimension=20, window_size=5, number_of_negative_samples = 10)

w2v.train("testText")

print("Embedding of 'the' :", w2v.getEmbeddingOf("the"))