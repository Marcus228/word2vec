import word2vec as w2v
import ProjectionScript as ps

w2v = w2v.Word2Vec(epochs=5, embedding_dimension=20, window_size=5, number_of_negative_samples = 5)

# w2v.train("testText")

ps.plot_top_embeddings("word_embeddings.npy", "token_map.npy", "unigram_frequency.npy")