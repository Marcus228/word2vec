import numpy as np

import Tokeniser as tk
import BinaryClassifiers as bcs

t = tk.SimpleTokeniser()
corpus = np.loadtxt("testText", dtype=str)
tokens = t.tokeniseCorpus(corpus)
print(corpus, corpus.shape)
for cord in corpus:
    print(cord)
print("dimension of corpus", corpus.shape)
print("tokens size : ", len(tokens))
print(tokens)

embedding_dimension = 10
# each row is a word embedding
word_matrix = np.random.rand(len(tokens), embedding_dimension)
context_matrix = np.random.rand(len(tokens), embedding_dimension)
# context_matrix = np.zeroes(tokens)
print(word_matrix, word_matrix.shape)
print(context_matrix, context_matrix.shape)

# experimenting with numpy broadcasting feature
# using broadcasting can technically allow me to do all the dot product computations at once
# choose an arbitrary word
word_index = 2
context_index1 = word_index + 5
context_index2 = word_index - 1
print("The word is",word_matrix[word_index])
print("The contexts are",context_matrix[context_index1])
# extend the original word embedding to the context embedding
brdcstd_arrs = np.broadcast_arrays(word_matrix[word_index], [context_matrix[context_index1], context_matrix[context_index2]])
print("The broadcasted word_embedding is", brdcstd_arrs[0])
print("The original word_contexts are", brdcstd_arrs[1])
# map a dot product over the broadcasted word_embedding and word_contexts
dot_product = np.array([np.dot(word, context) for word, context in zip(brdcstd_arrs[0], brdcstd_arrs[1])])
print("The dot product is", dot_product)
# map sigmoid function over the dot product matrix
sigmoid = np.array([bcs.sigmoid(dot) for dot in dot_product])
print(sigmoid)