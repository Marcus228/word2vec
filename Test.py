import numpy as np
import w2vHelperModule as tk
import BinaryClassifiers as bcs

corpus = np.loadtxt("testText", dtype=str)
helper = tk.Word2VecHelper(corpus)
tokens = helper.getTokens()
print("dimension of corpus", corpus.shape)
print("tokens size : ", len(tokens))

embedding_dimension = 10
# each row is a word embedding
word_matrix = np.random.uniform(-0.5, 0.5, (len(tokens), embedding_dimension))
context_matrix = np.random.uniform(-0.5, 0.5, (len(tokens), embedding_dimension))
# context_matrix = np.zeroes(tokens)
print("Word matrix", word_matrix)
print("Word matrix shape", word_matrix.shape)
print("Context matrix shape", context_matrix.shape)
epochs = 100

while epochs > 0:
    # experimenting with numpy broadcasting feature
    # using broadcasting can technically allow me to do all the dot product computations at once
    # choose an arbitrary word
    word_index = tokens["term"]
    context_index1 = word_index + 5
    context_index2 = word_index - 1
    print("The word is",word_matrix[word_index])
    print("The contexts are",  [context_matrix[context_index1], context_matrix[context_index2]])

    # extend the original word embedding to the context embedding
    brdcstd_arrs = np.broadcast_arrays(word_matrix[word_index], [context_matrix[context_index1], context_matrix[context_index2]])
    print("The broadcasted word_embedding is", brdcstd_arrs[0])
    print("The original word_contexts are", brdcstd_arrs[1])

    # map a dot product over the broadcasted word_embedding and word_contexts
    dot_product = np.array([np.dot(word, context) for word, context in zip(brdcstd_arrs[0], brdcstd_arrs[1])])
    print("The dot products are", dot_product)

    # map sigmoid function over the dot product matrix
    sigmoid = np.array([bcs.sigmoid(dot) for dot in dot_product])
    print("Classification scores are", sigmoid)

    # calculate the loss and gradient
    learning_rate = 0.025
    errors = np.array([0,1]) - sigmoid
    print("Loss is", errors)
    gradient = errors[:, np.newaxis] * word_matrix[word_index]
    print("Gradient is", gradient)

    # update the word embeddings
    print("The original word embedding is", word_matrix[word_index])
    np.add.at(word_matrix, [word_index, word_index], gradient * learning_rate)
    print("Updated word embedding is", word_matrix[word_index])
    epochs -= 1