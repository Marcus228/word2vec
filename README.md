# NSSK word2vec
Single-layer perceptron model which produces word embeddings for the given text corpus.

## Pre-conditions
The model assumes a well-formed, cleaned corpus text to train on. Passing a non-clean text might result in upredictable results.

## How to use
The file `word2vec.py` contains the class that wraps, and manages the model. It requires numpy module to be installed in the SDK and `BinaryClassifiers` with `w2vHelperModule` modules to be present in the same location as the file itself. 
To use the model, create a new instance of the Word2Vec object, and optionally provide hyperparameters. Hyperparameters of the model include:
```
* learning_rate : float // Determines the learning step of the model.
* epochs : int // Number of reruns of the training loop. 
* embedding_dimension : int // The dimension of final word embedding.
* window_size : int // The context window size of a target word.
* number_of_negative_samples : int // The number of unrelated word samples per target word.
* batch_size : int // The size of batches in which the training loop processes data.
* save_embeddings : bool // Setting to false will not save the word embeddings after training.
```
Then, call `train()` method, with text corpus as an input. <br />
Run the Python script. <br />
The model automatically saves itself after it finishes training. <br />
To get a specific embedding call `getEmbeddingOf()`, with a desired string. <br />


## Visualisation of Results
The repo also contains a `VisualiserScript.py` file. It requires a couple of files to be present in the same location as itself. Those include:
* `word_embeddings.npy`
* `tokens_map.npy`
* `unigram_frequency.npy`

The repo contains the previously mentioned files for a 650000 word Wiki scrape text data.

Note that the model generates initial embedding data during training using `numpy.random.uniform` with a pre-defined seed of 42.

## Technical details
The model is based on word2vec methodology with a negative sampling skip-gram approach.
It is realised in a matrix factorisation style and is optimised to leverage numpy's vectorisation feature for faster processing.

## Built using
* numpy
* python

## References and Acknowledgments
Thanks to [Dr. Chiraag Lala](https://www.linkedin.com/in/chiraagrlala/) for his explanation of the concepts and ideas used throughout. 

Some resources I used for inspiration and research: 
* [nuric Blog](https://www.doc.ic.ac.uk/~nuric/posts/teaching/word-representations/)
* _Efficient Estimation of Word Representations in Vector Space_, Mikolov et al., 2013, [Ref.](https://arxiv.org/abs/1301.3781)
* Google Gemini (research)
* [Project Tensorflow](https://projector.tensorflow.org/)
