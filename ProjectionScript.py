import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_top_embeddings(embeddings_path, token_map_path, unigram_path, num_words=300):
    print("Loading data...")

    # 1. Load the numpy arrays
    embeddings = np.load(embeddings_path)

    # allow_pickle=True is required to unpack the dictionaries
    token_map = np.load(token_map_path, allow_pickle=True).item()
    unigram_distribution = np.load(unigram_path, allow_pickle=True).item()

    # 2. Sort words by frequency (highest to lowest)
    # This ensures we only plot the words the model has learned best
    print("Sorting vocabulary by frequency...")
    sorted_words = sorted(unigram_distribution.keys(),
                          key=lambda w: unigram_distribution[w],
                          reverse=True)

    # 3. Grab the top N most frequent words
    words_to_plot = sorted_words[:num_words]

    # 4. Get their specific IDs to fetch their embeddings
    # Note: If your token_map IDs start at 1, but your embeddings matrix
    # is 0-indexed without a padding row, you may need to do `token_map[w] - 1`
    indices_to_plot = [token_map[w] for w in words_to_plot]
    embeddings_to_plot = embeddings[indices_to_plot]

    # 5. Project down to 2D using t-SNE
    print(f"Projecting top {num_words} words to 2D using t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=10)
    embeddings_2d = tsne.fit_transform(embeddings_to_plot)

    # 6. Plot the results
    plt.figure(figsize=(14, 10))

    x_coords = embeddings_2d[:, 0]
    y_coords = embeddings_2d[:, 1]
    plt.scatter(x_coords, y_coords, alpha=0.6, color='dodgerblue')

    # Annotate each point with its actual word
    for i, word in enumerate(words_to_plot):
        plt.annotate(word,
                     (x_coords[i], y_coords[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom',
                     fontsize=9)

    plt.title(f"Top {num_words} Word Embeddings (t-SNE)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()