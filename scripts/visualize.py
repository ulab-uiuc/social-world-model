import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from litellm import embedding
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def find_top_similar_pairs(embeddings, assumptions, assumption_map, top_n=5):
    """
    Find the top N most similar assumption pairs based on cosine similarity,
    including the arxiv_id for each assumption.
    """
    # Compute cosine similarity matrix for all assumption pairs
    similarity_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(similarity_matrix, 0)  # Ignore self-similarity

    # Find the top N pairs
    top_pairs = []
    for _ in range(top_n):
        max_sim = np.max(similarity_matrix)
        max_idx = np.unravel_index(
            np.argmax(similarity_matrix), similarity_matrix.shape
        )
        i, j = max_idx
        arxiv_id_1 = assumption_map[assumptions[i]]
        arxiv_id_2 = assumption_map[assumptions[j]]
        top_pairs.append(
            (arxiv_id_1, assumptions[i], arxiv_id_2, assumptions[j], max_sim)
        )
        similarity_matrix[i, j] = 0  # Zero out to avoid reselecting the same pair

    return top_pairs


def get_embedding(text, model='text-embedding-3-large'):
    """
    Generate an embedding for a given text using OpenAI's text-embedding API.
    """
    response = embedding(model=model, input=[text])
    return response['data'][0]['embedding']


def load_assumptions(input_file: str):
    """
    Load assumptions from the input JSON file.
    """
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Collect all assumptions into a list
    assumptions = []
    assumption_map = {}  # To map assumptions back to their arxiv_id
    for arxiv_id, content in data.items():
        for assumption in content['assumptions']:
            assumptions.append(assumption)
            assumption_map[assumption] = arxiv_id
    return assumptions, assumption_map


def embed_assumptions(assumptions, cache_file='embeddings_cache.json'):
    """
    Generate embeddings for each assumption sentence, with caching.
    """
    # Load cache if it exists
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cached_embeddings = json.load(f)
    else:
        cached_embeddings = {}

    embeddings = []
    updated_cache = False  # Track if we add new embeddings to the cache

    for assumption in tqdm(assumptions):
        if assumption in cached_embeddings:
            embedding = cached_embeddings[assumption]
        else:
            embedding = get_embedding(assumption)
            cached_embeddings[assumption] = embedding
            updated_cache = True
        embeddings.append(embedding)

    # Save updated cache if new embeddings were added
    if updated_cache:
        with open(cache_file, 'w') as f:
            json.dump(cached_embeddings, f)

    return np.array(embeddings)


def visualize_embeddings(embeddings, assumptions, output_file=None):
    """
    Visualize embeddings using PCA for dimensionality reduction and save or show the plot.
    """
    # Reduce dimensions to 2D for visualization
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.6)

    # Annotate each point with a small sample of the assumption text for reference
    for i, assumption in enumerate(assumptions):
        plt.annotate(
            assumption[:15] + '...',
            (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
            fontsize=8,
            alpha=0.7,
        )

    plt.title('Embedding Visualization of Assumptions')
    plt.xlabel('PCA Dimension 1')
    plt.ylabel('PCA Dimension 2')

    # Save or show plot based on `output_file`
    if output_file:
        plt.savefig(output_file)
        print(f'Figure saved to {output_file}')
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize or save embeddings of assumption sentences.'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        default='../data/iclrbench_decompose.json',
        help='Path to the input JSON file containing assumptions.',
    )
    parser.add_argument(
        '--output_file',
        type=str,
        help='Path to save the output visualization image. If not specified, the plot will be shown instead.',
    )
    parser.add_argument(
        '--top_n',
        type=int,
        default=50,
        help='Number of top similar assumption pairs to display.',
    )
    parser.add_argument(
        '--cache_file',
        type=str,
        default='embeddings_cache.json',
        help='File to store cached embeddings.',
    )

    args = parser.parse_args()

    # Step 1: Load assumptions
    assumptions, assumption_map = load_assumptions(args.input_file)

    # Step 2: Embed assumptions (with caching)
    embeddings = embed_assumptions(assumptions, cache_file=args.cache_file)

    # Step 3: Visualize embeddings (either save or show based on `output_file`)
    visualize_embeddings(embeddings, assumptions, output_file=args.output_file)

    # Step 4: Find and display top similar assumption pairs
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize or save embeddings of assumption sentences.'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        default='../data/iclrbench_decompose.json',
        help='Path to the input JSON file containing assumptions.',
    )
    parser.add_argument(
        '--output_file',
        type=str,
        help='Path to save the output visualization image. If not specified, the plot will be shown instead.',
    )
    parser.add_argument(
        '--top_n',
        type=int,
        default=100,
        help='Number of top similar assumption pairs to display.',
    )
    parser.add_argument(
        '--cache_file',
        type=str,
        default='embeddings_cache.json',
        help='File to store cached embeddings.',
    )

    args = parser.parse_args()

    # Step 1: Load assumptions
    assumptions, assumption_map = load_assumptions(args.input_file)

    # Step 2: Embed assumptions (with caching)
    embeddings = embed_assumptions(assumptions, cache_file=args.cache_file)

    # Step 3: Visualize embeddings (either save or show based on `output_file`)
    visualize_embeddings(embeddings, assumptions, output_file=args.output_file)

    # Step 4: Find and display top similar assumption pairs with arxiv_id
    top_pairs = find_top_similar_pairs(
        embeddings, assumptions, assumption_map, top_n=args.top_n
    )
    print(f'\nTop {args.top_n} most similar assumption pairs with arxiv_id:')
    for i, (arxiv_id_1, assumption1, arxiv_id_2, assumption2, similarity) in enumerate(
        top_pairs, start=1
    ):
        if arxiv_id_1 == arxiv_id_2:
            continue
        print(f'\nPair {i} (Similarity: {similarity:.4f}):')
        print(f'Arxiv ID 1: {arxiv_id_1} | Assumption 1: {assumption1}')
        print(f'Arxiv ID 2: {arxiv_id_2} | Assumption 2: {assumption2}')
