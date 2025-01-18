import torch
import gensim.downloader as api
from gensim.downloader import load
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load pretrained Word2Vec embeddings (Google News or any other corpus)
print("Loading Word2Vec model...")
word2vec_model = load("glove-twitter-25")  # About 100MB model

print("Word2Vec model loaded.")

# Define example terms
terms = [
    "king", "queen", "man", "woman", "prince", "princess",  # Royalty
    "apple", "orange", "banana", "grape",                   # Fruits
    "car", "bus", "train", "bicycle",                      # Transportation
    "dog", "cat", "lion", "tiger",                         # Animals
    "ocean", "river", "mountain", "forest"                 # Nature
]

# Filter terms present in the Word2Vec vocabulary
available_terms = [term for term in terms if term in word2vec_model]

# Retrieve embeddings for the terms
term_embeddings = np.array([word2vec_model[term] for term in available_terms])

# Calculate pairwise cosine similarity
def cosine_similarity(embedding1, embedding2):
    embedding1 = torch.tensor(embedding1, dtype=torch.float32)
    embedding2 = torch.tensor(embedding2, dtype=torch.float32)
    return torch.nn.functional.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0)).item()

# Store pairwise similarities in a list
similarity_scores = []
for i, term1 in enumerate(available_terms):
    for j, term2 in enumerate(available_terms):
        if i < j:  # Avoid duplicate pairs and self-comparison
            similarity = cosine_similarity(term_embeddings[i], term_embeddings[j])
            similarity_scores.append((term1, term2, similarity))

# Sort similarities in descending order
sorted_similarities = sorted(similarity_scores, key=lambda x: x[2], reverse=True)

# Print ordered similarities
print("\nPairwise Cosine Similarities (High to Low):")
for term1, term2, similarity in sorted_similarities:
    print(f"{term1} vs {term2}: {similarity:.2f}")

# Visualize embeddings using t-SNE
def plot_embeddings(embeddings, terms):
    tsne = TSNE(n_components=2, perplexity=min(len(terms) // 2, 30), random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    for i, term in enumerate(terms):
        x, y = reduced_embeddings[i]
        plt.scatter(x, y, marker='o')
        plt.text(x + 0.02, y + 0.02, term, fontsize=12)
    plt.title("Word2Vec Embeddings: Semantic Relationships")
    plt.grid(True)
    plt.show()

plot_embeddings(term_embeddings, available_terms)



def plot_embeddings_semantic(embeddings, terms):
    tsne = TSNE(n_components=2, perplexity=min(len(terms) // 2, 30), random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 8))
    for i, term in enumerate(terms):
        x, y = reduced_embeddings[i]
        plt.scatter(x, y, marker='o', color='blue')
        plt.text(x + 0.02, y + 0.02, term, fontsize=12, color='black')
    plt.title("Semantic Relationships in Word Embeddings")
    plt.grid(True)
    plt.show()

# Example Usage:
plot_embeddings_semantic(term_embeddings, terms)

import umap.umap_ as umap

def plot_umap(embeddings, terms):
    reducer = umap.UMAP(n_neighbors=10, random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 8))
    for i, term in enumerate(terms):
        x, y = reduced_embeddings[i]
        plt.scatter(x, y, marker='o', color='green')
        plt.text(x + 0.02, y + 0.02, term, fontsize=12, color='darkred')
    plt.title("UMAP Visualization of Semantic Relationships")
    plt.grid(True)
    plt.show()

# Example Usage:
plot_umap(term_embeddings, terms)
