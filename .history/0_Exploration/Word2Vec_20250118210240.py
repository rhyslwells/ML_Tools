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

import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

def plot_network(terms, embeddings):
    G = nx.Graph()

    # Convert embeddings to NumPy array for compatibility
    embeddings_np = np.array(embeddings)

    # Add nodes and edges based on cosine similarity
    for i, term1 in enumerate(terms):
        G.add_node(term1)
        for j, term2 in enumerate(terms):
            if i != j:
                similarity = cosine_similarity(embeddings_np[i].reshape(1, -1), embeddings_np[j].reshape(1, -1))[0][0]
                if similarity > 0.5:  # Threshold for similarity
                    G.add_edge(term1, term2, weight=similarity)

    # Create a spring layout for the network graph
    pos = nx.spring_layout(G)
    
    # Plot the graph
    plt.figure(figsize=(12, 12))
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color="skyblue")
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=12)
    plt.title("Network of Semantic Relationships")
    plt.show()

# Example Usage:
# Assume `terms` and `term_embeddings` are defined
plot_network(terms, term_embeddings)
