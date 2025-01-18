import torch
import gensim.downloader as api
from gensim.downloader import load
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load pretrained Word2Vec embeddings (Google News or any other corpus)
print("Loading Word2Vec model...")
word2vec_model = load("glove-wiki-gigaword-100")  # About 100MB model
print("Word2Vec model loaded.")

# Define example terms
terms = ["king", "queen", "man", "woman", "apple", "orange", "car", "bus"]

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
    tsne = TSNE(n_components=2, random_state=42)
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
