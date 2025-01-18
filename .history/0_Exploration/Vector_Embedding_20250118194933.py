import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Define example terms
terms = ["king", "queen", "man", "woman", "apple", "orange", "car", "bus"]

# Create a mapping of terms to indices
vocab = {term: idx for idx, term in enumerate(terms)}

# Number of terms in the vocabulary
vocab_size = len(vocab)

# Embedding dimension
embedding_dim = 2  # Use 2D for easier visualization

# Define an embedding layer
embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

# Initialize embeddings
torch.manual_seed(42)  # For reproducibility
embeddings = embedding_layer.weight.data

# Assign random embeddings
embedding_layer.weight.data.uniform_(-1, 1)

# Retrieve embeddings for the terms
term_embeddings = embeddings.detach().numpy()

# Calculate pairwise cosine similarity for the terms
def cosine_similarity(embedding1, embedding2):
    return torch.nn.functional.cosine_similarity(
        torch.tensor(embedding1).unsqueeze(0), 
        torch.tensor(embedding2).unsqueeze(0)
    ).item()

# Store pairwise similarities in a list
similarity_scores = []
for i, term1 in enumerate(terms):
    for j, term2 in enumerate(terms):
        if i < j:  # Avoid duplicate pairs and self-comparison
            similarity = cosine_similarity(term_embeddings[i], term_embeddings[j])
            similarity_scores.append((term1, term2, similarity))

# Sort similarities in descending order
sorted_similarities = sorted(similarity_scores, key=lambda x: x[2], reverse=True)

# Print ordered similarities
print("Pairwise Cosine Similarities (High to Low):")
for term1, term2, similarity in sorted_similarities:
    print(f"{term1} vs {term2}: {similarity:.2f}")


# Visualize the embeddings in 2D space
def plot_embeddings(embeddings, terms):
    plt.figure(figsize=(10, 8))
    for i, term in enumerate(terms):
        x, y = embeddings[i]
        plt.scatter(x, y, marker='o')
        plt.text(x + 0.02, y + 0.02, term, fontsize=12)
    plt.title("Semantic Relationships in Embedding Space")
    plt.grid(True)
    plt.show()

plot_embeddings(term_embeddings, terms)

# Optional: Use t-SNE for dimensionality reduction (for higher dimensions)
if embedding_dim > 2:
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(term_embeddings)
    plot_embeddings(reduced_embeddings, terms)
