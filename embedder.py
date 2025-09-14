from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model once
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def get_embeddings(texts: List[str]) -> np.ndarray:
    """
    Generate embeddings for a list of text chunks.

    Args:
        texts (List[str]): List of sentence or chunk strings

    Returns:
        np.ndarray: 2D numpy array of shape (n_chunks, embedding_dim)
    """
    return np.array(model.encode(texts, convert_to_numpy=True, normalize_embeddings=True))

# Test
if __name__ == "__main__":
    chunks = [
        "Large language models have shown strong capabilities.",
        "This paper proposes a retrieval-augmented method for summarization."
    ]
    vectors = get_embeddings(chunks)
    print("Embedding shape:", vectors.shape)
    print("First vector (truncated):", vectors[0][:5])
