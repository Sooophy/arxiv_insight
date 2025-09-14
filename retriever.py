from typing import List, Tuple
import faiss
import numpy as np
from embedder import get_embeddings

class RAGRetriever:
    def __init__(self):
        self.index = None
        self.chunk_texts = []

    def build_index(self, chunks: List[str]):
        """
        Build FAISS index from list of text chunks.
        """
        self.chunk_texts = chunks
        embeddings = get_embeddings(chunks)  # (n_chunks, dim)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Cosine similarity (with normalized vectors), chosen because of small dataset and no compression
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search top-k relevant chunks for a given query string.

        Returns:
            List of (chunk_text, similarity_score)
        """
        query_embedding = get_embeddings([query])  # (1, dim)
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for i, score in zip(I[0], D[0]):
            if 0 <= i < len(self.chunk_texts):
                results.append((self.chunk_texts[i], float(score)))
        return results


# Test
if __name__ == "__main__":
    chunks = [
        "This paper introduces a new pose estimation method.",
        "The authors propose a transformer for object detection.",
        "We explore reinforcement learning in robotics."
    ]

    retriever = RAGRetriever()
    retriever.build_index(chunks)

    query = "pose estimation in 3D"
    results = retriever.search(query, top_k=2)

    print("Query:", query)
    print("Top results:")
    for chunk, score in results:
        print(f"- ({score:.4f}) {chunk}")
