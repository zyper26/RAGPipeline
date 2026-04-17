import numpy as np
from typing import List, Tuple

class DenseRetriever:
    def __init__(self, documents, embedding_dim=384):
        self.documents = documents
        self.embedding = np.zeros((len(self.documents), embedding_dim))

    def add_documents(self, documents: List[str], embeddings: np.ndarray):
        self.embedding = embeddings

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        dot_product = np.dot(b, a)
        norm = np.linalg.norm(a) * np.linalg.norm(b, axis=1)
        return dot_product/norm

    def retrieve(self, query_embedding: np.ndarray, top_k=5) -> List[Tuple[str, float]]:
        scores = self.cosine_similarity(query_embedding, self.embedding)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.documents[i], scores[i]) for i in top_indices]