from bm25 import * 
from dense_retriever import * 
import math

class HybridRetriever:
    def __init__(self, documents, embeddings, embedding_dim=384):
        self.bm25_docs    = documents
        self.dense        = DenseRetriever(documents, embedding_dim)
        self.dense.add_documents(documents, embeddings)

    def retrieve(self, query: str, query_embedding: np.ndarray, top_k=5, k=60):
        sparse_embedding = rank_documents(query, documents=self.bm25_docs)
        dense_embedding = self.dense.retrieve(query_embedding=query_embedding, top_k = len(self.bm25_docs))
        rrf_scores = {}
        for rank, (doc, _) in enumerate(sparse_embedding, 1):
            rrf_scores[doc] = rrf_scores.get(doc, 0) + (1/(k+rank))
        for rank, (doc, _) in enumerate(dense_embedding, 1):
            rrf_scores[doc] = rrf_scores.get(doc, 0) + (1/(k+rank))
        return sorted(rrf_scores.items(), key=lambda x:x[1], reverse=True)[:top_k]


# ── Test ──────────────────────────────────────────────────────
 
if __name__ == "__main__":
    np.random.seed(42)
    documents = [
        "transformer attention mechanism query key value",
        "BM25 sparse retrieval term frequency algorithm",
        "dense embeddings cosine similarity vector search",
        "RAG retrieval augmented generation language model",
        "scaled dot product attention softmax weights",
    ]
 
    embedding_dim   = 16
    embeddings      = np.random.randn(len(documents), embedding_dim)
    embeddings[4]   = embeddings[0] + np.random.randn(embedding_dim) * 0.1
 
    query           = "attention mechanism transformer"
    query_embedding = embeddings[0] + np.random.randn(embedding_dim) * 0.1
 
    hybrid = HybridRetriever(documents, embeddings, embedding_dim)
 
    print(f"Query: '{query}'\n")
    print("BM25 (sparse):")
    for doc, score in rank_documents(query, documents)[:3]:
        print(f"  {score:.4f}  {doc}")
 
    print("\nDense:")
    for doc, score in hybrid.dense.retrieve(query_embedding, top_k=3):
        print(f"  {score:.4f}  {doc}")
 
    print("\nHybrid RRF:")
    for doc, score in hybrid.retrieve(query, query_embedding, top_k=3):
        print(f"  {score:.4f}  {doc}")
 