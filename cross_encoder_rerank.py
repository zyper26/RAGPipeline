from typing import List, Tuple
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # in production: load pretrained cross-encoder
        # for now: simulate with a scoring function
        self.model = CrossEncoder(model_name)

    def score(self, query: str, document: str) -> float:
        return self.model.predict([(query, document)])[0]


    def rerank(self, query: str, documents: List[str], top_k=5) -> List[Tuple[str, float]]:
        # score all documents
        # return top_k sorted by score
        scores = []
        for doc in documents:
            scores.append((doc, self.score(query, doc)))
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]


if __name__ == "__main__":
    documents = [
        "transformer attention mechanism query key value",
        "BM25 sparse retrieval term frequency algorithm",
        "dense embeddings cosine similarity vector search",
        "RAG retrieval augmented generation language model",
        "scaled dot product attention softmax weights",
    ]
    reranker = CrossEncoderReranker()
    for query in ["attention mechanism transformer", "retrieval search embeddings"]:
        print(f"\nQuery: '{query}'")
        print("-" * 55)
        for doc, score in reranker.rerank(query, documents):
            print(f"  {score:.4f}  {doc[:55]}...")