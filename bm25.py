import math
from typing import List


def documents_containing_term(term: str, documents: List[str]) -> int:
    return sum(1 for doc in documents if term in doc.lower().split())


def inverse_document_frequency(term: str, documents: List[str]) -> float:
    df = documents_containing_term(term, documents)
    if df == 0:
        return 0.0
    return math.log((len(documents) - df + 0.5) / (df + 0.5) + 1)


def term_frequency(term: str, document: str) -> float:
    words = document.lower().split()
    return words.count(term) / len(words)


def avg_document_len(documents: List[str]) -> float:
    return sum(len(doc.split()) for doc in documents) / len(documents)


def bm25(query: str, document: str, documents: List[str], k1=1.5, b=0.75) -> float:
    score   = 0.0
    avgdl   = avg_document_len(documents)
    doc_len = len(document.lower().split())
    for term in query.lower().split():
        tf  = term_frequency(term, document)
        idf = inverse_document_frequency(term, documents)
        score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl))
    return score


def rank_documents(query: str, documents: List[str]) -> List[tuple]:
    scores = [(doc, bm25(query, doc, documents)) for doc in documents]
    return sorted(scores, key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    documents = [
        "the transformer model uses attention mechanism for sequence modeling",
        "BM25 is a sparse retrieval algorithm based on term frequency",
        "dense retrieval uses embeddings and cosine similarity for search",
        "attention mechanism computes scaled dot product of query key value",
        "RAG combines retrieval with language model generation",
    ]

    for query in ["attention mechanism transformer", "retrieval search embeddings"]:
        print(f"\nQuery: '{query}'")
        print("-" * 55)
        for doc, score in rank_documents(query, documents):
            print(f"  {score:.4f}  {doc[:55]}...")