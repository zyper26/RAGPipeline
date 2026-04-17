from hybrid_retriever import HybridRetriever
from cross_encoder_rerank import CrossEncoderReranker
from sentence_transformers import SentenceTransformer
import numpy as np

class RAGPipeline:
    def __init__(self, documents, embeddings, embedding_dim=384, embedding_model=None):
        self.documents = documents
        self.retriever = HybridRetriever(documents, embeddings, embedding_dim)
        self.reranker  = CrossEncoderReranker()
        if embedding_model is None:
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        else:
            self.embedding_model = embedding_model

    def query(self, query: str, query_embedding = None, top_k=5) -> str:
        if query_embedding is None:
            query_embedding = np.array(self.embedding_model.encode([query]), dtype=np.float32)
        top_50_docs = self.retriever.retrieve(query=query, \
                                            query_embedding=query_embedding[0], \
                                            top_k=top_k)
        
        retrieved_docs = [doc for doc, _ in top_50_docs] 
        top_5 = self.reranker.rerank(query=query, documents=retrieved_docs, top_k=top_k)
        
        context = [doc for doc, _ in top_5]
        context_prompt = ""
        for con in context:
            context_prompt += con + "\n" 
        prompt = f"Answer the question using the context below.\n\nContext:\n{context_prompt}\n\nQuestion: {query}\nAnswer:"
        return prompt

# ── Test ──────────────────────────────────────────────────────
 
if __name__ == "__main__":
    np.random.seed(42)
 
    documents = [
        "transformer attention mechanism uses query key value projections",
        "BM25 is a sparse retrieval algorithm based on term frequency and IDF",
        "dense retrieval encodes documents as vectors and uses cosine similarity",
        "RAG combines retrieval with language model generation for grounded answers",
        "scaled dot product attention divides by sqrt of d_k for stable gradients",
        "cross encoders jointly encode query and document for accurate reranking",
        "HNSW is a graph based approximate nearest neighbour algorithm",
        "reciprocal rank fusion combines sparse and dense retrieval rankings",
    ]

    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    document_embedding = embedding_model.encode(documents)
 
    embedding_dim = 384
    embeddings    = np.random.randn(len(documents), embedding_dim)
 
    pipeline      = RAGPipeline(documents, document_embedding, embedding_dim, embedding_model=embedding_model)
    query         = "how does attention mechanism work"
   
    print(f"Prompt: {pipeline.query(query, top_k=3)}")
 