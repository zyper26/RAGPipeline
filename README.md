# RAG Pipeline

A from-scratch Retrieval-Augmented Generation (RAG) pipeline built in Python, featuring hybrid retrieval (BM25 + dense embeddings) fused with Reciprocal Rank Fusion (RRF) and cross-encoder reranking.

## Architecture

```
Query
  │
  ├──► BM25 (sparse)        ──┐
  │                            ├──► RRF Fusion ──► Cross-Encoder Rerank ──► Prompt
  └──► Dense (cosine sim)  ──┘
```

### Components

| Module | Class / Function | Description |
|---|---|---|
| `bm25.py` | `bm25()`, `rank_documents()` | BM25 scoring with TF-IDF-style term weighting |
| `dense_retriever.py` | `DenseRetriever` | Cosine similarity retrieval over pre-computed embeddings |
| `hybrid_retriever.py` | `HybridRetriever` | Fuses BM25 + dense rankings via Reciprocal Rank Fusion |
| `cross_encoder_rerank.py` | `CrossEncoderReranker` | Re-scores top candidates with a cross-encoder model |
| `RAGPipeline.py` | `RAGPipeline` | End-to-end pipeline: retrieve → rerank → build prompt |

## How It Works

1. **BM25 (Sparse Retrieval)** — scores documents using term frequency and inverse document frequency with length normalization (`k1=1.5`, `b=0.75`).

2. **Dense Retrieval** — encodes query and documents as vectors; ranks by cosine similarity using `sentence-transformers/all-MiniLM-L6-v2` (384-dim).

3. **Reciprocal Rank Fusion** — merges both ranked lists without requiring score normalization:
   ```
   RRF(doc) = Σ 1 / (k + rank)   where k = 60
   ```

4. **Cross-Encoder Reranking** — re-scores the fused top-N candidates jointly on (query, document) pairs using `cross-encoder/ms-marco-MiniLM-L-6-v2` for high-precision ranking.

5. **Prompt Construction** — top-k documents are assembled into a context block and returned as a ready-to-use LLM prompt.

## Installation

```bash
pip install sentence-transformers numpy
```

## Usage

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from RAGPipeline import RAGPipeline

documents = [
    "transformer attention mechanism uses query key value projections",
    "BM25 is a sparse retrieval algorithm based on term frequency and IDF",
    "dense retrieval encodes documents as vectors and uses cosine similarity",
    "RAG combines retrieval with language model generation for grounded answers",
]

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = embedding_model.encode(documents)

pipeline = RAGPipeline(documents, embeddings, embedding_dim=384, embedding_model=embedding_model)

prompt = pipeline.query("how does attention mechanism work", top_k=3)
print(prompt)
```

**Output:**
```
Answer the question using the context below.

Context:
transformer attention mechanism uses query key value projections
scaled dot product attention divides by sqrt of d_k for stable gradients
cross encoders jointly encode query and document for accurate reranking

Question: how does attention mechanism work
Answer:
```

## Running Individual Modules

```bash
# Test BM25 retrieval
python bm25.py

# Test hybrid retrieval (BM25 + dense + RRF)
python hybrid_retriever.py

# Test cross-encoder reranking
python cross_encoder_rerank.py

# Run the full pipeline
python RAGPipeline.py
```

## Models Used

| Role | Model |
|---|---|
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |

## Project Structure

```
RAG/
├── RAGPipeline.py          # End-to-end pipeline
├── hybrid_retriever.py     # RRF fusion of sparse + dense
├── dense_retriever.py      # Cosine similarity retrieval
├── bm25.py                 # BM25 sparse retrieval
├── cross_encoder_rerank.py # Cross-encoder reranker
└── test.ipynb              # Exploratory notebook
```
