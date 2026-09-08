# RAG From Scratch

A modular Retrieval-Augmented Generation system built from primitives —
implementing document processing, chunking, embeddings, vector search,
sparse retrieval, reranking, and evaluation without high-level RAG frameworks.

**Status:** In development (Phase 1 — environment setup)

## Motivation

Most RAG implementations rely on frameworks that abstract away the retrieval
logic. This project implements the core components directly in order to
study how each design decision — chunking strategy, embedding model,
similarity metric, retrieval depth, reranking — affects retrieval and
answer quality, measured empirically.

## Planned components

- Document loading and preprocessing
- Multiple chunking strategies (fixed, sentence-aware, recursive, semantic)
- Transformer embeddings with manual pooling
- Custom vector index (flat and IVF-style), benchmarked against FAISS
- BM25 sparse retrieval and hybrid fusion
- Cross-encoder reranking
- Evaluation framework: Recall@k, MRR, nDCG@k, answer quality, latency
- Experimental comparison across configurations

## License

MIT