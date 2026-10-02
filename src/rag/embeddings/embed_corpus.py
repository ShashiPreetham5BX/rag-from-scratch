"""Load -> clean -> chunk -> embed all of SciFact, and save vectors + chunk table."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from rag.chunking.fixed import FixedSizeChunker
from rag.chunking.sentence import SentenceChunker
from rag.embeddings.embedder import Embedder
from rag.loading.jsonl_loader import JSONLLoader
from rag.preprocessing.pipelines import get_pipeline
from rag.tokenization.tokenizer import Tokenizer

CHUNKERS = {"fixed": FixedSizeChunker, "sentence": SentenceChunker}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--chunker", choices=list(CHUNKERS), default="sentence")
    p.add_argument("--size", type=int, default=254)
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--limit", type=int, default=None, help="only the first N documents")
    args = p.parse_args()

    tok = Tokenizer()
    if args.size > tok.content_budget:
        raise SystemExit(f"--size {args.size} exceeds the real budget of {tok.content_budget} tokens "
                         f"(chunks would be silently truncated).")

    docs = JSONLLoader("data/raw/scifact/corpus.jsonl").load()
    if args.limit:
        docs = docs[:args.limit]
    cleaner = get_pipeline("standard")
    chunker = CHUNKERS[args.chunker](tok, args.size, args.overlap)

    chunks = []
    for d in tqdm(docs, desc="Chunking"):
        chunks.extend(chunker.chunk(d.doc_id, cleaner.clean(d.text)))
    print(f"{len(docs):,} docs -> {len(chunks):,} chunks")

    embedder = Embedder(max_length=tok.max_length)
    texts = [c.text for c in chunks]
    t0 = time.time()
    parts = [embedder.embed(texts[i:i + 256]) for i in tqdm(range(0, len(texts), 256), desc="Embedding")]
    emb = np.vstack(parts).astype(np.float32)
    seconds = time.time() - t0

    out = Path(f"data/processed/scifact_{args.chunker}_{args.size}")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", emb)
    with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    meta = {**vars(args), "model": embedder.model_name, "n_docs": len(docs),
            "n_chunks": len(chunks), "dim": int(emb.shape[1]), "embed_seconds": round(seconds, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nSaved {emb.shape} ({emb.nbytes / 1e6:.1f} MB) to {out}")
    print(f"Embedding took {seconds:.0f}s ({len(chunks) / seconds:.0f} chunks/s)")


if __name__ == "__main__":
    main()