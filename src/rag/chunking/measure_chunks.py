"""Measure how chunk size changes chunk counts across all of SciFact."""

import json
import statistics
from pathlib import Path

from rag.chunking.fixed import FixedSizeChunker
from rag.loading.jsonl_loader import JSONLLoader
from rag.preprocessing.pipelines import get_pipeline
from rag.tokenization.tokenizer import Tokenizer

SIZES = [64, 128, 254]   # 510 = 512 minus the 2 special tokens
OVERLAP_FRACTION = 8          # overlap = size // 8  (12.5%)


def main() -> None:
    docs = JSONLLoader("data/raw/scifact/corpus.jsonl").load()
    cleaner = get_pipeline("standard")
    texts = [(d.doc_id, cleaner.clean(d.text)) for d in docs]

    tok = Tokenizer()
    doc_tokens = [tok.count(t) for _, t in texts]
    print(f"\nDocuments: {len(texts):,}")
    print(f"Tokens per doc: median={statistics.median(doc_tokens):.0f}  "
          f"mean={statistics.mean(doc_tokens):.0f}  max={max(doc_tokens)}")

    results = {}
    print(f"\n{'size':>5} {'overlap':>7} {'chunks':>8} {'per doc':>8} "
          f"{'1-chunk docs':>13} {'avg toks':>9} {'min':>4}")
    for size in SIZES:
        overlap = size // OVERLAP_FRACTION
        chunker = FixedSizeChunker(tok, chunk_size=size, overlap=overlap)
        counts, lengths = [], []
        for doc_id, text in texts:
            chunks = chunker.chunk(doc_id, text)
            counts.append(len(chunks))
            lengths.extend(c.n_tokens for c in chunks)

        total = sum(counts)
        single = sum(1 for n in counts if n == 1) / len(counts)
        results[size] = {
            "overlap": overlap,
            "total_chunks": total,
            "chunks_per_doc": round(total / len(counts), 2),
            "pct_single_chunk_docs": round(100 * single, 1),
            "avg_chunk_tokens": round(statistics.mean(lengths), 1),
            "min_chunk_tokens": min(lengths),
        }
        print(f"{size:>5} {overlap:>7} {total:>8,} {total / len(counts):>8.2f} "
              f"{100 * single:>12.1f}% {statistics.mean(lengths):>9.1f} {min(lengths):>4}")

    out = Path("results/chunk_stats.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()