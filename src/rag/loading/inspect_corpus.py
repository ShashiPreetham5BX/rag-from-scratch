"""
Sanity-check the SciFact corpus load.

Run from the project root:
    python src/rag/loading/inspect_corpus.py
"""

from pathlib import Path

from rag.loading.jsonl_loader import JSONLLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CORPUS_PATH = PROJECT_ROOT / "data" / "raw" / "scifact" / "corpus.jsonl"


def main() -> None:
    loader = JSONLLoader(CORPUS_PATH, source_name="scifact")
    docs = loader.load()

    print(f"\nLoaded {len(docs):,} documents\n")

    print("--- First document ---")
    d = docs[0]
    print(f"id:     {d.doc_id!r}  (type: {type(d.doc_id).__name__})")
    print(f"title:  {d.title[:90]}")
    print(f"text:   {d.text[:200]}...")
    print(f"source: {d.source}")
    print(f"chars:  {len(d)}")

    lengths = [len(d) for d in docs]
    lengths.sort()
    n = len(lengths)
    print("\n--- Length distribution (characters) ---")
    print(f"min:     {lengths[0]:,}")
    print(f"median:  {lengths[n // 2]:,}")
    print(f"mean:    {sum(lengths) // n:,}")
    print(f"p95:     {lengths[int(n * 0.95)]:,}")
    print(f"max:     {lengths[-1]:,}")

    ids = [d.doc_id for d in docs]
    print(f"\nunique ids: {len(set(ids)):,} / {len(ids):,}")
    print(f"all ids are str: {all(isinstance(i, str) for i in ids)}")


if __name__ == "__main__":
    main()