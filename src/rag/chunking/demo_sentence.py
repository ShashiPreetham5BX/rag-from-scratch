import json
import statistics
from itertools import islice

from rag.chunking.fixed import FixedSizeChunker
from rag.chunking.sentence import SentenceChunker
from rag.preprocessing.pipelines import get_pipeline
from rag.tokenization.tokenizer import Tokenizer


def main() -> None:
    tok = Tokenizer()
    cleaner = get_pipeline("standard")
    with open("data/raw/scifact/corpus.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in islice(f, 200)]
    docs = [(str(r["_id"]), cleaner.clean(r["text"])) for r in rows]

    for name, chunker in [("fixed", FixedSizeChunker(tok, 128, 16)),
                          ("sentence", SentenceChunker(tok, 128, 16))]:
        chunks = [c for doc_id, text in docs for c in chunker.chunk(doc_id, text)]
        starts = 100 * sum(c.text[0].isupper() for c in chunks) / len(chunks)
        ends = 100 * sum(c.text[-1] in ".!?" for c in chunks) / len(chunks)
        avg = statistics.mean(c.n_tokens for c in chunks)
        print(f"{name:>8}: {len(chunks):>4} chunks | avg {avg:5.1f} tokens | "
              f"start capital {starts:5.1f}% | end with . ! ? {ends:5.1f}%")

    print("\nFirst document, sentence chunker (size 64, overlap 16):")
    doc_id, text = docs[0]
    for c in SentenceChunker(tok, 64, 16).chunk(doc_id, text):
        print(f"{c.chunk_id:>10} tokens={c.n_tokens:>3}  {c.text[:45]!r} ... {c.text[-25:]!r}")


if __name__ == "__main__":
    main()