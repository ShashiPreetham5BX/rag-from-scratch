import json

from rag.chunking.fixed import FixedSizeChunker
from rag.tokenization.tokenizer import Tokenizer


def main() -> None:
    tok = Tokenizer()
    print(tok)

    with open("data/raw/scifact/corpus.jsonl", encoding="utf-8") as f:
        row = json.loads(f.readline())
    text = row["text"]
    print(f"doc {row['_id']}: {len(text)} chars, {tok.count(text)} tokens\n")

    chunker = FixedSizeChunker(tok, chunk_size=64, overlap=16)
    for c in chunker.chunk(str(row["_id"]), text):
        print(f"{c.chunk_id:>10}  tokens={c.n_tokens:>3}  chars {c.char_start:>4}-{c.char_end:<4}  {c.text[:50]!r}...")


if __name__ == "__main__":
    main()