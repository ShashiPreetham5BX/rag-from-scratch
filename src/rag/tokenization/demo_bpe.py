"""
Train the toy BPE on SciFact and observe what it learns.

    python src/rag/tokenization/demo_bpe.py
"""

from rag.loading.jsonl_loader import JSONLLoader
from rag.preprocessing.pipelines import get_pipeline
from rag.tokenization.toy_bpe import ToyBPE


def main() -> None:
    docs = JSONLLoader("data/raw/scifact/corpus.jsonl", source_name="scifact").load()
    pipeline = get_pipeline("standard")

    # 500 documents is plenty to see the pattern and keeps training fast.
    corpus = [pipeline.clean(d.full_text) for d in docs[:500]]

    print(f"\nTraining BPE on {len(corpus)} documents...\n")
    bpe = ToyBPE(num_merges=300)
    bpe.train(corpus, verbose=True)

    print(f"\n{bpe}\n")

    print("--- Segmentation of sample words ---")
    for word in [
        "the", "cell", "cells", "cellular",
        "protein", "proteins",
        "immunohistochemistry",
        "zzzqx",
    ]:
        pieces = bpe.tokenize(word)
        print(f"  {word:24s} -> {pieces}")


if __name__ == "__main__":
    main()