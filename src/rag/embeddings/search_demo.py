"""Brute-force search over saved vectors, shown for a few SciFact test queries."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from rag.embeddings.embedder import Embedder


def load_qrels(path: str) -> dict[str, set[str]]:
    rel = defaultdict(set)
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)                                   # header row
        for qid, did, score in reader:
            if int(score) > 0:
                rel[str(qid)].add(str(did))
    return rel


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="data/processed/scifact_sentence_254")
    p.add_argument("--n", type=int, default=5, help="number of queries to show")
    args = p.parse_args()
    run = Path(args.dir)

    emb = np.load(run / "embeddings.npy")
    chunks = [json.loads(line) for line in open(run / "chunks.jsonl", encoding="utf-8")]
    qrels = load_qrels("data/raw/scifact/qrels/test.tsv")
    queries = [json.loads(line) for line in open("data/raw/scifact/queries.jsonl", encoding="utf-8")]
    queries = [q for q in queries if str(q["_id"]) in qrels][:args.n]

    embedder = Embedder()
    for q in queries:
        qid = str(q["_id"])
        scores = emb @ embedder.embed([q["text"]])[0]       # cosine, since all vectors have length 1
        top_docs, seen = [], set()
        for i in np.argsort(-scores):                       # best chunk first
            doc_id = chunks[i]["doc_id"]
            if doc_id not in seen:                          # a document's score = its best chunk
                seen.add(doc_id)
                top_docs.append((doc_id, float(scores[i]), chunks[i]["text"]))
            if len(top_docs) == 5:
                break

        print(f"\nQUERY {qid}: {q['text']}")
        print(f"  relevant docs (qrels): {sorted(qrels[qid])}")
        for rank, (doc_id, score, text) in enumerate(top_docs, 1):
            mark = "RELEVANT" if doc_id in qrels[qid] else "        "
            print(f"  {rank}. {mark} doc {doc_id:>9}  score={score:.3f}  {text[:70]!r}")


if __name__ == "__main__":
    main()