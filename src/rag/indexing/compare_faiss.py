"""Compare our FlatIndex with FAISS IndexFlatIP: agreement and speed."""

import time

import faiss
import numpy as np

from rag.indexing.flat import FlatIndex

K = 10


def unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def bench(name: str, vectors: np.ndarray, queries: np.ndarray) -> None:
    n, dim = vectors.shape
    mine = FlatIndex(dim)
    mine.add(vectors, [str(i) for i in range(n)])
    ref = faiss.IndexFlatIP(dim)
    ref.add(vectors)

    t = time.perf_counter()
    mine_res = mine.search(queries, K)
    t_mine = time.perf_counter() - t
    t = time.perf_counter()
    _, ref_idx = ref.search(queries, K)
    t_ref = time.perf_counter() - t

    agree = np.mean([len({i for i, _ in r} & {str(j) for j in row}) / K
                     for r, row in zip(mine_res, ref_idx)])

    few = queries[:50]
    t = time.perf_counter()
    for q in few:
        mine.search(q, K)
    lat_mine = (time.perf_counter() - t) / len(few)
    t = time.perf_counter()
    for q in few:
        ref.search(q[None, :], K)
    lat_ref = (time.perf_counter() - t) / len(few)

    nq = len(queries)
    print(f"\n{name}: N={n:,}, dim={dim}, {nq} queries, k={K}")
    print(f"  top-{K} agreement with FAISS : {100 * agree:.1f}%")
    print(f"  batch  ms/query  mine={1000 * t_mine / nq:.3f}  faiss={1000 * t_ref / nq:.3f}")
    print(f"  single ms/query  mine={1000 * lat_mine:.3f}  faiss={1000 * lat_ref:.3f}")


def main() -> None:
    rng = np.random.default_rng(0)

    real = np.load("data/processed/scifact_sentence_254/embeddings.npy")
    picked = real[rng.choice(len(real), 500, replace=False)]
    noisy = unit(picked + 0.1 * rng.normal(size=picked.shape).astype(np.float32))
    bench("SciFact real vectors", real, noisy)

    big = unit(rng.normal(size=(100_000, 384)).astype(np.float32))
    bench("Synthetic scale-up", big, unit(rng.normal(size=(200, 384)).astype(np.float32)))


if __name__ == "__main__":
    main()