"""A hand-written exact (brute-force) vector index."""

import numpy as np


class FlatIndex:
    """Stores unit-length vectors and returns the top-k by dot product (= cosine).

    Exact: every stored vector is scored for every query.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._ids: list[str] = []

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, vectors, ids) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected vectors of shape (n, {self.dim}), got {vectors.shape}")
        if len(ids) != len(vectors):
            raise ValueError(f"{len(vectors)} vectors but {len(ids)} ids")
        self._vectors = np.vstack([self._vectors, vectors])
        self._ids.extend(str(i) for i in ids)

    def search(self, queries, k: int = 10, block: int = 128):
        """One query (shape (dim,)) -> list of (id, score). Many (n, dim) -> list of such lists."""
        q = np.asarray(queries, dtype=np.float32)
        single = q.ndim == 1
        if single:
            q = q[None, :]
        if q.shape[1] != self.dim:
            raise ValueError(f"expected queries of dimension {self.dim}, got {q.shape[1]}")

        k = min(k, len(self))
        if k == 0:
            empty = [[] for _ in q]
            return empty[0] if single else empty

        results = []
        for s in range(0, len(q), block):            # blocks bound memory use
            scores = q[s:s + block] @ self._vectors.T                # (block, N)
            top = np.argpartition(-scores, k - 1, axis=1)[:, :k]    # k best, unordered
            top_scores = np.take_along_axis(scores, top, axis=1)
            order = np.argsort(-top_scores, axis=1)                 # sort only those k
            for r in range(len(top)):
                idxs = top[r][order[r]]
                results.append([(self._ids[i], float(scores[r, i])) for i in idxs])
        return results[0] if single else results