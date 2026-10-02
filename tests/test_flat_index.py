import numpy as np
import pytest

from rag.indexing.flat import FlatIndex


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def test_known_ranking_and_cosine_score():
    idx = FlatIndex(2)
    idx.add(unit([[1, 0], [0, 1], [1, 1]]), ["a", "b", "c"])
    q = unit([1, 0.1])
    res = idx.search(q, k=3)
    assert [i for i, _ in res] == ["a", "c", "b"]
    assert np.isclose(res[0][1], float(q @ unit([1, 0])), atol=1e-5)


def test_k_larger_than_index_returns_everything():
    idx = FlatIndex(2)
    idx.add(unit([[1, 0], [0, 1], [1, 1]]), ["a", "b", "c"])
    assert len(idx.search(unit([1, 0]), k=10)) == 3


def test_empty_index_returns_nothing():
    assert FlatIndex(4).search(np.ones(4), k=5) == []


def test_two_adds_equal_one_add():
    v = unit(np.random.default_rng(1).normal(size=(6, 4)))
    one, two = FlatIndex(4), FlatIndex(4)
    one.add(v, list("abcdef"))
    two.add(v[:3], list("abc"))
    two.add(v[3:], list("def"))
    q = unit(np.ones(4))
    assert one.search(q, k=6) == two.search(q, k=6)


def test_matches_full_sort_on_random_data():
    rng = np.random.default_rng(0)
    V = unit(rng.normal(size=(500, 16)))
    Q = unit(rng.normal(size=(20, 16)))
    idx = FlatIndex(16)
    idx.add(V, [str(i) for i in range(500)])
    results = idx.search(Q, k=10, block=7)       # block < n_queries exercises batching
    for q, res in zip(Q, results):
        expected = np.argsort(-(V @ q))[:10]
        assert [i for i, _ in res] == [str(i) for i in expected]


def test_single_query_matches_batch():
    idx = FlatIndex(2)
    idx.add(unit([[1, 0], [0, 1]]), ["a", "b"])
    single = idx.search(unit([1, 0]), k=1)
    batch = idx.search(unit([[1, 0], [0, 1]]), k=1)
    assert single == batch[0]
    assert [r[0][0] for r in batch] == ["a", "b"]


def test_wrong_dimension_raises():
    idx = FlatIndex(3)
    with pytest.raises(ValueError):
        idx.add(np.ones((2, 4)), ["a", "b"])
    with pytest.raises(ValueError):
        idx.search(np.ones(5), k=1)


def test_id_count_mismatch_raises():
    with pytest.raises(ValueError):
        FlatIndex(2).add(np.ones((2, 2)), ["only-one"])