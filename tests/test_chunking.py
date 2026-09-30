import re

import pytest

from rag.chunking.fixed import FixedSizeChunker


class WhitespaceTokenizer:
    """Fake tokenizer: one token per word. Keeps tests fast and offline."""

    def encode_with_offsets(self, text):
        matches = list(re.finditer(r"\S+", text))
        return list(range(len(matches))), [m.span() for m in matches]


TEXT = " ".join(f"w{i}" for i in range(10))  # "w0 w1 ... w9"


def make(size, overlap):
    return FixedSizeChunker(WhitespaceTokenizer(), chunk_size=size, overlap=overlap)


def test_window_and_overlap():
    chunks = make(4, 1).chunk("d", TEXT)  # step = 3
    assert [c.text for c in chunks] == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]


def test_offsets_slice_original_text():
    for c in make(4, 1).chunk("d", TEXT):
        assert TEXT[c.char_start:c.char_end] == c.text


def test_short_text_is_one_chunk():
    chunks = make(100, 10).chunk("d", TEXT)
    assert len(chunks) == 1 and chunks[0].text == TEXT


def test_empty_text_gives_no_chunks():
    assert make(4, 1).chunk("d", "") == []


def test_whole_text_is_covered():
    chunks = make(5, 2).chunk("d", TEXT)
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(TEXT)


def test_chunk_ids_unique_and_sequential():
    ids = [c.chunk_id for c in make(4, 1).chunk("d", TEXT)]
    assert ids == ["d::0", "d::1", "d::2"]


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        make(4, 4)