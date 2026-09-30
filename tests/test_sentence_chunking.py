import re

import pytest

from rag.chunking.sentence import SentenceChunker, split_sentences


class WhitespaceTokenizer:
    """Fake tokenizer: one token per whitespace-separated word."""

    def encode_with_offsets(self, text):
        matches = list(re.finditer(r"\S+", text))
        return list(range(len(matches))), [m.span() for m in matches]


def make(size, overlap):
    return SentenceChunker(WhitespaceTokenizer(), chunk_size=size, overlap=overlap)


# Sentences have 3, 4, 2 and 5 tokens.
TEXT = "A b c. D e f g. H i. J k l m n."
LONG = ("Alpha beta gamma. Delta epsilon zeta eta theta. Iota kappa. "
        "Lambda mu nu xi omicron pi rho sigma. Tau upsilon phi chi psi omega.")


def test_split_sentences_basic():
    text = "First one. Second one! Third? fourth stays."
    assert [text[s:e] for s, e in split_sentences(text)] == [
        "First one.", "Second one!", "Third? fourth stays."]


def test_decimals_are_not_split():
    text = "The ratio was 1.4 versus 0.24 overall. Next sentence here."
    assert [text[s:e] for s, e in split_sentences(text)] == [
        "The ratio was 1.4 versus 0.24 overall.", "Next sentence here."]


def test_packs_whole_sentences():
    chunks = make(7, 0).chunk("d", TEXT)
    assert [c.text for c in chunks] == ["A b c. D e f g.", "H i. J k l m n."]


def test_overlap_carries_last_sentence():
    chunks = make(9, 4).chunk("d", TEXT)
    assert [c.text for c in chunks] == ["A b c. D e f g. H i.", "H i. J k l m n."]


def test_oversized_sentence_is_hard_split():
    chunks = make(3, 0).chunk("d", "a b c d e f g h")
    assert [c.text for c in chunks] == ["a b c", "d e f", "g h"]


@pytest.mark.parametrize("size", [3, 5, 8, 20])
@pytest.mark.parametrize("overlap", [0, 2])
def test_invariants(size, overlap):
    chunks = make(size, overlap).chunk("d", LONG)
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(LONG)
    for c in chunks:
        assert c.n_tokens <= size
        assert LONG[c.char_start:c.char_end] == c.text
        assert len(c.text.split()) == c.n_tokens
    for a, b in zip(chunks, chunks[1:]):
        assert b.char_start > a.char_start                    # always advances
        assert LONG[a.char_end:b.char_start].strip() == ""    # no text skipped


def test_empty_text():
    assert make(5, 1).chunk("d", "") == []


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        make(4, 4)