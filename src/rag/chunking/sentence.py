"""Sentence-aware chunking: pack whole sentences into chunks up to a token budget."""

import re
from bisect import bisect_left

from rag.chunking.chunk import Chunk

# Boundary = . ! ? then whitespace then a capital letter, OR a blank line.
_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|\n\s*\n")


def split_sentences(text: str) -> list[tuple[int, int]]:
    """(start, end) character spans of sentences, without surrounding whitespace."""
    spans, start = [], 0
    for m in _BOUNDARY.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))

    trimmed = []
    for s, e in spans:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            trimmed.append((s, e))
    return trimmed


class SentenceChunker:
    """Pack whole sentences into chunks of at most `chunk_size` tokens.

    `overlap` is a MAXIMUM, measured in whole sentences: the next chunk repeats
    trailing sentences only while their total stays <= overlap tokens.
    """

    def __init__(self, tokenizer, chunk_size: int = 256, overlap: int = 32) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc_id: str, text: str) -> list[Chunk]:
        ids, offsets = self.tokenizer.encode_with_offsets(text)
        if not ids:
            return []
        starts = [o[0] for o in offsets]

        # 1. Sentences -> token ranges [ta, tb). Oversized sentences are hard-split.
        units: list[tuple[int, int]] = []
        for s, e in split_sentences(text):
            ta, tb = bisect_left(starts, s), bisect_left(starts, e)
            while tb - ta > self.chunk_size:
                units.append((ta, ta + self.chunk_size))
                ta += self.chunk_size
            if tb > ta:
                units.append((ta, tb))

        size = lambda u: u[1] - u[0]

        # 2. Greedy packing with sentence-level overlap.
        chunks: list[Chunk] = []
        i, n = 0, len(units)
        while i < n:
            j, used = i, 0
            while j < n and used + size(units[j]) <= self.chunk_size:
                used += size(units[j])
                j += 1
            chunks.append(self._make(doc_id, text, offsets, len(chunks),
                                     units[i][0], units[j - 1][1]))
            if j == n:
                break

            # Where does the next chunk start? Carry back whole sentences that fit
            # in `overlap`, but only if unit j still fits afterwards (guarantees progress).
            nxt = size(units[j])
            k, carried = j, 0
            while k - 1 > i:
                s = size(units[k - 1])
                if carried + s > self.overlap or carried + s + nxt > self.chunk_size:
                    break
                carried += s
                k -= 1
            i = k
        return chunks

    @staticmethod
    def _make(doc_id, text, offsets, index, ta, tb) -> Chunk:
        cs, ce = offsets[ta][0], offsets[tb - 1][1]
        return Chunk(
            chunk_id=f"{doc_id}::{index}",
            doc_id=str(doc_id),
            text=text[cs:ce],
            char_start=cs,
            char_end=ce,
            n_tokens=tb - ta,
        )