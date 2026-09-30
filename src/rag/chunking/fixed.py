"""Fixed-size token-window chunking with overlap."""

from rag.chunking.chunk import Chunk


class FixedSizeChunker:
    """Slide a window of `chunk_size` tokens, moving `chunk_size - overlap` each step.

    `tokenizer` must provide encode_with_offsets(text). Passing it in (rather
    than creating it here) lets tests use a tiny fake tokenizer.
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
        n = len(ids)
        if n == 0:
            return []

        step = self.chunk_size - self.overlap
        chunks: list[Chunk] = []
        start = 0
        while start < n:
            end = min(start + self.chunk_size, n)
            char_start = offsets[start][0]
            char_end = offsets[end - 1][1]
            chunks.append(Chunk(
                chunk_id=f"{doc_id}::{len(chunks)}",
                doc_id=str(doc_id),
                text=text[char_start:char_end],
                char_start=char_start,
                char_end=char_end,
                n_tokens=end - start,
            ))
            if end == n:      # reached the end; stop so we never emit a tail
                break         # chunk that is fully inside the previous one
            start += step
        return chunks