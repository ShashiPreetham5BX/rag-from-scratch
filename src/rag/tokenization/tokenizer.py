"""
Wrapper around a pretrained HuggingFace tokenizer.

WE ARE USING A LIBRARY HERE (transformers) BECAUSE a tokenizer must match
its model's vocabulary exactly. A pretrained model's embedding matrix has
one row per vocabulary entry; token ID 2043 indexes one specific row. A
tokenizer producing different IDs makes every lookup wrong — and it fails
SILENTLY, yielding meaningless embeddings rather than an exception.

Underneath, this is the BPE/WordPiece algorithm demonstrated in toy_bpe.py,
with a vocabulary learned once during the model's pretraining.
"""

from transformers import AutoTokenizer


class Tokenizer:
    """Token counting and encoding for a specific model.

    Args:
        model_name: HuggingFace model identifier. Must be the same model
                    used for embedding, or counts will be wrong.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._tok = AutoTokenizer.from_pretrained(model_name)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Text to token IDs."""
        return self._tok.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids: list[int]) -> str:
        """Token IDs back to text. Not always identical to the input —
        tokenization is lossy for whitespace and casing in some models."""
        return self._tok.decode(ids, skip_special_tokens=True)

    def tokens(self, text: str) -> list[str]:
        """Human-readable token strings. For inspection, not for the model."""
        return self._tok.tokenize(text)

    def count(self, text: str, add_special_tokens: bool = False) -> int:
        """Number of tokens. This is the unit chunk sizes are measured in."""
        return len(self.encode(text, add_special_tokens=add_special_tokens))
    
    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """Token IDs plus the (char_start, char_end) of each token in `text`.

        The offsets let a chunker cut the ORIGINAL text at token boundaries.
        """
        enc = self._tok(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        return enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]]
    @property
    def max_length(self) -> int:
        """Maximum sequence length the model accepts, INCLUDING special tokens."""
        return self._tok.model_max_length

    @property
    def content_budget(self) -> int:
        """Tokens available for actual content.

        Special tokens are added at embedding time: [CLS] at the start and
        [SEP] at the end for BERT-family models. Those consume 2 of the
        max_length budget. A chunk sized exactly at max_length will be
        SILENTLY TRUNCATED by 2 tokens when embedded.
        """
        return self.max_length - self._tok.num_special_tokens_to_add()

    def __repr__(self) -> str:
        return (f"Tokenizer(model={self.model_name!r}, "
                f"vocab={self._tok.vocab_size:,}, max_length={self.max_length})")