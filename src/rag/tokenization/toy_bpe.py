"""
A minimal Byte Pair Encoding implementation, written from scratch.

This is NOT used in the pipeline — the production path uses HuggingFace
tokenizers, which must match a model's exact vocabulary. This exists to
demonstrate the mechanism that subword tokenization rests on.

BPE algorithm (Sennrich et al., 2016, adapted from a data-compression
technique):
    1. Represent each word as a sequence of characters
    2. Count all adjacent symbol pairs, weighted by word frequency
    3. Merge the most frequent pair everywhere
    4. Repeat N times

The learned merge list IS the tokenizer. Applying merges in the order they
were learned reproduces the segmentation.
"""

from collections import Counter


class ToyBPE:
    """Byte Pair Encoding, trained on a text corpus.

    Args:
        num_merges: How many merge operations to learn. More merges means
                    a larger vocabulary and shorter token sequences.
    """

    # Marks the end of a word, so that "est" appearing word-finally
    # ("largest") is distinguishable from "est" appearing mid-word
    # ("estimate"). Without this, BPE would conflate them.
    END = "</w>"

    def __init__(self, num_merges: int = 200) -> None:
        self.num_merges = num_merges
        self.merges: list[tuple[str, str]] = []
        self.vocab: set[str] = set()

    @staticmethod
    def _word_frequencies(corpus: list[str]) -> Counter:
        counts: Counter = Counter()
        for text in corpus:
            for raw in text.lower().split():
                word = "".join(ch for ch in raw if ch.isalnum())
                if word:
                    counts[word] += 1
        return counts

    def _initial_splits(self, word_freqs: Counter) -> dict[tuple[str, ...], int]:
        """Each word becomes a tuple of single characters plus the end marker."""
        return {tuple(word) + (self.END,): freq for word, freq in word_freqs.items()}

    @staticmethod
    def _count_pairs(splits: dict[tuple[str, ...], int]) -> Counter:
        """Count adjacent symbol pairs, weighted by how often the word occurs."""
        pairs: Counter = Counter()
        for symbols, freq in splits.items():
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    @staticmethod
    def _apply_merge(pair: tuple[str, str], symbols: tuple[str, ...]) -> tuple[str, ...]:
        """Replace every occurrence of `pair` in `symbols` with the joined symbol."""
        merged: list[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                merged.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        return tuple(merged)

    def train(self, corpus: list[str], verbose: bool = False) -> None:
        word_freqs = self._word_frequencies(corpus)
        splits = self._initial_splits(word_freqs)

        for step in range(self.num_merges):
            pairs = self._count_pairs(splits)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            best_count = pairs[best_pair]

            splits = {
                self._apply_merge(best_pair, symbols): freq
                for symbols, freq in splits.items()
            }
            self.merges.append(best_pair)

            if verbose and step < 25:
                joined = best_pair[0] + best_pair[1]
                print(f"  merge {step + 1:3d}: {best_pair[0]!r} + {best_pair[1]!r} "
                      f"-> {joined!r}   (seen {best_count:,} times)")

        for symbols in splits:
            self.vocab.update(symbols)

    def tokenize(self, word: str) -> list[str]:
        """Segment a single word by replaying the learned merges in order."""
        symbols = tuple(word.lower()) + (self.END,)
        for pair in self.merges:
            symbols = self._apply_merge(pair, symbols)
        return list(symbols)

    def __repr__(self) -> str:
        return f"ToyBPE(merges={len(self.merges)}, vocab={len(self.vocab)})"