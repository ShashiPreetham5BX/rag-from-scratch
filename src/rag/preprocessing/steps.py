"""
Individual text cleaning steps.

WE ARE USING LIBRARIES HERE (unicodedata, re) BECAUSE both are Python
standard-library language features, not RAG abstractions. unicodedata
implements the Unicode Consortium's normalization algorithms, which are
lookup tables defined by specification — reimplementing them would mean
transcribing the Unicode Character Database. re is a regular expression
engine, a general text-processing primitive.
"""

import re
import unicodedata

from rag.preprocessing.base import CleaningStep


class UnicodeNormalize(CleaningStep):
    """Apply Unicode normalization.

    The same visible character can have multiple byte representations.
    'e' + combining acute renders identically to precomposed 'é' but
    compares unequal. Normalization maps them to one canonical form.

    Forms:
        NFC  - compose (canonical)
        NFD  - decompose (canonical)
        NFKC - compose + compatibility folding   <- our default
        NFKD - decompose + compatibility folding

    The K (compatibility) folding is what fixes ligatures: 'ﬁ' (U+FB01,
    a single character) becomes 'f' + 'i'. Without it, searching for
    "find" will not match a PDF containing "ﬁnd".

    LOSSY: NFKC also maps superscripts to plain digits, so 'x²' becomes
    'x2'. Acceptable for prose retrieval; questionable for mathematical
    or chemical corpora. This is a documented trade-off, not an oversight.
    """

    def __init__(self, form: str = "NFKC") -> None:
        if form not in {"NFC", "NFD", "NFKC", "NFKD"}:
            raise ValueError(f"Invalid normalization form: {form}")
        self.form = form

    def apply(self, text: str) -> str:
        return unicodedata.normalize(self.form, text)

    def __repr__(self) -> str:
        return f"UnicodeNormalize(form={self.form!r})"


class NormalizeWhitespace(CleaningStep):
    """Collapse whitespace variants into ordinary spaces.

    NFKC does NOT handle these. The non-breaking space (U+00A0) is
    extremely common in PDF extraction and is not a regular space —
    a regex matching ' ' will miss it.

    Also handles zero-width characters, which are invisible, carry no
    meaning, and silently corrupt token boundaries.
    """

    # Non-breaking space, en/em spaces, thin space, narrow NBSP, ideographic space
    _SPACE_LIKE = "\u00a0\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000"
    # Zero-width space, ZWNJ, ZWJ, BOM/zero-width no-break space, soft hyphen
    _ZERO_WIDTH = "\u200b\u200c\u200d\ufeff\u00ad"

    def __init__(self) -> None:
        self._space_re = re.compile(f"[{self._SPACE_LIKE}]")
        self._zero_re = re.compile(f"[{self._ZERO_WIDTH}]")

    def apply(self, text: str) -> str:
        text = self._zero_re.sub("", text)
        text = self._space_re.sub(" ", text)
        return text


class NormalizePunctuation(CleaningStep):
    """Map Unicode punctuation variants to ASCII equivalents.

    PDFs and typeset text use typographic quotes and dashes. The Unicode
    minus sign U+2212 (seen in the RAG paper as '−log') is not the ASCII
    hyphen. Left as-is, these produce distinct tokens for what a reader
    considers the same character.

    Optional by design: for some corpora the distinction carries meaning.
    """

    _MAP = {
        "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
        "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
        "\u2013": "-", "\u2014": "-", "\u2012": "-", "\u2212": "-",
        "\u2026": "...",
        "\u00b4": "'", "\u02bc": "'",
    }

    def __init__(self) -> None:
        self._table = str.maketrans(self._MAP)

    def apply(self, text: str) -> str:
        return text.translate(self._table)


class FixHyphenation(CleaningStep):
    """Rejoin words split by a hyphen at a line break.

    Typeset text breaks long words across lines: 'informa-\\ntion'.
    Left alone this yields two broken tokens instead of one real word.

    HEURISTIC, and it will sometimes be wrong. A genuine compound
    hyphenated word that happens to fall at a line break ('state-\\nof-the-art')
    gets incorrectly joined into 'stateof-the-art'. The pattern requires
    lowercase on both sides to limit the damage, but it is not exact.
    """

    _PATTERN = re.compile(r"([a-z])-\s*\n\s*([a-z])")

    def apply(self, text: str) -> str:
        return self._PATTERN.sub(r"\1\2", text)


class CollapseNewlines(CleaningStep):
    """Reduce runs of newlines, and optionally unwrap soft line breaks.

    In extracted PDF text most newlines are visual line wraps, not
    semantic breaks — 'costly as\\nit requires' is one sentence. But a
    blank line usually IS a real paragraph boundary.

    Convention adopted here:
        2+ newlines  -> "\\n\\n"  (paragraph break, preserved)
        1 newline    -> " "       (line wrap, unwrapped)  [if unwrap=True]
    """

    _MULTI = re.compile(r"\n{2,}")
    _SINGLE = re.compile(r"(?<!\n)\n(?!\n)")

    def __init__(self, unwrap_single: bool = True) -> None:
        self.unwrap_single = unwrap_single

    def apply(self, text: str) -> str:
        text = self._MULTI.sub("\x00PARA\x00", text)
        if self.unwrap_single:
            text = self._SINGLE.sub(" ", text)
        text = text.replace("\x00PARA\x00", "\n\n")
        return text

    def __repr__(self) -> str:
        return f"CollapseNewlines(unwrap_single={self.unwrap_single})"


class CollapseSpaces(CleaningStep):
    """Collapse runs of spaces/tabs and strip trailing space on each line."""

    _RUNS = re.compile(r"[ \t]+")
    _TRAILING = re.compile(r"[ \t]+\n")

    def apply(self, text: str) -> str:
        text = self._RUNS.sub(" ", text)
        text = self._TRAILING.sub("\n", text)
        return text.strip()


class RemoveControlChars(CleaningStep):
    """Strip non-printing control characters.

    Keeps \\n and \\t. Removes the rest, which can appear in malformed
    PDFs and break downstream tokenizers.
    """

    def apply(self, text: str) -> str:
        return "".join(
            ch for ch in text
            if ch in "\n\t" or unicodedata.category(ch)[0] != "C"
        )