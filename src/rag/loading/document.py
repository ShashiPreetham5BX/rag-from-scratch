"""
The Document data model.

Every loader in this project produces Document objects. Downstream stages
(cleaning, chunking, embedding) consume Documents without knowing whether
they originated from JSONL, PDF, or plain text. This decoupling is what
allows the corpus to be swapped without changing the pipeline.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A single document before chunking.

    Attributes:
        doc_id:   Unique identifier. ALWAYS a string, never an int — see note below.
        text:     The main body text.
        title:    Optional title or heading.
        source:   Where this came from (file path, dataset name).
        metadata: Anything else worth carrying along.
    """

    doc_id: str
    text: str
    title: str = ""
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # IDs are coerced to str on construction.
        #
        # SciFact query IDs ("1", "3", "5") and corpus IDs ("31715818") live in
        # SEPARATE ID SPACES. Mixing int and str IDs means a lookup like
        # qrels["1"] silently misses when the key is stored as 1. That failure
        # is silent — it shows up as near-zero recall, not as an exception —
        # so we eliminate it structurally by never using ints.
        self.doc_id = str(self.doc_id)

    @property
    def full_text(self) -> str:
        """Title and body combined.

        Whether to include the title in the embedded text is an EXPERIMENTAL
        VARIABLE, not a fixed choice. Titles are dense with topical signal, so
        including them often improves retrieval — but they also skew the
        embedding away from the body content. We will measure this in Phase 15.
        """
        if self.title:
            return f"{self.title}\n\n{self.text}"
        return self.text

    def __len__(self) -> int:
        """Character count of the body text."""
        return len(self.text)

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        suffix = "..." if len(self.text) > 60 else ""
        return f"Document(id={self.doc_id!r}, len={len(self.text)}, text={preview!r}{suffix})"