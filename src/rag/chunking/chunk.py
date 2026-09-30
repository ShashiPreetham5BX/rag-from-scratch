"""The Chunk data model: the unit of embedding, retrieval and citation."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    chunk_id: str      # "<doc_id>::<index>", e.g. "4983::2"
    doc_id: str        # which document this came from
    text: str          # exact substring of the cleaned document text
    char_start: int    # where the chunk starts in the document text
    char_end: int      # where it ends (exclusive)
    n_tokens: int      # chunk length in tokens
    metadata: dict[str, Any] = field(default_factory=dict)