"""
Loader for JSONL files, including the BEIR corpus format.

BEIR corpus entries have the shape:
    {"_id": "4983", "title": "...", "text": "...", "metadata": {}}
"""

import json
from pathlib import Path

from tqdm import tqdm

from rag.loading.base import BaseLoader
from rag.loading.document import Document


class JSONLLoader(BaseLoader):
    """Load documents from a JSON Lines file.

    Args:
        path:          Path to the .jsonl file.
        id_field:      Key holding the document ID.
        text_field:    Key holding the body text.
        title_field:   Key holding the title. Optional.
        source_name:   Label recorded on each Document.
        skip_empty:    Drop documents whose text is empty after stripping.
    """

    def __init__(
        self,
        path: str | Path,
        id_field: str = "_id",
        text_field: str = "text",
        title_field: str | None = "title",
        source_name: str | None = None,
        skip_empty: bool = True,
    ) -> None:
        self.path = Path(path)
        self.id_field = id_field
        self.text_field = text_field
        self.title_field = title_field
        self.source_name = source_name or self.path.stem
        self.skip_empty = skip_empty

    def load(self) -> list[Document]:
        if not self.path.exists():
            raise FileNotFoundError(
                f"No such file: {self.path}\n"
                f"Did you run download_scifact.py, and are you in the project root?"
            )

        documents: list[Document] = []
        malformed = 0
        empty = 0

        # encoding="utf-8" is explicit because Windows defaults to a legacy
        # encoding, which raises UnicodeDecodeError on any non-ASCII character.
        with open(self.path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(tqdm(f, desc=f"Loading {self.path.name}"), start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # One bad line should not abort a 5,000-document load.
                    malformed += 1
                    continue

                text = (record.get(self.text_field) or "").strip()
                if self.skip_empty and not text:
                    empty += 1
                    continue

                title = ""
                if self.title_field:
                    title = (record.get(self.title_field) or "").strip()

                # Everything not explicitly mapped is preserved in metadata,
                # so no information from the source file is silently discarded.
                extra = {
                    k: v for k, v in record.items()
                    if k not in {self.id_field, self.text_field, self.title_field}
                }

                documents.append(
                    Document(
                        doc_id=record.get(self.id_field, f"line-{line_number}"),
                        text=text,
                        title=title,
                        source=self.source_name,
                        metadata=extra,
                    )
                )

        if malformed:
            print(f"  warning: skipped {malformed} malformed line(s)")
        if empty:
            print(f"  warning: skipped {empty} document(s) with empty text")

        return documents

    def __repr__(self) -> str:
        return f"JSONLLoader(path={str(self.path)!r})"