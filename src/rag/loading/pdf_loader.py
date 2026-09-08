"""
PDF document loader.

WE ARE USING A LIBRARY HERE (PyMuPDF) BECAUSE PDF is a page-description
format, not a text format. A PDF stores glyphs at coordinates, with no
inherent notion of words, lines, paragraphs, or reading order. Reconstructing
text requires resolving font encodings to Unicode, clustering glyphs into
words by spatial proximity, grouping words into lines and blocks, and
inferring a reading order from layout. PyMuPDF wraps the mature MuPDF engine,
which does all of this.

It is not perfect. Multi-column layouts, tables, and running headers are
frequent failure modes. We therefore extract page by page and retain page
numbers so that any bad extraction can be traced back to its source.
"""

from pathlib import Path

import pymupdf  
from tqdm import tqdm

from rag.loading.base import BaseLoader
from rag.loading.document import Document


class PDFLoader(BaseLoader):
    """Load text from one PDF or a directory of PDFs.

    Args:
        path:            A .pdf file, or a directory containing PDFs.
        per_page:        If True, emit one Document per page (recommended —
                         preserves page provenance). If False, one per file.
        min_chars:       Skip pages/documents shorter than this. Filters out
                         blank pages and image-only pages.
        recursive:       Search subdirectories when path is a directory.
    """

    def __init__(
        self,
        path: str | Path,
        per_page: bool = True,
        min_chars: int = 50,
        recursive: bool = True,
    ) -> None:
        self.path = Path(path)
        self.per_page = per_page
        self.min_chars = min_chars
        self.recursive = recursive

    def _pdf_paths(self) -> list[Path]:
        if self.path.is_file():
            return [self.path]
        if not self.path.exists():
            raise FileNotFoundError(f"No such path: {self.path}")
        pattern = "**/*.pdf" if self.recursive else "*.pdf"
        return sorted(self.path.glob(pattern))

    def _load_one(self, pdf_path: Path) -> list[Document]:
        docs: list[Document] = []

        with pymupdf.open(pdf_path) as pdf:
            # PDF metadata is often absent or wrong, so we default carefully.
            pdf_title = (pdf.metadata or {}).get("title") or pdf_path.stem
            page_texts: list[str] = []

            for page_number, page in enumerate(pdf, start=1):
                # "text" mode returns plain reading-order text. PyMuPDF also
                # offers "blocks" (with coordinates) and "dict" (full layout
                # detail) — useful later if we need to handle columns properly.
                text = page.get_text("text").strip()

                if self.per_page:
                    if len(text) < self.min_chars:
                        continue
                    docs.append(
                        Document(
                            doc_id=f"{pdf_path.stem}-p{page_number}",
                            text=text,
                            title=pdf_title,
                            source=str(pdf_path),
                            metadata={
                                "page": page_number,
                                "total_pages": pdf.page_count,
                                "file_name": pdf_path.name,
                            },
                        )
                    )
                else:
                    page_texts.append(text)

            if not self.per_page:
                combined = "\n\n".join(t for t in page_texts if t).strip()
                if len(combined) >= self.min_chars:
                    docs.append(
                        Document(
                            doc_id=pdf_path.stem,
                            text=combined,
                            title=pdf_title,
                            source=str(pdf_path),
                            metadata={
                                "total_pages": pdf.page_count,
                                "file_name": pdf_path.name,
                            },
                        )
                    )

        return docs

    def load(self) -> list[Document]:
        paths = self._pdf_paths()
        if not paths:
            print(f"warning: no PDFs found under {self.path}")
            return []

        documents: list[Document] = []
        failed: list[tuple[str, str]] = []

        for pdf_path in tqdm(paths, desc="Loading PDFs"):
            try:
                documents.extend(self._load_one(pdf_path))
            except Exception as exc:
                # One corrupt or encrypted PDF must not abort a batch load.
                failed.append((pdf_path.name, type(exc).__name__))

        if failed:
            print(f"  warning: {len(failed)} file(s) failed:")
            for name, err in failed[:5]:
                print(f"    {name}: {err}")

        return documents

    def __repr__(self) -> str:
        return f"PDFLoader(path={str(self.path)!r}, per_page={self.per_page})"