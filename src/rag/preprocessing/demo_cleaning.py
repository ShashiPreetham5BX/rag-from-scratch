"""
Demonstrate cleaning on the real extracted PDF text.

Run from the project root:
    python src/rag/preprocessing/demo_cleaning.py
"""

from rag.loading.pdf_loader import PDFLoader
from rag.preprocessing.pipelines import get_pipeline


def main() -> None:
    docs = PDFLoader("data/raw/pdfs").load()
    if not docs:
        print("No PDFs found in data/raw/pdfs")
        return

    raw = docs[3].text[:900]

    print("=" * 70)
    print("BEFORE")
    print("=" * 70)
    print(repr(raw))

    pipeline = get_pipeline("standard")
    cleaned, history = pipeline.clean_verbose(raw)

    print()
    print("=" * 70)
    print("AFTER (standard pipeline)")
    print("=" * 70)
    print(repr(cleaned))

    print()
    print("=" * 70)
    print("PER-STEP CHARACTER COUNTS")
    print("=" * 70)
    for name, before, after in history:
        delta = after - before
        print(f"{name:24s} {before:5d} -> {after:5d}  ({delta:+d})")

    print()
    print("=" * 70)
    print("LIGATURE CHECK")
    print("=" * 70)
    for lig in ["\ufb01", "\ufb02", "\ufb00"]:
        n = unicodedata_name(lig)
        print(f"  {n}: raw={raw.count(lig)}  cleaned={cleaned.count(lig)}")
    print(f"  'find' appears in cleaned: {'find' in cleaned}")


def unicodedata_name(ch: str) -> str:
    import unicodedata
    try:
        return unicodedata.name(ch)
    except ValueError:
        return repr(ch)


if __name__ == "__main__":
    main()