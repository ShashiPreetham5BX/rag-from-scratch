"""
Download and extract the SciFact dataset from the BEIR benchmark.

SciFact consists of ~5,180 biomedical abstracts and ~1,100 scientific claims
with human relevance judgments. We use it as the evaluation corpus because it
is small enough to re-embed repeatedly on CPU while providing real qrels.

Run from the project root:
    python src/rag/loading/download_scifact.py
"""

import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# BEIR hosts its datasets at this location.
SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"

# Resolve paths relative to this file, not the current working directory.
# This makes the script work no matter where it is called from.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / "scifact.zip"
EXTRACT_DIR = RAW_DIR


def download_file(url: str, destination: Path) -> None:
    """Download a file, showing a progress bar.

    Streams the response in chunks rather than loading it entirely into
    memory, which matters for large files.
    """
    if destination.exists():
        print(f"Already downloaded: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading from {url}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()  # raises an exception on HTTP 4xx/5xx

    total_bytes = int(response.headers.get("content-length", 0))

    with open(destination, "wb") as f, tqdm(
        total=total_bytes, unit="B", unit_scale=True, desc=destination.name
    ) as progress:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            progress.update(len(chunk))

    print(f"Saved to {destination}")


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a zip archive."""
    print(f"Extracting {zip_path.name}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_to)
    print(f"Extracted to {extract_to}")


def main() -> None:
    download_file(SCIFACT_URL, ZIP_PATH)
    extract_zip(ZIP_PATH, EXTRACT_DIR)

    scifact_dir = EXTRACT_DIR / "scifact"
    if scifact_dir.exists():
        print("\nFiles found:")
        for path in sorted(scifact_dir.rglob("*")):
            if path.is_file():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"  {path.relative_to(scifact_dir)}  ({size_mb:.2f} MB)")
    else:
        print("Warning: expected folder 'scifact' was not created.")


if __name__ == "__main__":
    main()