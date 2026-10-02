import numpy as np
from sentence_transformers import SentenceTransformer

from rag.embeddings.embedder import DEFAULT_MODEL, Embedder

TEXTS = [
    "Vitamin D supplementation reduces the risk of respiratory infection.",
    "Cholecalciferol intake lowered rates of lung illness in the trial.",
    "The stock market closed higher on Tuesday after strong earnings.",
    "Alterations of the architecture of cerebral white matter can affect cortical development.",
    "word " * 400,   # about 400 tokens, so it is truncated at 256
]


def main() -> None:
    mine = Embedder()
    print(mine)
    ours = mine.embed(TEXTS)
    print("shape:", ours.shape, "| row norms:", np.round(np.linalg.norm(ours, axis=1), 4))

    # Semantic sanity check: paraphrase vs unrelated (dot product = cosine)
    sim = ours @ ours.T
    print(f"\nparaphrase pair (0,1): {sim[0, 1]:.3f}")
    print(f"unrelated pair  (0,2): {sim[0, 2]:.3f}")

    # Verification against the reference implementation
    ref = SentenceTransformer(DEFAULT_MODEL).encode(TEXTS, normalize_embeddings=True)
    print("\nreference max_seq_length:", SentenceTransformer(DEFAULT_MODEL).max_seq_length)
    cos = np.sum(ours * ref, axis=1)
    print("cosine(ours, reference) per text:", np.round(cos, 5))
    print("MATCH" if cos.min() > 0.999 else "MISMATCH")


if __name__ == "__main__":
    main()