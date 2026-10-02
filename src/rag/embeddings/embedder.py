"""A hand-written embedder: tokenize -> transformer -> mean pool -> L2 normalize."""

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """Turns texts into unit-length vectors.

    WE USE A LIBRARY (transformers) FOR THE MODEL WEIGHTS because a trained
    transformer cannot be rebuilt by hand. Pooling and normalization, which
    decide what the vector means, are written explicitly below.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, max_length: int = 256,
                 batch_size: int = 32) -> None:
        self.model_name = model_name
        self.max_length = max_length          # real limit, INCLUDING [CLS] and [SEP]
        self.batch_size = batch_size
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).eval()
        self.dim = self._model.config.hidden_size

    @torch.no_grad()
    def embed(self, texts: list[str]) -> np.ndarray:
        """Returns an array of shape (len(texts), dim); every row has length 1."""
        batches = []
        for i in range(0, len(texts), self.batch_size):
            enc = self._tok(
                texts[i:i + self.batch_size],
                padding=True,                  # pad short texts so the batch is rectangular
                truncation=True,               # cut anything beyond max_length
                max_length=self.max_length,
                return_tensors="pt",
            )
            hidden = self._model(**enc).last_hidden_state          # (batch, tokens, dim)
            mask = enc["attention_mask"].unsqueeze(-1).float()     # (batch, tokens, 1): 1 = real token
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # mean over REAL tokens
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)             # length 1
            batches.append(pooled.numpy())
        if not batches:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack(batches)

    def __repr__(self) -> str:
        return f"Embedder(model={self.model_name!r}, dim={self.dim}, max_length={self.max_length})"