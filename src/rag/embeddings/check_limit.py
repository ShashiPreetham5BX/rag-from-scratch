"""Find the model's REAL maximum input length, not the tokenizer's claim."""

import json

from huggingface_hub import hf_hub_download
from transformers import AutoConfig

NAME = "sentence-transformers/all-MiniLM-L6-v2"

path = hf_hub_download(NAME, "sentence_bert_config.json")
with open(path, encoding="utf-8") as f:
    print("sentence_bert_config.json :", json.load(f))

cfg = AutoConfig.from_pretrained(NAME)
print("position embeddings (arch):", cfg.max_position_embeddings)
print("embedding size            :", cfg.hidden_size)