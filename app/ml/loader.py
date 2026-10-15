from pathlib import Path
from types import SimpleNamespace

import torch
from tokenizers import Tokenizer

from app.ml.model import GPT


def load_model(ckpt_path: str | Path, device: str = "cpu") -> GPT:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = SimpleNamespace(**ckpt["config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])  # fp16-веса тут сами приведутся к fp32
    model.to(device)
    model.eval()
    return model


def load_tokenizer(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))