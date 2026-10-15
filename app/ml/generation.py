from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from app.ml.model import GPT


@dataclass
class GenerationParams:
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_k: int = 50


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    new_tokens: int


class ChatGenerator:
    def __init__(self, model: GPT, tokenizer: Tokenizer, device: str = "cpu"):
        self.model = model
        self.tok = tokenizer
        self.device = device
        self.block_size = model.cfg.block_size
        self.eot_id = tokenizer.token_to_id("<|endoftext|>")
        if self.eot_id is None:
            raise ValueError("В токенизаторе нет токена <|endoftext|>")

    @staticmethod
    def build_prompt(instruction: str, inp: str = "") -> str:
        text = "Пользователь: " + instruction.strip()
        if inp and inp.strip():
            text += "\n" + inp.strip()
        return text + "\nАссистент:"

    @torch.inference_mode()
    def generate(self, question: str, params: GenerationParams | None = None) -> GenerationResult:
        p = params or GenerationParams()
        prompt_ids = self.tok.encode(self.build_prompt(question)).ids
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        start = idx.shape[1]

        for _ in range(p.max_new_tokens):
            logits, _ = self.model(idx[:, -self.block_size:])
            logits = logits[:, -1, :]

            if p.temperature <= 0:  # жадный режим: всегда самый вероятный токен
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / p.temperature
                k = min(p.top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits[logits < v[:, [-1]]] = -float("inf")
                next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

            if next_id.item() == self.eot_id:
                break
            idx = torch.cat([idx, next_id], dim=1)

        new_ids = idx[0, start:].tolist()
        return GenerationResult(
            text=self.tok.decode(new_ids).strip(),
            prompt_tokens=start,
            new_tokens=len(new_ids),
        )