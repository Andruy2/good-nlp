import argparse
import time

import torch

from app.ml.generation import ChatGenerator, GenerationParams
from app.ml.loader import load_model, load_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="models/ckpt_sft.pt")
    parser.add_argument("--tokenizer", default="models/tokenizer.json")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    print(f"torch threads: {torch.get_num_threads()}")
    t0 = time.perf_counter()
    model = load_model(args.ckpt)
    tok = load_tokenizer(args.tokenizer)
    gen = ChatGenerator(model, tok)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Модель загружена за {time.perf_counter() - t0:.1f} c, параметров: {n_params / 1e6:.1f}M")

    params = GenerationParams(args.max_new_tokens, args.temperature, args.top_k)
    while True:
        try:
            question = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        t0 = time.perf_counter()
        res = gen.generate(question, params)
        dt = time.perf_counter() - t0
        print(f"Модель: {res.text}")
        print(f"[{res.new_tokens} ток. за {dt:.2f} c, {res.new_tokens / dt:.1f} ток/с]")


if __name__ == "__main__":
    main()