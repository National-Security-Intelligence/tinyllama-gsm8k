#!/usr/bin/env python3
"""GSM8K exact match on #### via MLX. Default: random 200 test rows."""

from __future__ import annotations

import argparse
import re

from datasets import load_dataset
from mlx_lm import generate, load

from model import MODEL, SYSTEM

ANS = re.compile(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)")


def extract(text: str) -> str:
    found = ANS.findall(text.replace(",", ""))
    return found[-1] if found else ""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="", help="MLX adapter folder; empty = base")
    p.add_argument("--sample", type=int, default=200, help="0 = full GSM8K test (1319)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    adapter = args.adapter or None
    model, tok = load(MODEL, adapter_path=adapter)
    test = load_dataset("openai/gsm8k", "main", split="test")
    if args.sample and args.sample < len(test):
        test = test.shuffle(seed=args.seed).select(range(args.sample))

    ok = 0
    n = len(test)
    print(f"GSM8K n={n} mlx adapter={adapter or 'base'} seed={args.seed}")
    for i, ex in enumerate(test):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": ex["question"]},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text = generate(model, tok, prompt=prompt, max_tokens=256, verbose=False)
        pred, gold = extract(text), extract(ex["answer"])
        ok += int(pred == gold and pred != "")
        if (i + 1) % 10 == 0 or i + 1 == n:
            print(f"{i + 1}/{n}  {100 * ok / (i + 1):.1f}%")
    print(f"GSM8K exact match: {ok}/{n} = {100 * ok / n:.2f}%  base={MODEL}")


if __name__ == "__main__":
    main()
