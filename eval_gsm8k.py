#!/usr/bin/env python3
"""GSM8K on MLX. Strict #### match + flexible last-number. Default: 200 random test rows, 4-shot."""

from __future__ import annotations

import argparse
import re

from datasets import load_dataset
from mlx_lm import generate, load

from model import MODEL, SYSTEM

STRICT = re.compile(r"####\\s*(-?[0-9][0-9,]*(?:\\.[0-9]+)?)")
LOOSE = re.compile(r"-?[0-9][0-9,]*(?:\\.[0-9]+)?")


def strict_num(text: str) -> str:
    found = STRICT.findall(text.replace(",", ""))
    return found[-1] if found else ""


def loose_num(text: str) -> str:
    s = strict_num(text)
    if s:
        return s
    found = LOOSE.findall(text.replace(",", ""))
    return found[-1] if found else ""


def gold_num(answer: str) -> str:
    return strict_num(answer) or loose_num(answer)


def shots(train, k: int) -> list[dict]:
    out = []
    for i, ex in enumerate(train):
        if i >= k:
            break
        out.append({"role": "user", "content": SYSTEM + "\\n\\n" + ex["question"]})
        out.append({"role": "assistant", "content": ex["answer"]})
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="", help="MLX adapter folder; empty = base")
    p.add_argument("--sample", type=int, default=200, help="0 = full 1319")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shots", type=int, default=4)
    p.add_argument("--show", type=int, default=2, help="print first N completions")
    args = p.parse_args()

    adapter = args.adapter or None
    model, tok = load(MODEL, adapter_path=adapter)
    raw = load_dataset("openai/gsm8k", "main")
    prefix = shots(raw["train"], args.shots)
    test = raw["test"]
    if args.sample and args.sample < len(test):
        test = test.shuffle(seed=args.seed).select(range(args.sample))

    ok_s = ok_l = 0
    n = len(test)
    print(f"GSM8K n={n} shots={args.shots} mlx {MODEL} adapter={adapter or 'base'}")
    for i, ex in enumerate(test):
        messages = [
            *prefix,
            {"role": "user", "content": SYSTEM + "\\n\\n" + ex["question"]},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text = generate(model, tok, prompt=prompt, max_tokens=320, verbose=False)
        gold = gold_num(ex["answer"])
        s, l = strict_num(text), loose_num(text)
        ok_s += int(s == gold and gold != "")
        ok_l += int(l == gold and gold != "")
        if i < args.show:
            print("--- sample", i)
            print(text[-500:])
            print("pred_strict", s, "pred_flex", l, "gold", gold)
        if (i + 1) % 10 == 0 or i + 1 == n:
            print(
                f"{i + 1}/{n}  strict {100 * ok_s / (i + 1):.1f}%  flex {100 * ok_l / (i + 1):.1f}%"
            )
    print(
        f"GSM8K strict ####: {ok_s}/{n} = {100 * ok_s / n:.2f}%  "
        f"flexible: {ok_l}/{n} = {100 * ok_l / n:.2f}%  model={MODEL}"
    )


if __name__ == "__main__":
    main()
