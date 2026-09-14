#!/usr/bin/env python3
"""GSM8K exact match on ####. Default: random 200 test rows. --sample 0 = full 1319."""

from __future__ import annotations

import argparse
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from device import dtype, kind
from model import MODEL

ANS = re.compile(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)")
SYSTEM = "Solve grade-school math. Show steps. End with #### <number>."


def extract(text: str) -> str:
    found = ANS.findall(text.replace(",", ""))
    return found[-1] if found else ""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="", help="LoRA folder; empty = base model")
    p.add_argument("--sample", type=int, default=200, help="0 = full GSM8K test")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype())
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    dev = kind()
    if dev == "mps":
        model = model.to("mps")
    elif dev == "cuda":
        model = model.to("cuda")
    model.eval()

    test = load_dataset("openai/gsm8k", "main", split="test")
    if args.sample and args.sample < len(test):
        test = test.shuffle(seed=args.seed).select(range(args.sample))

    ok = 0
    n = len(test)
    print(f"GSM8K n={n} device={dev} adapter={args.adapter or 'base'} seed={args.seed}")
    for i, ex in enumerate(test):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": ex["question"]},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt")
        if dev in {"mps", "cuda"}:
            ids = {k: v.to(dev) for k, v in ids.items()}
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=256, do_sample=False)
        text = tok.decode(out[0], skip_special_tokens=True)
        pred, gold = extract(text), extract(ex["answer"])
        hit = pred == gold and pred != ""
        ok += int(hit)
        if (i + 1) % 10 == 0 or i + 1 == n:
            print(f"{i + 1}/{n}  {100 * ok / (i + 1):.1f}%")

    print(f"GSM8K exact match: {ok}/{n} = {100 * ok / n:.2f}%  base={MODEL}")


if __name__ == "__main__":
    main()
