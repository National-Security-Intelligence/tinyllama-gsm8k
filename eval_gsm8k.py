#!/usr/bin/env python3
import argparse, re, torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import MODEL

ANS = re.compile(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)")


def extract(text: str) -> str:
    m = ANS.findall(text.replace(",", ""))
    return m[-1] if m else ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="outputs/sft")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="auto", device_map="auto")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    test = load_dataset("openai/gsm8k", "main", split="test")
    if args.limit:
        test = test.select(range(args.limit))
    ok = 0
    for i, ex in enumerate(test):
        messages = [
            {"role": "system", "content": "Solve grade-school math. Show steps. End with #### <number>."},
            {"role": "user", "content": ex["question"]},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=256, do_sample=False)
        text = tok.decode(out[0], skip_special_tokens=True)
        pred, g = extract(text), extract(ex["answer"])
        ok += int(pred == g and pred != "")
        if (i + 1) % 50 == 0:
            print(i + 1, f"{100 * ok / (i + 1):.1f}%")
    n = len(test)
    print(f"GSM8K exact match: {ok}/{n} = {100 * ok / n:.2f}%  base={MODEL}")


if __name__ == "__main__":
    main()
