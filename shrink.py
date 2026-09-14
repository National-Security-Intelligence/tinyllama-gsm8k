#!/usr/bin/env python3
"""Build a smaller Llama by copying evenly spaced TinyLlama layers.

TinyLlama: 22 layers, 2048 hidden, ~1.10B (embed+lm_head stay ~131M).

  --layers 16  → ~0.84B
  --layers 10  → ~0.57B
  --layers 8   → ~0.48B   (under 0.5B)

This is not lossless. Next step is SFT (and optional distill) on GSM8K.
"""

import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
from model import MODEL

OUT = "outputs/student"


def pick_indices(n_src: int, n_dst: int) -> list[int]:
    if n_dst >= n_src:
        return list(range(n_src))
    return [round(i * (n_src - 1) / (n_dst - 1)) for i in range(n_dst)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, default=8, help="8≈0.48B  10≈0.57B  16≈0.84B")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    teacher = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    cfg = LlamaConfig.from_pretrained(MODEL)
    src_n = cfg.num_hidden_layers
    cfg.num_hidden_layers = args.layers
    student = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)

    idx = pick_indices(src_n, args.layers)
    sd_t = teacher.state_dict()
    sd_s = student.state_dict()
    copied = 0
    for k, v in sd_s.items():
        if ".layers." in k:
            rest = k.split(".layers.", 1)[1]
            li_s, tail = rest.split(".", 1)
            src_k = k.replace(f".layers.{li_s}.", f".layers.{idx[int(li_s)]}.")
            if src_k in sd_t and sd_t[src_k].shape == v.shape:
                sd_s[k] = sd_t[src_k]
                copied += 1
        elif k in sd_t and sd_t[k].shape == v.shape:
            sd_s[k] = sd_t[k]
            copied += 1
    student.load_state_dict(sd_s)
    n = sum(t.numel() for t in student.parameters())
    Path(OUT).mkdir(parents=True, exist_ok=True)
    student.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    print(f"layers {src_n}->{args.layers} via {idx}")
    print(f"params {n/1e9:.3f}B  tensors copied {copied}  saved {OUT}")
    print("This dropped capacity. Run train_sft.py against this folder next.")


if __name__ == "__main__":
    main()
