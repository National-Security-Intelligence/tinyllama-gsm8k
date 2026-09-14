#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

from model import SYSTEM

OUT = Path("data/gsm8k")


def row(question: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": SYSTEM + "\n\n" + question.strip()},
            {"role": "assistant", "content": answer.strip()},
        ]
    }


def dump(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ds = load_dataset("openai/gsm8k", "main")
    train = [row(x["question"], x["answer"]) for x in ds["train"]]
    n_val = min(200, max(1, len(train) // 20))
    valid, train = train[-n_val:], train[:-n_val]
    dump(OUT / "train.jsonl", train)
    dump(OUT / "valid.jsonl", valid)
    print(f"wrote {len(train)} train {len(valid)} valid -> {OUT}")


if __name__ == "__main__":
    main()
