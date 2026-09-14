#!/usr/bin/env python3
"""If a run fails, adjust lr/steps from the log and retry. uv run python autotune.py"""

from __future__ import annotations

import json
from pathlib import Path

LOG = Path("outputs/runs.jsonl")


def read_runs() -> list[dict]:
    if not LOG.exists():
        return []
    out = []
    for line in LOG.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def decide(runs: list[dict]) -> dict:
    if not runs:
        return {"lr": 2e-4, "epochs": 2, "note": "first run"}
    last = runs[-1]
    lr = float(last.get("lr", 2e-4))
    epochs = int(last.get("epochs", 2))
    loss = last.get("loss")
    status = last.get("status", "fail")
    if loss is None or not isinstance(loss, (int, float)) or loss > 8:
        return {"lr": max(lr * 0.25, 1e-6), "epochs": epochs, "note": "loss blew up → cut lr 4×"}
    if status == "fail" and loss > 1.5:
        return {"lr": min(lr * 1.6, 5e-4), "epochs": min(epochs + 1, 4), "note": "flat/fail → raise lr, +1 epoch"}
    if status == "ok":
        return {"lr": lr, "epochs": epochs, "note": "keep", "stop": True}
    return {"lr": lr * 0.7, "epochs": min(epochs + 1, 4), "note": "improving but not there → lower lr, longer"}


def append(run: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(run) + "\n")


def main() -> None:
    runs = read_runs()
    nxt = decide(runs)
    print(json.dumps(nxt, indent=2))
    if nxt.get("stop"):
        print("no retry — last run was ok")
        return
    print("next:")
    print(f"  uv run python train_sft.py  # set lr={nxt['lr']} epochs={nxt['epochs']}")
    print("after the run, append {lr, epochs, loss, status} to outputs/runs.jsonl")
    print("then: uv run python autotune.py")


if __name__ == "__main__":
    main()
