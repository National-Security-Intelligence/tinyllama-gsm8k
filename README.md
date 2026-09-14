# tinyllama-gsm8k

**Package manager: [uv](https://docs.astral.sh/uv/) only.** No pip, no venv module.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run python train_sft.py
uv run python autotune.py
```

Base: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (not Qwen).

| Command | What |
|---|---|
| `uv sync` | create `.venv` and install deps |
| `uv run python shrink.py --layers 8` | ~0.48B student |
| `uv run python train_sft.py` | SFT on GSM8K |
| `uv run python autotune.py` | if a run fails, retune lr/steps and retry |
| `uv run python eval_gsm8k.py` | GSM8K exact match |
