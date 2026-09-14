# tinyllama-gsm8k

**[uv](https://docs.astral.sh/uv/) only.** No pip.

## macOS (you)

```bash
brew install uv
uv --version
```

Then in this repo:

```bash
uv sync
uv run python train_sft.py
```

Apple Silicon uses Metal (MPS). Scripts already turn off bf16 on Mac.
Need ~16 GB unified memory for 1.1B LoRA. A 8 GB Mac will swap and crawl.

## Commands

```bash
uv sync
uv run python shrink.py --layers 8
uv run python train_sft.py
uv run python autotune.py
uv run python eval_gsm8k.py
```
