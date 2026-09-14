# tinyllama-gsm8k

**uv only. Ruff for lint/format.**

macOS:

```bash
git clone https://github.com/National-Security-Intelligence/tinyllama-gsm8k.git
cd tinyllama-gsm8k
uv sync
uv run ruff check .
uv run ruff format .
```

## Evaluate GSM8K

Random 200 from the official test split (default, sane on a Mac):

```bash
uv run python eval_gsm8k.py --sample 200 --seed 0
```

Full benchmark (1,319). Slow on MPS:

```bash
uv run python eval_gsm8k.py --sample 0
```

No `--adapter` = base TinyLlama (baseline). After SFT:

```bash
uv run python eval_gsm8k.py --sample 200 --adapter outputs/sft
```

Gold match is the `#### <number>` field. That is GSM8K.

## Train

```bash
uv run python train_sft.py
uv run python autotune.py
```
