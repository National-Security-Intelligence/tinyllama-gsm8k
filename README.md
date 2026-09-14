# gsm8k-gemma (MLX)

TinyLlama hit 0% on `####` match. That is expected. This repo now uses **Gemma 3 1B Instruct** (Google, US).

```bash
git pull
uv sync
uv run python eval_gsm8k.py --sample 200 --seed 0
```

Prints **strict** (`####`) and **flexible** (last number). First two completions are dumped so you can see format vs math errors.

Then train:

```bash
uv run python prepare_gsm8k.py
uv run python train_sft.py
uv run python eval_gsm8k.py --sample 200 --adapter outputs/sft
```
