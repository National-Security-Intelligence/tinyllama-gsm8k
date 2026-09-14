# tinyllama-gsm8k

Apple Silicon via **MLX** (not PyTorch MPS). **uv** + **Ruff** only.

```bash
git clone https://github.com/National-Security-Intelligence/tinyllama-gsm8k.git
cd tinyllama-gsm8k
uv sync
uv run ruff check .
uv run python eval_gsm8k.py --sample 200 --seed 0
```

`--sample 200` = random 200 from official GSM8K test. `--sample 0` = full 1,319.

Train (after baseline):

```bash
uv run python prepare_gsm8k.py
uv run python train_sft.py
uv run python eval_gsm8k.py --sample 200 --adapter outputs/sft
```
