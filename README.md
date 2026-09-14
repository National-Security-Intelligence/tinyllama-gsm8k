# tinyllama-gsm8k

Public training scripts. **Not Qwen.** Base is [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0).

| | |
|---|---|
| Params | 1.1B (there is no official TinyLlama 0.5B) |
| Architecture | Llama (Meta, US) |
| Pretrain | TinyLlama project, SUTD Singapore, 3T tokens |
| License | Apache-2.0 |
| PRC weights | None |

Do not sell a Qwen checkpoint as this model. Scores do not transfer. Retrain from GSM8K gold traces only — no distillation from Qwen.

If procurement needs **US-origin and ≤0.5B**, set `MODEL` in `model.py` to `google/gemma-3-270m-it`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train_sft.py
python train_grpo.py
python eval_gsm8k.py --adapter outputs/sft
```
