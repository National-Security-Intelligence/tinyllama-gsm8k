#!/usr/bin/env python3
import re
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from model import MODEL

ADAPTER = "outputs/sft"
OUT = "outputs/grpo"
ANS = re.compile(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)")


def gold(text: str) -> str:
    m = ANS.findall(text.replace(",", ""))
    return m[-1] if m else ""


def reward_fn(completions, **kwargs):
    answers = kwargs.get("answer", [""] * len(completions))
    return [
        1.0 if gold(c if isinstance(c, str) else str(c)) and gold(c if isinstance(c, str) else str(c)) == gold(g) else 0.0
        for c, g in zip(completions, answers)
    ]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="auto")
    model = PeftModel.from_pretrained(base, ADAPTER)
    raw = load_dataset("openai/gsm8k", "main", split="train")

    def to_prompt(ex):
        messages = [
            {"role": "system", "content": "Solve grade-school math. Show steps. End with #### <number>."},
            {"role": "user", "content": ex["question"]},
        ]
        return {
            "prompt": tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            "answer": ex["answer"],
        }

    ds = raw.map(to_prompt, remove_columns=raw.column_names)
    args = GRPOConfig(
        output_dir=OUT, num_train_epochs=1, per_device_train_batch_size=1,
        gradient_accumulation_steps=8, learning_rate=5e-6, logging_steps=5,
        bf16=True, max_completion_length=512, num_generations=4,
    )
    trainer = GRPOTrainer(
        model=model, args=args, train_dataset=ds,
        processing_class=tok, reward_funcs=reward_fn,
    )
    trainer.train()
    trainer.save_model(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
