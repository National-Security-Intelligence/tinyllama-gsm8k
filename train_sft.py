#!/usr/bin/env python3
"""LoRA SFT of TinyLlama on GSM8K. Mac MPS uses float32 (no bf16)."""

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from model import LORA_TARGETS, MODEL
from device import dtype, kind, sft_precision

OUT = "outputs/sft"
SYSTEM = "Solve grade-school math. Show steps. End with #### <number>."


def format_row(ex, tok):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": ex["question"].strip()},
        {"role": "assistant", "content": ex["answer"].strip()},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def main():
    print("device", kind())
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = load_dataset("openai/gsm8k", "main")
    train = ds["train"].map(lambda ex: format_row(ex, tok), remove_columns=ds["train"].column_names)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype())
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=LORA_TARGETS, task_type="CAUSAL_LM",
    )
    batch = 1 if kind() == "mps" else 2
    args = SFTConfig(
        output_dir=OUT, num_train_epochs=2,
        per_device_train_batch_size=batch, gradient_accumulation_steps=16,
        learning_rate=2e-4, logging_steps=10, save_strategy="epoch",
        max_seq_length=1024, packing=False,
        use_mps_device=kind() == "mps",
        **sft_precision(),
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=train,
        processing_class=tok, peft_config=lora,
    )
    trainer.train()
    trainer.save_model(OUT)
    tok.save_pretrained(OUT)
    print("saved", OUT, "base", MODEL, "device", kind())


if __name__ == "__main__":
    main()
