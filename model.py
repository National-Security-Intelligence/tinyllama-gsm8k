MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# US-origin ≤0.5B swap:
# MODEL = "google/gemma-3-270m-it"

LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
