# Google Gemma 3 1B Instruct — US origin, not Qwen, not TinyLlama.
# Official 8-shot GSM8K CoT is ~63%. TinyLlama base is ~0% on #### match.
MODEL = "mlx-community/gemma-3-1b-it-8bit"
# Higher fidelity on 16GB+: "mlx-community/gemma-3-1b-it-bf16"
SYSTEM = "Solve grade-school math. Show every step. Put the final number on its own line as #### <number>."
