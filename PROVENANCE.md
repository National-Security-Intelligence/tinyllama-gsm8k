# Provenance

- **Weights:** Gemma 3 1B Instruct (Google DeepMind, US)
- **MLX convert:** mlx-community/gemma-3-1b-it-8bit
- **Not used:** Qwen, DeepSeek, Yi, GLM, TinyLlama
- **Eval data:** openai/gsm8k test only
- **License:** Gemma Terms of Use (acknowledge on Hugging Face once)

TinyLlama scored ~0% here because it is not a math model and rarely emits `####`.
Gemma 3 1B published ~63% GSM8K 8-shot CoT. That is still not the Qwen 80% run.
Do not sell the Qwen checkpoint. SFT this Gemma on GSM8K gold traces.
