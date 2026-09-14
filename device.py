import torch


def kind() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def dtype():
    k = kind()
    if k == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def sft_precision() -> dict:
    return {"bf16": kind() == "cuda", "fp16": False}
