
import torch
from collections import defaultdict, Counter

from transformers import (
    Mistral3ForConditionalGeneration,
    MistralCommonBackend,
)

MODEL_ID = "mistralai/Ministral-3-3B-Base-2512"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_model():
    device = get_device()
    print(f"Loading model on: {device}")

    if device.type == "cuda":
        assert torch.cuda.is_bf16_supported(), "BF16 unsupported"

    tokenizer = MistralCommonBackend.from_pretrained(MODEL_ID)

    model, loading_info = (
        Mistral3ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            output_loading_info=True,
        )
    )
    print("\n\n")

    name_categories = Counter()
    for name, param in model.named_parameters():
        name_blocks = name.split(".")[1]
        name_categories[name_blocks] += param.numel()/1e9
    print("Categories of weights and their respective count in billions:\n")
    for key, val in name_categories.items():
        print(key, val)
    print("\n\n")

    # we can check that language is 3.4B, vision is 0.4B

    # let's freeze every non language weight:
    for name, param in model.named_parameters():
        if name.split(".")[1] != "language_model":
            param.requires_grad_(False)

    # Now let's check that any non language weight is indeed frozen:
    stats = defaultdict(lambda: {"tensors": 0, "params": 0, "trainable": 0})
    for name, param in model.named_parameters():
        category = name.split(".")[1]
        stats[category]["tensors"] += 1
        stats[category]["params"] += param.numel()

        if param.requires_grad:
            stats[category]["trainable"] += param.numel()

    print("Remaining trainable parameters by category after freezing non language weights attempt:\n")
    for category, values in stats.items():
        print(category, values)
    print("\n\n")
    model = model.to(device)

    model.eval()

    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = load_model()

    answer = input("Run model.generate? [y/N]: ").strip().lower()

    if answer == "y":
        inputs = tokenizer.encode(
                "How to play sudoku ?",
                return_tensors="pt",
            ).to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                inputs,
                max_new_tokens=30,
            )

        print(tokenizer.decode(outputs[0]))
