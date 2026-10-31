
import torch

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

    model = model.to(device)

    def count_parameters(module):
        # Deduplicate shared/tied parameters
        params = {
            id(p): p
            for p in module.parameters()
        }
        return sum(p.numel() for p in params.values())


    total = count_parameters(model)

    print(f"Total parameters: {total:,}")
    print(f"Total parameters (B): {total / 1e9:.3f} B")
    model.eval()

    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = load_model()

    inputs = tokenizer.encode(
        "To be or not",
        return_tensors="pt",
    ).to(model.device)

    answer = input("Run model.generate? [y/N]: ").strip().lower()

    if answer == "y":
        with torch.inference_mode():
            outputs = model.generate(
                inputs,
                max_new_tokens=30,
            )

    print(tokenizer.decode(outputs[0]))
