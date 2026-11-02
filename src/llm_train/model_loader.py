import torch
from collections import defaultdict

from transformers import (
    Mistral3ForConditionalGeneration,
    MistralCommonBackend,
)

MODEL_ID = "mistralai/Ministral-3-3B-Base-2512"
MODEL_REVISION = "main"  # TODO: pin to a commit sha from the Hub before any measured run

# Top-level parameter groups that belong to the text model and must stay trainable.
# Verify them with print(model): the LM head usually sits outside the language_model submodule.
LANGUAGE_MODEL_CATEGORIES = {"language_model", "lm_head"}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def param_category_of(name: str) -> str:
    # "model.language_model.layers.0..."  -> "language_model"   (recent Transformers layout)
    # "language_model.model.layers.0..."  -> "language_model"   (older layout)
    # "lm_head.weight"                    -> "lm_head"
    parts = name.split(".")
    if parts[0] == "model":
        return parts[1]
    return parts[0]


def freeze_vision(model: Mistral3ForConditionalGeneration):
    for name, param in model.named_parameters():
        if param_category_of(name) not in LANGUAGE_MODEL_CATEGORIES:
            param.requires_grad_(False)  # freezing vision tower and multimodal projector


def param_type_count(model: Mistral3ForConditionalGeneration):
    # Separate pass: reads the state left by freeze_vision, so the checks below can actually fail.
    dtype_counts = defaultdict(int)
    param_category = defaultdict(lambda: [0, 0])  # category -> [numel, trainable numel]
    non_bf16_names = []
    for name, param in model.named_parameters():
        category = param_category_of(name)
        numel = param.numel()
        param_category[category][0] += numel  # counting lm, vision and multimodal params
        dtype_counts[param.dtype] += numel  # counting params in each dtype
        if param.requires_grad:
            param_category[category][1] += numel
        if param.dtype != torch.bfloat16:
            non_bf16_names.append(name)

    return dtype_counts, param_category, non_bf16_names


def check_for_missing_or_unexpected_param(loading_info):
    issues = {
        key: values
        for key, values in loading_info.items()
        if values
    }

    if issues:
        raise RuntimeError(f"Model loading issues: {issues}")
    return None


def type_check_and_freeze_check(dtype_counts, param_category, non_bf16_names):
    non_bf16 = sum(count for dtype, count in dtype_counts.items() if dtype != torch.bfloat16)

    if non_bf16 > 0:
        raise RuntimeError(
            f"Found {non_bf16} non-BF16 parameters ({dict(dtype_counts)}), "
            f"e.g. {non_bf16_names[:5]}"
        )

    # Catches both a naming surprise (no language-model group found) and over-freezing.
    lm_trainable = sum(
        param_category[cat][1] for cat in LANGUAGE_MODEL_CATEGORIES if cat in param_category
    )
    if lm_trainable == 0:
        raise RuntimeError(
            f"No trainable language-model parameters; categories found: {list(param_category)}"
        )

    for cat, (numel, trainable) in param_category.items():
        if cat not in LANGUAGE_MODEL_CATEGORIES and trainable > 0:
            raise RuntimeError(f"Found {trainable} non frozen params (out of {numel}) in {cat}.")
    return None


def load_tokenizer():
    return MistralCommonBackend.from_pretrained(MODEL_ID, revision=MODEL_REVISION)


def load_model():
    device = get_device()
    print(f"Loading model on: {device}")

    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unsupported on this GPU")

    model, loading_info = (
        Mistral3ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            dtype=torch.bfloat16,
            output_loading_info=True,
        )
    )

    check_for_missing_or_unexpected_param(loading_info)
    freeze_vision(model)
    dtype_counts, param_category, non_bf16_names = param_type_count(model)
    type_check_and_freeze_check(dtype_counts, param_category, non_bf16_names)

    model = model.to(device)

    return model


if __name__ == "__main__":
    model = load_model()
    tokenizer = load_tokenizer()

    _, param_category, _ = param_type_count(model)
    for cat, (numel, trainable) in param_category.items():
        print(f"{cat}: {numel / 1e9:.3f} B params, {trainable / 1e9:.3f} B trainable")

    answer = input("Run model.generate? [y/N]: ").strip().lower()

    if answer == "y":
        model.eval()
        prompt = "How to play sudoku ?"
        user_prompt = input("Type in special prompt [empty for default]:").strip()
        if user_prompt != "":
            prompt = user_prompt
        inputs = tokenizer.encode(
            prompt,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                inputs,
                max_new_tokens=30,
            )

        print(tokenizer.decode(outputs[0]))