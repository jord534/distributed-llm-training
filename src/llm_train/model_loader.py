import torch
from collections import defaultdict

from transformers import (
    Mistral3ForConditionalGeneration,
    MistralCommonBackend,
    AutoConfig
)

LANGUAGE_MODEL_CATEGORIES = {"language_model", "lm_head"}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def param_category_of(name: str) -> str:
    # can be "model.language_model.layers.0..." 
    # or "language_model.model.layers.0..." 
    # or "lm_head.weight"  
    # we are interested in lm_head or language_model
    parts = name.split(".")
    if parts[0] == "model":
        return parts[1]
    return parts[0]


def freeze_vision(model: Mistral3ForConditionalGeneration):
    for name, param in model.named_parameters():
        if param_category_of(name) not in LANGUAGE_MODEL_CATEGORIES:
            param.requires_grad_(False)  # freezing vision and multimodal weights


def param_type_count(model: Mistral3ForConditionalGeneration, dtype: torch.dtype):
    # Separate pass: reads the state left by freeze_vision
    dtype_counts = defaultdict(int)
    param_category = defaultdict(lambda: [0, 0])  # category -> [numel, trainable numel]
    wrong_dtype_names = []
    for name, param in model.named_parameters():
        category = param_category_of(name)
        numel = param.numel()
        param_category[category][0] += numel  # counting lm, vision and multimodal params
        dtype_counts[param.dtype] += numel  # counting params in each dtype
        if param.requires_grad:
            param_category[category][1] += numel
        if param.dtype != dtype:
            wrong_dtype_names.append(name)

    return dtype_counts, param_category, wrong_dtype_names


def check_for_missing_or_unexpected_param(loading_info):
    issues = {
        key: values
        for key, values in loading_info.items()
        if values
    }

    if issues:
        raise RuntimeError(f"Model loading issues: {issues}")
    return None


def type_check_and_freeze_check(dtype_counts, param_category, wrong_dtype_names, dtype:torch.dtype):
    wrong_dtypes = sum(count for dt, count in dtype_counts.items() if dt != dtype)

    if wrong_dtypes > 0:
        raise RuntimeError(
            f"Found {wrong_dtypes} wrong dtype parameters ({dict(dtype_counts)}), "
            f"e.g. {wrong_dtype_names[:5]}"
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
    return lm_trainable


def load_tokenizer(model_id: str, revision: str, mode: str) -> MistralCommonBackend:
    # mode="finetuning" for training data (conversations ending with an assistant answer),
    # mode="test" for generation requests (ending with a user message)
    return MistralCommonBackend.from_pretrained(model_id, revision=revision, mode=mode)


def load_model(model_id, model_revision, device, dtype: torch.dtype):
    print(f"Loading model on: {device}")

    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unsupported on this GPU") # BF16 is import for mixed precision and therefore speed

    model, loading_info = (
        Mistral3ForConditionalGeneration.from_pretrained(
            model_id,
            revision=model_revision,
            dtype=dtype,
            output_loading_info=True,
        )
    )

    check_for_missing_or_unexpected_param(loading_info)
    freeze_vision(model)
    dtype_counts, param_category, wrong_dtype_names = param_type_count(model, dtype)
    lm_trainable = type_check_and_freeze_check(dtype_counts, param_category, wrong_dtype_names, dtype)

    model = model.to(device)

    return model, lm_trainable

def shrink_sub_config(sub_config, dims: dict, name: str):
    # dims maps a config field to its reduced value, e.g. {"num_hidden_layers": 2, "hidden_size": 256}
    for field, value in dims.items():
        # setattr on a config silently creates unknown fields: a typo would be ignored
        if not hasattr(sub_config, field):
            raise RuntimeError(f"Unknown field {field!r} in {name}")
        setattr(sub_config, field, value)

    # GQA: each KV head serves a group of query heads
    kv_heads = getattr(sub_config, "num_key_value_heads", sub_config.num_attention_heads)
    if sub_config.num_attention_heads % kv_heads != 0:
        raise RuntimeError(
            f"{name}: num_attention_heads {sub_config.num_attention_heads} "
            f"not divisible by num_key_value_heads {kv_heads}"
        )

    # rotary embeddings rotate pairs of dimensions: head_dim must be even
    head_dim = getattr(sub_config, "head_dim", None)
    if head_dim is not None and head_dim % 2 != 0:
        raise RuntimeError(f"{name}: head_dim {head_dim} must be even")
    return None


def create_dev_model(model_id, model_revision, device, text_dims: dict, vision_dims: dict, dtype: torch.dtype):
    # same architecture and parameter names as the real model, random weights
    # vocab_size is left untouched: the real tokenizer and data cache stay usable
    print(f"Creating dev model on: {device}")

    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unsupported on this GPU")

    if "vocab_size" in text_dims:
        raise RuntimeError("vocab_size must not be changed: the dev model shares the real tokenizer")

    config = AutoConfig.from_pretrained(model_id, revision=model_revision)  # download the OG config json
    shrink_sub_config(config.text_config, text_dims, "text_config") # shrinking to specified values in the argument
    shrink_sub_config(config.vision_config, vision_dims, "vision_config")

    model = Mistral3ForConditionalGeneration(config).to(dtype)  # random init of weights and getting a model object analogous to the large model

    # same guarantees as load_model
    freeze_vision(model)
    dtype_counts, param_category, wrong_dtype_names = param_type_count(model, dtype)
    lm_trainable = type_check_and_freeze_check(dtype_counts, param_category, wrong_dtype_names, dtype)

    model = model.to(device)

    return model, lm_trainable