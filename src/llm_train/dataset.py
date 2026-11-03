from datasets import load_dataset
from .serialization import get_tokens_from_convo

def get_dataset(set_name: str, revision: str, split: str,
                sample_size: int, seed: int, val_frac: float, cache_dir : str):

    dataset = load_dataset(
        path=set_name,
        revision=revision,
        split=split,
        cache_dir=cache_dir,
    )
    
    # shuffle
    dataset = dataset.shuffle(seed)
    # sample size
    if sample_size > len(dataset):
        raise ValueError(f"Asking for {sample_size} but dataset only contains {len(dataset)} samples.")
    dataset = dataset.select(range(sample_size))
    # val frac
    splits = dataset.train_test_split(test_size = val_frac, seed = seed)

    # here zip() is faster that list comprehension
    train = list(zip(splits["train"]["prompt_id"], splits["train"]["messages"]))
    val = list(zip(splits["test"]["prompt_id"], splits["test"]["messages"]))
    return train, val


def get_truncated_convo(prompt_id, convo, max_len, tokenizer):
    token_list = get_tokens_from_convo(convo, tokenizer)
    return {
        "prompt_id": prompt_id,
        "input_ids": token_list[:max_len],
        "truncated": max_len < len(token_list)
    }