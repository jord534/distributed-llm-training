from datasets import load_dataset


def get_dataset(set_name: str, revision: str, split: str,
                sample_size: int, seed: int, val_frac: float, cache_dir : str):

    dataset = load_dataset(
        path=set_name,
        revision=revision,
        split=split,
        cache_dir=cache_dir,
    )

    # now let's get a seed based shuffle, the amount of samples we need and the val samples split

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