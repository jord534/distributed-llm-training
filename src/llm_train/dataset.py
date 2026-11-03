from pathlib import Path
from datasets import load_dataset, load_from_disk


def get_dataset(set_name: str, revision: str, split: str,
                sample_size: int, seed: int, val_frac: float):

    dataset = load_dataset(
        path=set_name,
        revision=revision,
        split=split
    )

    # now let's get a seed based shuffle, the amount of samples we need and the val samples split

    # shuffle
    dataset = dataset.shuffle(seed)
    # sample size
    if sample_size > dataset._indices:
        print(f"Asked for {sample_size} samples but only {dataset._indices} present, serving all that are present.")
        
    dataset = dataset.select(range(min(sample_size, dataset._indices))
    # val frac
    splits = dataset.train_test_split(test_size = val_frac, seed = seed)

    train = splits["train"]
    val = splits["test"]
    skipped = None

    return train, val, skipped