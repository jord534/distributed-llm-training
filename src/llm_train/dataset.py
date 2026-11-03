import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
from datasets import Dataset, Features, Sequence, Value, load_dataset
from .serialization import get_tokens_from_convo

FEATURES = Features({
    "prompt_id": Value("string"),
    "input_ids": Sequence(Value("int32")), # vocab size (131k) exceeds 16bits so we use int32
    "truncated": Value("bool"),
})


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


def make_cached_dataset(train_examples, val_examples, description: dict, output_path: str):
    # train_examples and val_examples are lists of dicts built by get_truncated_convo
    outpath = Path(output_path)
    if outpath.exists():
        raise FileExistsError(f"Cache {outpath} already exists.")
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # write everything in a temp dir so as to write in the approprite location only if no exception is raised
    tmp_path = Path(tempfile.mkdtemp(dir=outpath.parent, prefix=f".{outpath.name}."))
    try:
        Dataset.from_list(train_examples, features=FEATURES).save_to_disk(str(tmp_path / "train"))
        Dataset.from_list(val_examples, features=FEATURES).save_to_disk(str(tmp_path / "val"))

        all_examples = train_examples + val_examples
        manifest = {
            **description,  # dataset, tokenizer, selection, max_seq_length, environment...
            "stats": {
                "train": len(train_examples),
                "val": len(val_examples),
                "tokens": sum(len(e["input_ids"]) for e in all_examples),
                "truncated": sum(e["truncated"] for e in all_examples),
            },
            "fingerprint": {
                "train": compute_fingerprint(train_examples),
                "val": compute_fingerprint(val_examples),
            },
        }
        with open(tmp_path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=4)

        os.replace(tmp_path, outpath)  # atomic rename on the same disk
    except BaseException:
        shutil.rmtree(tmp_path, ignore_errors=True)  # don't leave a half-written temp folder behind
        raise

    return manifest

def compute_fingerprint(examples):
    # hash of the content not of the files so that (same ids and tokens) <=> (same fingerprint)
    h = hashlib.sha256()
    for example in examples:
        h.update(example["prompt_id"].encode())
        # length prefix: two different splits of the same token stream can't collide
        h.update(len(example["input_ids"]).to_bytes(4, "little"))
        h.update(np.asarray(example["input_ids"], dtype="<i4").tobytes())
    return "sha256:" + h.hexdigest()