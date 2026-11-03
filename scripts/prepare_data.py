from llm_train.model_loader import load_tokenizer
import hashlib
import json
import sys
from pathlib import Path
import yaml
from llm_train.dataset import get_dataset, get_truncated_convo, make_cached_dataset


# Path(__file__).resolve().parent.parent is the repo root when this .py is inside scripts/
REPO_ROOT = Path(__file__).resolve().parent.parent


def write_examples(data_config_path: str):
    # get model id, revision and mode from the config:
    yaml_path = Path(data_config_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Could not find the data config file: {yaml_path}")
    with open(yaml_path, "r") as file:
        data_config_dict = yaml.safe_load(file)

    #### TOKENIZER
    # read tokenizer spec
    tokenizer_config = data_config_dict["tokenizer"]
    model_id = tokenizer_config["model_id"]
    revision = tokenizer_config["revision"]
    mode = tokenizer_config["mode"]

    # load the tokenizer
    tokenizer = load_tokenizer(model_id, revision, mode)
    ####

    #### DATASET PREP
    # point to the required dataset
    dataset_config = data_config_dict["dataset"]
    dataset_id = dataset_config["id"]
    dataset_revision = dataset_config["revision"]
    dataset_split = dataset_config["split"]

    # data selection
    data_selection = data_config_dict["selection"]
    sample_size = data_selection["sample_size"]
    seed = data_selection["seed"]
    val_frac = data_selection["val_fraction"]

    data_processing = data_config_dict["processing"]
    max_seq_len = data_processing["max_seq_length"]

    # cache: paths in the config are relative to the repo root, not to where the script is launched
    cache = data_config_dict["cache"]
    root = REPO_ROOT / cache["root"]
    hf_cache = REPO_ROOT / cache["hf_cache"]

    train, val = get_dataset(dataset_id, dataset_revision, dataset_split, sample_size, seed, val_frac, str(hf_cache))
    ####

    #### EXAMPLES
    train_examples = [get_truncated_convo(pid, convo, max_seq_len, tokenizer) for pid, convo in train]
    val_examples = [get_truncated_convo(pid, convo, max_seq_len, tokenizer) for pid, convo in val]
    ####

    #### WRITE THE CACHE
    # essentially we are ensuring a one to one mapping of config yaml to hash
    content = {key: data_config_dict[key] for key in ("dataset", "tokenizer", "selection", "processing")}
    # short hash of that content in the folder name
    key = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:8]
    output_path = root / f"{dataset_id.split('/')[-1]}_{dataset_split}_{key}"
    description = dict(content)

    manifest = make_cached_dataset(train_examples, val_examples, description, str(output_path))
    print(f"Cache written to {output_path}")
    print(f"Stats: {manifest['stats']}")
    print(f"Fingerprint: {manifest['fingerprint']}")
    ####

    return None


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: uv run python scripts/prepare_data.py <data config path>")
    write_examples(sys.argv[1])