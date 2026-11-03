from src.llm_train.model_loader import load_tokenizer
from pathlib import Path
import yaml
from src.llm_train.dataset import get_dataset, get_truncated_convo, make_cached_dataset

# find repo root to pass hf_cache path to gat_dataset
# Path(__file__).resolve().parent.parent is the repo root when inside this .py inside scripts/


# train_examples = [get_truncated_convo(pid, convo, max_len, tokenizer) for pid, convo in train]


def write_examples(data_config_path: str):
    # get model id, revision and mode from the config:
    yaml_path = Path(data_config_path)
    if not yaml_path.exists():
        raise FileNotFoundError("Could not find the data config file.")
    with open(yaml_path, "r") as file:
        data_config_dict = yaml.safe_load(file)

    #### TOKENIZER
    # read tokenizer spec
    tokenizer_config = data_config_dict["tokenizer"]
    model_id = tokenizer_config["model_id"]
    revision = tokenizer_config["revision"]
    mode = tokenizer_config["mode"]

    # laod the tokenizer
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

    # cache
    cache = data_config_dict["cache"]
    root = cache["root"]
    hf_cache = cache["hf_cache"]

    train, val = get_dataset(dataset_id, dataset_revision, dataset_split, sample_size, seed, val_frac, hf_cache)
    ####


    train_examples = [get_truncated_convo(pid, convo, max_seq_len, tokenizer,) for pid, convo in train]
    val_examples = [get_truncated_convo(pid, convo, max_seq_len, tokenizer) for pid, convo in val]

    make_cached_dataset(train_examples, val_examples, description, output_path)
    

    return None