from pathlib import Path
import yaml
import os
import hashlib
from audit_environment import get_audit
import shutil
from transformers import Mistral3ForConditionalGeneration
from datasets import Dataset
import json

from llm_train.train import (
    set_seed,
    get_optimizer_and_lr_scheduler,
    training_step,
    val_step,
    get_batch_from_examples
)

from llm_train.model_loader import (
    create_dev_model,
    load_model,
)

from llm_train.dataset import (
    read_cached_data
)

REPO_ROOT = Path(__file__).resolve().parent.parent

def copy_audit_and_config(train_cfg, this_run_dir, config):
    # copy the config file there
    shutil.copyfile(train_cfg, this_run_dir)
    # make a run directorys pecific to this run
    try:
        os.makedirs(this_run_dir, exist_ok=False)
    except OSError as e:
        print("The run name in the run config already has a directory. If your run is not a repeat, please provide a new name.")
        raise e
    

    # run the environment audit
    # generate audit hash:
    m = hashlib.sha256()
    m.update(config)
    hash = m.hexdigest()
    audit_file = get_audit(hash)

    # move it into this_run_dir
    shutil.copyfile(audit_file, this_run_dir)
    return None

def instantiate_model(config) -> Mistral3ForConditionalGeneration:
    # instantiate model
    device = config["device"]
    model_cfg = config["model"]

    model_id = model_cfg["model_id"]
    revision = model_cfg["revision"]

    if model_cfg["kind"] == "dev":
        text_dims = model_cfg["dev_dims"]["text_dims"]
        vision_dims = model_cfg["dev_dims"]["vision_dims"]
        model = create_dev_model(model_id, revision, device, text_dims, vision_dims)
    else:
        model = load_model(model_id, revision, device)

    return model

def fetch_optimizer_scheduler(config, model: Mistral3ForConditionalGeneration):
    train = config["train"]
    lr = train["learning_rate"]
    warmup_steps = train["warmup_steps"]
    total_steps = train["total_steps"]
    weight_decay = train["weight_decay"]

    optimizer, scheduler = get_optimizer_and_lr_scheduler(model.parameters, lr, weight_decay, warmup_steps, total_steps)
    return optimizer, scheduler, total_steps


def training_launcher(train_config: str):
    run_dir = REPO_ROOT / "artifacts" / "runs "
    # check artifacts/run exists, else create it
    os.makedirs(run_dir, exist_ok=True)

    train_cfg = Path(train_config)
    if not train_cfg.exists():
        raise FileNotFoundError(f"The following config file was not found: {train_cfg}")
    
    with open(train_cfg, "r") as file:
        config = yaml.safe_load(file)

    run_name = config["run_name"]
    this_run_dir = run_dir / run_name
    copy_audit_and_config(train_cfg, this_run_dir, config)

    # set seed
    seed = config["seed"]
    set_seed(seed)

    # instantiate model
    model = instantiate_model(config)

    # read cached data
    cache_dir = Path(config["data"]["cache"])
    train_set, val_set, manifest = read_cached_data(REPO_ROOT / cache_dir)

    # create optimizer
    optimizer, scheduler, total_steps = fetch_optimizer_scheduler(config, model)

    # loop over training steps
    train = config["train"]
    max_grad_norm = train["max_grad_norm"]
    autocast = train["autocast"]
    activate_checkpointing = train["activate_checkpointing"]
    eval_every = train["eval_every"]
    micro_batch_size = train["micro_batch_size"]
    accumulation_step = train["accumulation_step"]

    log_path = this_run_dir / "log.jsonl"

    train_set.batch

    i = 0
    for step in range(total_steps):
        # prepare the mini batch:
        mini_batches = [] 
        for _ in range(accumulation_step):
            examples = train_set[i:i + micro_batch_size]
            batch = get_batch_from_examples(examples)
            mini_batches.append(batch)
            i += micro_batch_size

        log = training_step(
            model,
            mini_batches,
            optimizer,
            scheduler,
            max_grad_norm,
            autocast
        )
        # save log line
        with open(log_path, "a") as file:
            json.dump(log, file)

        if (step + 1) % eval_every == 0 == 0:
            val_batches = []
            loss = val_step(model, val_batches)
    







    

    

    
