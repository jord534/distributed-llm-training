from pathlib import Path
import yaml
import os
from audit_environment import get_audit
import shutil
from transformers import Mistral3ForConditionalGeneration
from datasets import Dataset
import json
import torch
import sys
import hashlib
import time
import struct
import tempfile
import itertools
from torch.utils.data import DataLoader, DistributedSampler

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

TORCH_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

def copy_audit_and_config(this_run_dir, train_cfg):
    
    # make a run directorys pecific to this run
    try:
        os.makedirs(this_run_dir, exist_ok=False)
    except OSError as e:
        print("The run name in the run config already has a directory. If your run is not a repeat, please provide a new name.")
        raise e
    # copy the config file there
    shutil.copyfile(train_cfg, this_run_dir / "config.yaml")

    # run the environment audit
    m = hashlib.sha256()
    t = time.time()
    b = struct.pack("f", t)
    m.update(b)
    hash = m.hexdigest()
    audit_file = get_audit(hash)

    # move it into this_run_dir
    shutil.copyfile(audit_file, this_run_dir /  "environment.txt")
    return None

def instantiate_model(config) -> Mistral3ForConditionalGeneration:
    # instantiate model
    device = torch.device(config["device"])
    model_cfg = config["model"]

    model_id = model_cfg["model_id"]
    revision = model_cfg["revision"]
    dtype = model_cfg["dtype"]
    try:
        dtype = TORCH_DTYPES[dtype]
    except KeyError:
        raise ValueError(f"model.dtype must be one of {list(TORCH_DTYPES)}, got {dtype!r}") from None

    if model_cfg["kind"] == "dev":
        text_dims = config["dev_dims"]["text_dims"]
        vision_dims = config["dev_dims"]["vision_dims"]
        model = create_dev_model(model_id, revision, device, text_dims, vision_dims, dtype)
    else:
        model = load_model(model_id, revision, device, dtype)

    return model

def fetch_optimizer_scheduler(config, model: Mistral3ForConditionalGeneration):
    train = config["train"]
    lr = train["learning_rate"]
    warmup_steps = train["warmup_steps"]
    total_steps = train["total_steps"]
    weight_decay = train["weight_decay"]

    optimizer, scheduler = get_optimizer_and_lr_scheduler(model.parameters(), lr, weight_decay, warmup_steps, total_steps)
    return optimizer, scheduler, total_steps


def training_launcher(train_config: str):
    # This reads the config
    # and goes througha all steps to start training

    train_cfg = Path(train_config)
    if not train_cfg.exists():
        raise FileNotFoundError(f"The following config file was not found: {train_cfg}")
    
    run_dir = REPO_ROOT / "artifacts" / "runs"
    with open(train_cfg, "r") as file:
        config = yaml.safe_load(file)
    # check artifacts/run exists, if not create a temp dir and copy all in the actual dir at the end (to catch exceptions)
    run_name = config["run_name"]
    perm_run_dir = run_dir / run_name

    if perm_run_dir.exists():
        raise FileExistsError(f"Run already exists: {perm_run_dir}")

    
    with tempfile.TemporaryDirectory() as tmp_dir :
        tmp_dir = Path(tmp_dir)

        this_run_dir = tmp_dir / run_name
        
        copy_audit_and_config(this_run_dir, train_cfg)

        # set seed
        print("Setting seed...\n")
        seed = config["seed"]
        set_seed(seed)

        # instantiate model
        print("Instantiate model... \n")
        model = instantiate_model(config)

        # read cached data
        print("Reading cached data for training...\n")
        cache_dir = Path(config["data"]["cache"])
        max_training_examples = config["data"]["max_train_examples"]
        train_set, val_set, manifest = read_cached_data(REPO_ROOT / cache_dir)
        if max_training_examples is not None:
            if max_training_examples > len(train_set):
                raise ValueError("Requested number of training examples exceeds available examples")
            train_set = train_set.select(range(max_training_examples)) # allow for smaller examples to be used for testing purposes

        # create optimizer
        print("Fetching optimizer and scheduler... \n")
        optimizer, scheduler, total_steps = fetch_optimizer_scheduler(config, model)

        # loop over training steps
        train = config["train"]
        max_grad_norm = train["max_grad_norm"]
        autocast = train["autocast"]
        activation_checkpointing = train["activation_checkpointing"]
        eval_every = train["eval_every"]
        micro_batch_size = train["micro_batch_size"]
        accumulation_steps = train["accumulation_steps"]

        if activation_checkpointing:
            model.gradient_checkpointing_enable()


        # copy all from temp dir to actual dir
        shutil.move(str(this_run_dir), str(perm_run_dir))

    log_path = perm_run_dir / "log.jsonl"

    print("GEtting sampler and loader ready to serve batches... \n")
    train_sampler, train_loader = get_sampler_loader(train_set, seed, micro_batch_size, get_batch_from_examples)
    train_batches = batches_on_tap(train_loader, train_sampler)

    val_loader = DataLoader(val_set, batch_size=micro_batch_size, shuffle=False, collate_fn=get_batch_from_examples, drop_last=False)
    # val only needs a loader as we are going through all val exmaples each time we run val_step

    print("="*40)
    print("STARTING TRAINING LOOP\n\n")
    eta = None
    steptimes = []
    for step in range(total_steps):

        # prepare the micro_batches
        micro_batches = [next(train_batches) for _ in range(accumulation_steps)] 

        log = training_step(
            model,
            micro_batches,
            optimizer,
            scheduler,
            max_grad_norm,
            autocast
        )
        steptimes.append(log["step_time_s"])
        if len(steptimes)<20:
            eta = sum(steptimes)/len(steptimes)
        else:
            eta = sum(steptimes[-20:])/20
        
        print(f"Step [{step}/{total_steps}] (ETA: {(total_steps-step)*eta/60}min): {log}\n")
        # save log line
        with open(log_path, "a") as file:
            file.write(json.dumps(log) + "\n")


        if (step + 1) % eval_every == 0:
            val_loss = val_step(model, val_loader, autocast)
            val_log = {"step": step +1, "val_loss": val_loss}
            print(f"[VALIDATION AT STEP {step +1}]: {val_loss}\n")
            with open(log_path, "a") as file:
                file.write(json.dumps(val_log) + "\n")

    print("Training concluded.")
    print("="*40)


def batches_on_tap(loader: DataLoader, sampler: DistributedSampler):
    # loops over the data again and again, with a new shuffle each epoch
    for epoch in itertools.count():
        sampler.set_epoch(epoch)   # each time the yield returns the end, we iterate over this loop and start a new epoch.
        yield from loader # loader gives batches of size micro_batch_size and collates them using our specified function

def get_sampler_loader(dataset, seed, micro_batch_size, fn):
    sampler = DistributedSampler(dataset, num_replicas=1, rank=0,
                                 shuffle=True, seed=seed, drop_last=True)
    loader = DataLoader(dataset, batch_size=micro_batch_size, sampler=sampler,
                        collate_fn=fn, drop_last=True)
    return sampler, loader

if __name__ == "__main__":
    
    if len(sys.argv) != 2:
            raise SystemExit("Usage: uv run python scripts/training_pipeline.py <trianing config path>")
    train_config = sys.argv[1]
    training_launcher(train_config)