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

from typing import Callable, Tuple

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

from llm_train.distributed import (
    open_group,
    close_group,
    wrap_model_ddp,
    sum_accross_gpus,
    max_accross_gpus,
    group_single_action
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

def instantiate_model(config, world_size:int, local_rank:int) -> Mistral3ForConditionalGeneration:
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
        model, lm_trainable = create_dev_model(model_id, revision, device, text_dims, vision_dims, dtype)
    else:
        model, lm_trainable = load_model(model_id, revision, device, dtype)

    activation_checkpointing = config["train"]["activation_checkpointing"]
    if activation_checkpointing:
        model.gradient_checkpointing_enable() # this needs to happen before wrapping

    if world_size > 1:
        model = wrap_model_ddp(model, local_rank)
    
    return model, lm_trainable

def fetch_optimizer_scheduler(config, model: Mistral3ForConditionalGeneration):
    train = config["train"]
    lr = train["learning_rate"]
    warmup_steps = train["warmup_steps"]
    total_steps = train["total_steps"]
    weight_decay = train["weight_decay"]

    optimizer, scheduler = get_optimizer_and_lr_scheduler(model.parameters(), lr, weight_decay, warmup_steps, total_steps)
    return optimizer, scheduler, total_steps

def write_trainable_params_number(this_run_dir, lm_trainable):
    with open(this_run_dir / "lm_trainable.json", "w") as file:
        json.dump(lm_trainable, file)


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
    # Get rank and world_size from the torch run group:
    group = open_group(torch.device(config["device"]))
    rank = group["rank"]
    local_rank = group["local_rank"]
    world_size = group["world_size"]

    perm_run_dir = run_dir / run_name

    if rank == 0:
        if perm_run_dir.exists():
            raise FileExistsError(f"Run already exists: {perm_run_dir}")

    with tempfile.TemporaryDirectory() as tmp_dir :
        tmp_dir = Path(tmp_dir)
        this_run_dir = tmp_dir / run_name
        
        group_single_action(lambda: copy_audit_and_config(this_run_dir, train_cfg), rank)

        # set seed
        print("Setting seed...\n")
        seed = config["seed"]
        set_seed(seed)

        # instantiate model
        print("Instantiate model... \n")
        model, lm_trainable = instantiate_model(config, world_size, local_rank)

        # Write the number of trainable LM parameters down:
        group_single_action(lambda : write_trainable_params_number(this_run_dir, lm_trainable), rank)

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
        eval_every = train["eval_every"]
        micro_batch_size = train["micro_batch_size"]
        accumulation_steps = train["accumulation_steps"]

        # copy all from temp dir to actual dir
        group_single_action(lambda: shutil.move(str(this_run_dir), str(perm_run_dir)))

    log_path = perm_run_dir / "log.jsonl"

    print("Getting sampler and loader ready to serve batches... \n")
    train_sampler, train_loader = get_sampler_loader(
        train_set, 
        seed, 
        micro_batch_size, 
        get_batch_from_examples, 
        rank, 
        world_size
    )
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
            autocast,
            world_size
        )
        # should we have DDP we aggregate the loss from all of them to log it
        log["loss"] = sum_accross_gpus(log["loss"], model.device)
        log["tokens"] = sum_accross_gpus(log["tokens"], model.device)
        if log["peak_memory_gb"] is not None:
            log["peak_memory_gb"] = max_accross_gpus(log["peak_memory_gb"], model.device)

        # we need to find the max of all processes step times for the ETA
        steptimes.append(
            max_accross_gpus(log["step_time_s"], model.device)
        )

        group_single_action(lambda: write_log(log, log_path, total_steps, step, steptimes), rank) # update the .jsonl and the shell on just one GPU (it aggregated all metrics)

        if (step + 1) % eval_every == 0:
            val_loss = val_step(model, val_loader, autocast)
            val_log = {"step": step +1, "val_loss": val_loss}
            group_single_action(
                lambda: log_val(val_log, log_path, val_loss, step)
            )
            
            

    close_group() # closing our torchrun group
    print("Training concluded.")
    print("="*40)


def write_log(log, log_path, total_steps, step, steptimes):
    if len(steptimes)<20:
        steptime_ravg = sum(steptimes)/len(steptimes)
    else:
        steptime_ravg = sum(steptimes[-20:])/20

    eta = (total_steps-step)*steptime_ravg # total seconds
    eta_min = eta // 60
    eta_secs = int((eta - eta_min)*60)

    print(f"Step [{step}/{total_steps}] (ETA: {eta_min}min{eta_secs}s): {log}\n")
    
    # save log line
    with open(log_path, "a") as file:
        file.write(json.dumps(log) + "\n")

def log_val(val_log, log_path, val_loss, step):
    print(f"[VALIDATION AT STEP {step + 1}]: {val_loss}\n")
    with open(log_path, "a") as file:
        file.write(json.dumps(val_log) + "\n")


def batches_on_tap(loader: DataLoader, sampler: DistributedSampler):
    # loops over the data again and again, with a new shuffle each epoch
    for epoch in itertools.count():
        sampler.set_epoch(epoch)   # each time the yield returns the end, we iterate over this loop and start a new epoch.
        yield from loader # loader gives batches of size micro_batch_size and collates them using our specified function

def get_sampler_loader(dataset: Dataset, seed: int, micro_batch_size: int, fn: Callable, rank: int, world_size:int) -> Tuple[DistributedSampler, DataLoader]:
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=True, seed=seed, drop_last=True)
    loader = DataLoader(dataset, batch_size=micro_batch_size, sampler=sampler,
                        collate_fn=fn, drop_last=True)
    return sampler, loader

if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        raise SystemExit("Usage: uv run python scripts/training_pipeline.py <trianing config path>")
    train_config = sys.argv[1]
    training_launcher(train_config)