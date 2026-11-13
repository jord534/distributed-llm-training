import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import Mistral3ForConditionalGeneration
import time
import math
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from llm_train.distributed import (
    sum_accross_gpus,
)
from contextlib import nullcontext

def set_seed(seed: int):
    # every source of randomness depends on the seed: model init, data order, dropout
    random.seed(seed) # python's core randomness
    np.random.seed(seed) # numpy core randomness
    torch.manual_seed(seed) # torch on cpu (also seeds the gpus)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) # every gpu, explicitly
    return None

def get_batch_from_examples(examples : list):
    tensor_list = []
    masks = []
    n_tokens = 0
    n_rows = len(examples)
    for ex in examples:
        ids = ex["input_ids"]
        n_tokens += len(ids)
        tensor_list.append(torch.tensor(ids, dtype=torch.int64))
        masks.append(torch.ones(len(ids), dtype = torch.int64))

    n_learned = n_tokens - n_rows # each row's first token is not learned
    # pad every tensor
    input_ids = pad_sequence(tensor_list, batch_first = True, padding_value=0, padding_side="right")
    attention_mask = pad_sequence(masks, batch_first=True, padding_side="right", padding_value=0)
    labels = pad_sequence(tensor_list, batch_first = True, padding_value=-100, padding_side="right")

    B, longest = input_ids.shape
    n_padding = B*longest - n_tokens
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "n_tokens": n_tokens,
        "n_padding": n_padding,
        "n_learned": n_learned,
    }

    
    

def get_loss(logits, labels, n_learned):
    # cross entropy of logits and labels
    
    # B, length, vocab_size: logits
    # B, length: labels

    # the last logits in dim = 1 does not have a label
    # the first label in dim = 1 does not have a logit
    # we drop both these
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]

    ## manual version
    # get softmax over the vocab for the whole batch:
    # log_probs = torch.nn.functional.log_softmax(logits, dim = -1)

    # # gather the log_prob of the labelled token:
    # labelled_log_probs = torch.gather(log_probs, dim = -1, index = labels)
    # todo

    ## compute the cross entropy directly:
    # we ignore elements whose labels are -100 (padding)
    # also the .cross_entropy expects the first dim to be rows of the second dim which is vocab size
    # so let's reshape by flattening
    B, l, v = logits.shape
    B, l = labels.shape
    logits = torch.reshape(logits, (B*l, v)).float() # logits.float() bumping precision to fp32 from bf16 to avoid loss
    labels = torch.reshape(labels, (B*l,))
    loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100, reduction="sum") / n_learned
    # logits.float() to up precision to fp32 insterad of bf16 to avoid loss due to precision
    return loss


def get_optimizer_and_lr_scheduler(params, lr, weight_decay, warmup_step, total_steps):
    active_parms = [p for p in params if p.requires_grad]
    optimizer =  torch.optim.AdamW(
        params=active_parms,
        lr = lr,
        weight_decay=weight_decay,
    )
    multiplier = lambda step: (step<= warmup_step)*(step/warmup_step) + (step>warmup_step)* 0.5 * (1+math.cos(math.pi*(step- warmup_step)/(total_steps - warmup_step)))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)

    return optimizer, lr_scheduler


def training_step(
        model: Mistral3ForConditionalGeneration | DistributedDataParallel,
        micro_batches: list, 
        optimizer: torch.optim.AdamW, 
        scheduler: torch.optim.lr_scheduler.LambdaLR, 
        max_grad_norm: float, 
        use_autocast: bool, # to enable mixed precision for a quick forward pass and accurate loss and grad
        world_size: int,
        fsdp: bool,
    ):
    
    model.train()

    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device) 
        torch.cuda.synchronize() # wait for previous GPU work to finish

    start = time.perf_counter()

    optimizer.zero_grad()

    learned_tokens = sum([batch["n_learned"] for batch in micro_batches])
    total_tokens = sum([batch["n_tokens"] for batch in micro_batches])
    total_loss = 0
    if world_size > 1:
        learned_tokens = sum_accross_gpus(learned_tokens, model.device) # for DDP we will compute the average loss over all tokens involved accross GPUs

    for i, batch in enumerate(micro_batches):
        model_sync_condition = get_context(model, i, len(micro_batches), world_size, fsdp) # handle gradient syncing behaviour
        with model_sync_condition:
            labels = batch["labels"].to(model.device)
            input_ids = batch["input_ids"].to(model.device)
            attn_mask = batch["attention_mask"].to(model.device)
            with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16, enabled = use_autocast): # mixed precision
                model_output = model(input_ids=input_ids, attention_mask = attn_mask) # forward pass
            logits = model_output.logits
            loss = get_loss(logits, labels, learned_tokens)
            total_loss += loss.item() 
            (loss*world_size).backward() # the backward pass should account for the whole loss from the whole world
    
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    # checking the grad and the loss are < inf:
    if not torch.isfinite(grad_norm) :
        raise ValueError(f"The gradient exploded: {grad_norm.item()}.")
    if not math.isfinite(total_loss):
        raise ValueError(f"The loss has exploded: {total_loss}.")

    lr = optimizer.param_groups[0]["lr"]

    optimizer.step()
    scheduler.step()
    step = scheduler.last_epoch
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    step_time = time.perf_counter() - start

    peak_mem = torch.cuda.max_memory_allocated(model.device) / 1e9 if model.device.type == "cuda" else None

    return {
        "step": step,
        "loss": total_loss,
        "grad_norm": grad_norm.item(),
        "lr": lr,
        "tokens": total_tokens,
        "learned_tokens": learned_tokens,
        "step_time_s": step_time, # diagnostics
        "peak_memory_gb": peak_mem, # diagnostics
    }


def get_context(model: Mistral3ForConditionalGeneration | DistributedDataParallel,
                i: int,
                n_micro_batches: int,
                world_size: int,
                fsdp: bool
    ):
    if world_size == 1 or i == n_micro_batches - 1:  # either no parallel or final micro batch (where gradients need syncing between GPUs)
        return nullcontext()
    if i < n_micro_batches - 1: # either DDP or FSDP with gradient syncing left for the final micro batch only to prevent repeated syncing
        if fsdp:
            return model.set_requires_gradient_sync(False)
        return model.no_sync() # ddp case

            

def val_step(
        model: Mistral3ForConditionalGeneration | DistributedDataParallel,
        micro_batches: DataLoader, 
        use_autocast: bool, # to enable mixed precision for a quick forward pass and accurate loss and grad
    ):
    model.eval() # this disables dropouts and prompts batchnorm to use internal statistics
    loss_sum = 0
    with torch.no_grad():
        learned_tokens = 0
        for batch in micro_batches:
            learned_tokens += batch["n_learned"] # counting inside the loop so that the DataLoader object is only run through once per val step
            labels = batch["labels"].to(model.device)
            input_ids = batch["input_ids"].to(model.device)
            attn_mask = batch["attention_mask"].to(model.device)
            with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16, enabled = use_autocast): # mixed precision
                model_output = model(input_ids=input_ids, attention_mask = attn_mask) # forward pass
            logits = model_output.logits
            loss_sum += get_loss(logits, labels, 1).item()

    return loss_sum / learned_tokens