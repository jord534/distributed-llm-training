import os
import torch
import datetime
from transformers import Mistral3ForConditionalGeneration
from torch.nn.parallel import DistributedDataParallel

from typing import Callable

def open_group(default_device : str = "cpu"):
    rank = os.environ.get("RANK", None)
    local_rank = os.environ.get("LOCAL_RANK", None)
    world_size = os.environ.get("WORLD_SIZE", None)

    if rank is None or local_rank is None or world_size is None:
        return {
                "rank": 0,
                "local_rank": 0,
                "world_size": 1,
                "device": torch.device(default_device),
            }
    rank = int(rank)
    local_rank = int(local_rank)
    world_size = int(world_size)

    if torch.cuda.is_available():
        backend = "nccl"
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
    else:
        backend = "gloo"
        device = "cpu"

    timeout = datetime.timedelta(minutes=10) # apparently it is the default for pytorch's nccl
    torch.distributed.init_process_group(backend = backend, timeout = timeout)

    return {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": torch.device(device),
    }


def close_group():
    if not torch.distributed.is_initialized():
        return None
    torch.distributed.destroy_process_group()


def sum_accross_gpus(val: int | float, device: torch.device):
    if not torch.distributed.is_initialized():
        return val
    tsr = torch.tensor(val, device=device)
    torch.distributed.all_reduce(tsr, op = torch.distributed.ReduceOp.SUM)
    return tsr.item()


def max_accross_gpus(val: int | float, device: torch.device):
    if not torch.distributed.is_initialized():
        return val
    tsr = torch.tensor(val, device = device)
    torch.distributed.all_reduce(tsr, op = torch.distributed.ReduceOp.MAX)
    return tsr.item()

def group_single_action(action: Callable, rank: int):
    if rank == 0:
        action()



def wrap_model_ddp(model: Mistral3ForConditionalGeneration, local_rank: int)-> DistributedDataParallel:
    return DistributedDataParallel(model, device_ids = [local_rank])
