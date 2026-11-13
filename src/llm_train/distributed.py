import os
import torch
import datetime
from transformers import Mistral3ForConditionalGeneration
from torch.nn.parallel import DistributedDataParallel
import random
import numpy
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
    if model.device != torch.device("cpu"):
        return DistributedDataParallel(model, device_ids = [local_rank])
    return DistributedDataParallel(model)


def get_groupwide_random_states(world_size):
    states = {
        "python": random.getstate(),
        "numpy": numpy.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }
    if not torch.distributed.is_initialized():
        return [states]
    
    output = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(output, states)
    return output

def fsdp_split(
        model : Mistral3ForConditionalGeneration,
        setup: dict,
        param_dtype: torch.dtype = torch.bfloat16, # used for forward
        reduce_dtype: torch.dtype = torch.float32 # used for gradient updates
    ):

    # setup is a dict output by open_group {"rank", "local_rank", "world_size", "device"}
    # the idea is to assign shards to each device
    # treating decoder blocks first then the whole model

    # let's build a mesh of devices
    mesh = torch.distributed.device_mesh.init_device_mesh(
        device_type=setup["device"].type,
        mesh_shape = (setup["world_size"],)
    )

    # let's define the muxed precision policy we want our modules to use
    mp = torch.distributed.fsdp.MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    # fetch all decoder blocks:
    for layer in model.model.language_model.layers:
        # each layer is a Ministral3DecoderLayer with :
        # [self_attn (Ministral3Attention)  + mlp (Ministral3MLP) 
        # + input_layernorm (Ministral3RMSNorm) + post_attention_layernorm (Ministral3RMSNorm)]
        #  in the dev toy model there are 2 hidden layers
        # the structure can be observed with print(model) in pdb
        torch.distributed.fsdp.fully_shard(layer, mp_policy = mp, mesh = mesh)

    # let's shard the vision tower and multimodal part as well, so that their slicing is indepent from the root model
    # and since the root model is called and not the vison and multimodel bits, their slices won't take up memory on the gpus
    torch.distributed.fsdp.fully_shard(model.model.vision_tower, mp_policy = mp, mesh = mesh)
    torch.distributed.fsdp.fully_shard(model.model.multi_modal_projector, mp_policy = mp, mesh = mesh)

    # now the root model :
    torch.distributed.fsdp.fully_shard(model, mp_policy = mp, mesh = mesh)
    # we do layers first then whole model to enable the sharding at the layer level
    # otherwise it assimes as layer isn't to be sharded

    return None

def wait_for_all():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        