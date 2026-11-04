import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

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
    multiplier = lambda step: (step<= warmup_step)*(step/warmup_step) + (step>warmup_step)* 0.5 * (1+np.cos(np.pi*(step- warmup_step)/(total_steps - warmup_step)))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)

    return optimizer, lr_scheduler


def training_step(model, batch, optimizer, scheduler, clip_grad, autocast: bool = False):
    for b in batch:
        model.forward(b)
        
