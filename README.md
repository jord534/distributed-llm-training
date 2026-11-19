# distributed LLM training

fine-tunes Ministral 3B on ultrachat with DDP or FSDP, from one yaml per run

latest result: [DDP vs FSDP on 2x A100 80GB](reports/3b_ddp_vs_fsdp_80gb_s1337.md)

## Setup

uses [uv](https://docs.astral.sh/uv/). on linux it pulls the CUDA build of torch, on mac the CPU one

```bash
uv sync
```

## 1. Prepare the data

tokenizes a sample of ultrachat and caches it under `artifacts/data/`

```bash
uv run python scripts/prepare_data.py configs/data_config/config_large.yaml
```

it prints the cache folder, e.g. `artifacts/data/ultrachat_200k_train_sft_c6230c6b`. the suffix is a hash of the data config, so the same config always gives the same folder. that's the path to put in the train yaml (`data.cache`)

## 2. Train

one gpu or cpu:

```bash
uv run python scripts/training_pipeline.py configs/train/dev_single.yaml
```

several gpus on one machine (one process per gpu):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run torchrun --nproc_per_node=2 scripts/training_pipeline.py configs/train/3b_fsdp_80gb.yaml
```

everything goes to `artifacts/runs/<run_name>/`: `log.jsonl` (one line per step), the config, the environment and the param count

running the same `run_name` again resumes from its latest checkpoint. if the run has no checkpoints it stops with an error, so change the name to start over

no gpu? the `dev_*` configs use a tiny random model on cpu (`dev_nano_fsdp.yaml` even runs FSDP on cpu, launched with `torchrun`)

## 3. Plot

```bash
uv run python scripts/plot_run.py 3b_fsdp_80gb_s1337
```

writes loss, grad norm, lr, step time and peak memory to `reports/plots/<run_name>/`

## On rented gpus (vast.ai)

```bash
scripts/vast_run.sh
```

rents 2x A100 80GB, clones the branch, prepares the data, runs every config in `CONFIGS` one after the other, syncs results back and destroys the instance. it also destroys after `FAILSAFE_MIN` (59 min) no matter what. settings are at the top of the script. the configs need to be pushed first, since the instance clones from github

## Example config

`configs/train/3b_fsdp_80gb.yaml`, the FSDP run from the report:

```yaml
run_name: 3b_fsdp_80gb_s1337      # folder name in artifacts/runs/, reusing it resumes
seed: 1337
device: cuda                      # cuda or cpu

model:
  kind: pretrained                # pretrained = real weights, dev = tiny random model (needs dev_dims)
  model_id: mistralai/Ministral-3-3B-Base-2512
  revision: 6f9c4b12a95b139af68670a6713616b757923735   # pinned, must match the data config's tokenizer
  dtype: float32                  # master weights stay fp32

data:
  cache: artifacts/data/ultrachat_200k_train_sft_c6230c6b   # printed by prepare_data.py
  max_train_examples: null        # null = all of them, a number = a smaller subset

train:
  micro_batch_size: 1             # conversations per forward pass, per gpu
  accumulation_steps: 32          # micro batches per optimizer step
                                  # -> 1 x 32 x 2 gpus = 64 conversations per step
  learning_rate: 2.0e-5
  weight_decay: 0.1
  warmup_steps: 7                 # linear warmup, then cosine decay to 0 at total_steps
  total_steps: 70                 # optimizer steps (49,000 conversations / 64 = ~766 per epoch)
  max_grad_norm: 1.0              # gradient clipping
  autocast: false                 # bf16 autocast: true for DDP, false for FSDP (it casts itself)
  activation_checkpointing: true  # recompute activations in backward: less memory, slower
  eval_every: 35                  # validation loss every n steps
  checkpoint_every: null          # null = no checkpoints, n = save every n steps
  fsdp: true                      # false = DDP (or a single process)
  fsdp_param_dtype: bfloat16      # weights are gathered and used in bf16
  fsdp_reduce_dtype: float32      # gradients are averaged in fp32
```

the DDP version (`3b_ddp_80gb.yaml`) is the same with `autocast: true`, `fsdp: false` and no `fsdp_*` lines
