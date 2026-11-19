#!/bin/bash
set -e   # stop at the first error: never destroy after a failed copy

KEY=~/.ssh/id_ed25519
PORT=40426
HOST=root@216.243.220.248
ID=61101538
REMOTE=/workspace/distributed-llm-training
SSH="ssh -i $KEY -p $PORT $HOST"
RSYNC_SSH="ssh -i $KEY -p $PORT"

# done = the DDP run has started at some point, and no training process is left
while true; do
    if ! $SSH true; then echo "$(date +%H:%M) cannot reach instance, stopping"; exit 1; fi
    if $SSH "test -f $REMOTE/ddp_mb1.out && ! pgrep -f '[t]raining_pipeline.py'"; then break; fi
    echo "$(date +%H:%M) still training"; sleep 300
done

# pull runs, console outputs and the configs edited on the machine
rsync -av -e "$RSYNC_SSH" $HOST:$REMOTE/artifacts/runs/ artifacts/runs/
mkdir -p reports/raw
rsync -av -e "$RSYNC_SSH" "$HOST:$REMOTE/*.out" reports/raw/
rsync -av -e "$RSYNC_SSH" $HOST:$REMOTE/configs/train/ configs/train/

# destroy only if both logs arrived
test -s artifacts/runs/3b_fsdp_2gpu_2ep_s1337/log.jsonl
test -s artifacts/runs/3b_ddp_2gpu_2ep_s1337/log.jsonl
vastai destroy instance $ID
echo "results copied, instance destroyed"

# commit the configs, only on the fsdp branch
if [ "$(git branch --show-current)" = "fsdp" ]; then
    git add configs/train/
    git commit -m "3b configs used on vast: large cache, 2ep runs" || echo "nothing to commit"
    git push
else
    echo "not on fsdp: configs copied but not committed"
fi