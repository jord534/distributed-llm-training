#!/usr/bin/env bash
# rents 2 sxm gpus on vast.ai, runs both trainings one after the other, syncs everything new to the mac, destroys the instance
# destroys when both trainings are done, or at FAILSAFE_MIN after creation whatever the state; always syncs first
# usage, from the repo root on the mac: scripts/vast_run.sh
# before: configs pushed to origin/BRANCH (the instance clones it)
# ctrl-c = sync + destroy
set -uo pipefail

#### SETTINGS
REPO_URL="https://github.com/jord534/distributed-llm-training.git"
BRANCH="fsdp"
CONFIGS="configs/train/3b_fsdp_80gb.yaml configs/train/3b_ddp_80gb.yaml"
DATA_CONFIG="configs/data_config/config_large.yaml"
MODEL_ID="mistralai/Ministral-3-3B-Base-2512"
MODEL_REVISION="6f9c4b12a95b139af68670a6713616b757923735"
QUERY="${QUERY:-num_gpus=2 gpu_name=A100_SXM4 gpu_ram>=79 disk_space>=150 direct_port_count>=1 reliability>0.98 verified=true rentable=true}"
OFFER="${OFFER:-}"  # set it to rent a specific offer id (e.g. the same machine as last time), skips the search
IMAGE="${IMAGE:-nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04}"
DISK=150
KEY="${KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="/root/distributed-llm-training"
FAILSAFE_MIN="${FAILSAFE_MIN:-59}"
BOOT_TIMEOUT_MIN=15
POLL_S="${POLL_S:-120}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
####

#### LOGGING
mkdir -p "$LOCAL_DIR/reports"
LOG="$LOCAL_DIR/reports/vast_run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
say() { echo "[$(date +%H:%M:%S)] $*"; }
####

#### STATE in files, so the failsafe (a separate process) sees what the main script learns later
STATE="$(mktemp -d)"
MAIN_PID=$$
SSH_OPTS="-i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$STATE/known_hosts -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o LogLevel=ERROR"
get() { cat "$STATE/$1" 2>/dev/null; }
json() { python3 -c "import json,sys; d=json.load(sys.stdin); print($1)"; }
remote() { ssh $SSH_OPTS -p "$(get port)" "root@$(get host)" "$@"; }
####

#### SYNC: runs and this instance's report folder updated, every other new file copied, existing mac files never replaced
sync_back() {
    local rsh src ok=0
    rsh="ssh $SSH_OPTS -p $(get port)"
    src="root@$(get host):$REMOTE_DIR"
    mkdir -p "$LOCAL_DIR/artifacts/runs" "$LOCAL_DIR/$(get out)"
    rsync -az -e "$rsh" --exclude=checkpoints/ "$src/artifacts/runs/" "$LOCAL_DIR/artifacts/runs/" || ok=1
    rsync -az -e "$rsh" "$src/$(get out)/" "$LOCAL_DIR/$(get out)/" || ok=1
    rsync -az -e "$rsh" --ignore-existing --exclude=checkpoints/ \
        --exclude=.git/ --exclude=.venv/ --exclude=artifacts/data/ --exclude=artifacts/hf_cache/ --exclude=__pycache__/ \
        "$src/" "$LOCAL_DIR/" || ok=1
    return $ok
}
####

#### DESTROY, checked until the instance is gone
destroy() {
    local id ids
    id="$(get id)"
    [ -n "$id" ] || return 0
    for try in 1 2 3 4 5 6; do
        vastai destroy instance "$id" -y >/dev/null 2>&1
        sleep 5
        ids="$(vastai show instances --raw 2>/dev/null | json "' '.join(str(i['id']) for i in d) or 'none'" 2>/dev/null)"
        if [ -n "$ids" ] && ! echo " $ids " | grep -q " $id "; then
            say "instance $id destroyed"
            return 0
        fi
        say "instance $id still listed, retry $try"
    done
    say "!!! COULD NOT DESTROY INSTANCE $id: destroy it on cloud.vast.ai NOW"
    return 1
}
####

#### FINISH: sync then destroy, runs once (mkdir is atomic: the first caller gets the lock)
finish() {
    mkdir "$STATE/lock" 2>/dev/null || return 1
    if [ -n "$(get host)" ]; then
        say "syncing to the mac"
        for try in 1 2 3; do
            sync_back && break
            say "sync failed, retry $try"
            sleep 10
        done
    fi
    destroy
    say "log: $LOG"
    return 0
}
stop_failsafe() { kill "$(get failsafe)" 2>/dev/null; }
trap 'finish; stop_failsafe; exit 130' INT TERM
trap 'finish; stop_failsafe' EXIT
####

#### CHECKS ON THE MAC
[ -f "$KEY" ] || { echo "ssh key $KEY not found"; exit 1; }
git -C "$LOCAL_DIR" fetch -q origin "$BRANCH" || { echo "cannot fetch origin/$BRANCH"; exit 1; }
for cfg in $CONFIGS $DATA_CONFIG; do
    git -C "$LOCAL_DIR" cat-file -e "origin/$BRANCH:$cfg" 2>/dev/null || { echo "$cfg is not pushed to origin/$BRANCH"; exit 1; }
done
caffeinate -i -w $MAIN_PID &  # mac stays awake while this runs (plugged in, lid open)
####

#### RENT: OFFER if set, else the 5 cheapest matching offers in turn (an offer can be taken in the meantime)
if [ -z "$OFFER" ]; then
    say "searching: $QUERY"
    OFFER="$(vastai search offers "$QUERY" -o 'dph_total' --storage $DISK --raw 2>/dev/null | json "' '.join(str(o['id']) for o in d[:5])")"
    [ -n "$OFFER" ] || { say "no offer matches"; exit 1; }
fi
for offer in $OFFER; do
    out="$(vastai create instance "$offer" --image "$IMAGE" --disk $DISK --ssh --direct --raw 2>&1)"
    id="$(echo "$out" | json "d.get('new_contract') or ''" 2>/dev/null)"
    if [ -n "$id" ]; then echo "$id" > "$STATE/id"; echo "reports/vast_$id" > "$STATE/out"; break; fi
    say "offer $offer failed: $out"
done
[ -n "$(get id)" ] || { say "could not rent"; exit 1; }
START=$(date +%s)
minutes() { echo $(( ($(date +%s) - START) / 60 )); }
say "instance $(get id) created from offer $offer, failsafe at $(date -r $((START + FAILSAFE_MIN * 60)) +%H:%M)"
####

#### FAILSAFE: separate process, at FAILSAFE_MIN -> sync + destroy, then stops the main script
(
    trap - INT EXIT
    trap 'kill $nap 2>/dev/null; exit 0' TERM  # the main script kills it when it finishes first
    sleep $((FAILSAFE_MIN * 60)) >/dev/null 2>&1 &
    nap=$!
    wait $nap
    say "!!! FAILSAFE at $FAILSAFE_MIN min: sync + destroy"
    finish && kill -TERM $MAIN_PID
) &
echo $! > "$STATE/failsafe"
####

#### WAIT FOR THE INSTANCE, THEN SSH
while :; do
    status="$(vastai show instance "$(get id)" --raw 2>/dev/null | json "d.get('actual_status')" 2>/dev/null)"
    [ "$status" = running ] && break
    [ "$(minutes)" -ge $BOOT_TIMEOUT_MIN ] && { say "not running after $BOOT_TIMEOUT_MIN min ($status)"; exit 1; }
    say "status: ${status:-unknown}"
    sleep 15 & wait $!
done
url="$(vastai ssh-url "$(get id)" 2>/dev/null)"  # ssh://root@HOST:PORT
url="${url#ssh://}"; url="${url#root@}"
echo "${url%:*}" > "$STATE/host"
echo "${url##*:}" > "$STATE/port"
say "to watch: ssh -i $KEY -p $(get port) root@$(get host) then tmux attach -t train"
until remote true 2>/dev/null; do
    [ "$(minutes)" -ge $BOOT_TIMEOUT_MIN ] && { say "ssh not reachable after $BOOT_TIMEOUT_MIN min"; exit 1; }
    sleep 10 & wait $!
done
####

#### SET UP THE INSTANCE, stops at the first error (then finish destroys)
say "setting up"
remote bash -s -- "$REPO_URL" "$BRANCH" "$REMOTE_DIR" "$DATA_CONFIG" "$MODEL_ID" "$MODEL_REVISION" "${HF_TOKEN:-}" "$CONFIGS" "$(get out)" <<'SETUP'
set -euo pipefail
REPO_URL=$1; BRANCH=$2; REMOTE_DIR=$3; DATA_CONFIG=$4; MODEL_ID=$5; MODEL_REVISION=$6; HF_TOKEN=$7; CONFIGS=$8; OUT=$9
export PATH="$HOME/.local/bin:$PATH"
[ -n "$HF_TOKEN" ] && export HF_TOKEN

# tools the image may lack: git to clone, rsync to sync back, tmux to survive ssh drops, curl for uv
missing=""
for tool in git rsync tmux curl; do command -v $tool >/dev/null || missing="$missing $tool"; done
[ -z "$missing" ] || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $missing ca-certificates >/dev/null; }
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null

git clone -q --branch "$BRANCH" "$REPO_URL" "$REMOTE_DIR"
cd "$REMOTE_DIR"
mkdir -p "$OUT" artifacts/runs
git rev-parse HEAD > "$OUT/commit.txt"
uv sync --locked -q
uv run python -c "import torch; assert torch.cuda.device_count() == 2; print(torch.__version__, torch.cuda.get_device_name(0))"
{ nvidia-smi; nvidia-smi topo -m; } > "$OUT/nvidia_smi.txt"

uv run python scripts/prepare_data.py "$DATA_CONFIG"
for cfg in $CONFIGS; do
    cache=$(sed -n 's/^ *cache: *//p' "$cfg")
    [ -d "$cache" ] || { echo "$cfg wants $cache, prepare_data made something else"; exit 1; }
done
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_ID', revision='$MODEL_REVISION')" >/dev/null
SETUP
[ $? -eq 0 ] || { say "setup failed"; exit 1; }
say "setup done at +$(minutes) min"
####

#### TRAIN: both runs one after the other in tmux (survives ssh drops); TRAINING_DONE marks the end
remote bash -s -- "$REMOTE_DIR" "$CONFIGS" "$(get out)" <<'LAUNCH'
set -euo pipefail
REMOTE_DIR=$1; CONFIGS=$2; OUT=$3
cd "$REMOTE_DIR"
cat > "$OUT/run_all.sh" <<EOF
export PATH="\$HOME/.local/bin:\$PATH" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for cfg in $CONFIGS; do
    name=\$(basename "\$cfg" .yaml)
    uv run torchrun --nproc_per_node=2 scripts/training_pipeline.py "\$cfg" > "$OUT/\$name.out" 2>&1
    echo "\$name exit \$?" >> "$OUT/TRAINING_STATUS"
done
touch "$OUT/TRAINING_DONE"
EOF
tmux new -d -s train "bash $OUT/run_all.sh"
LAUNCH
say "training started"
####

#### WATCH: sync every poll; stop when done, or when ssh fails 5 polls in a row
fails=0
while :; do
    sleep "$POLL_S" & wait $!
    if flag="$(remote "test -f $REMOTE_DIR/$(get out)/TRAINING_DONE && echo done || echo running" 2>/dev/null)"; then
        fails=0
        [ "$flag" = done ] && { say "both trainings done"; break; }
        last="$(remote "cd $REMOTE_DIR && ls -t artifacts/runs/*/log.jsonl 2>/dev/null | head -1 | xargs -r tail -n1" 2>/dev/null)"
        say "+$(minutes) min | ${last:0:200}"
        sync_back >/dev/null 2>&1
    else
        fails=$((fails + 1))
        say "ssh failed ($fails/5)"
        [ $fails -ge 5 ] && { say "instance unreachable"; break; }
    fi
done
####

#### END
remote "cat $REMOTE_DIR/$(get out)/TRAINING_STATUS" 2>/dev/null
finish
stop_failsafe
say "done in $(minutes) min"
####