# to tstart a new session and have it persist beyond this shell:
# tmux new -s train

# to check the GPU and the driver
#nvidia-smi

# install uv, and make it available in this shell
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# get the code
git clone https://github.com/jord534/distributed-llm-training.git
cd distributed-llm-training
git log -1 --format=%H # this is the hash of the commit we cloned

# environment, and the model download folder
export HF_HOME=$HOME/hf_home
uv sync --locked

# reports folder and data cache
uv run python scripts/prepare_data.py configs/data_config/config.yaml

# train
uv run python scripts/training_pipeline.py configs/train/CHANGE_THIS.yaml