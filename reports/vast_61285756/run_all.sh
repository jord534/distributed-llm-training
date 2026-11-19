export PATH="$HOME/.local/bin:$PATH" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/distributed-llm-training
OUT=reports/vast_61285756
for cfg in configs/train/3b_fsdp_80gb.yaml configs/train/3b_ddp_80gb.yaml; do
    name=$(basename $cfg .yaml)
    uv run torchrun --nproc_per_node=2 scripts/training_pipeline.py $cfg > $OUT/$name.out 2>&1
    echo "$name exit $?" >> $OUT/TRAINING_STATUS
done
touch $OUT/TRAINING_DONE
