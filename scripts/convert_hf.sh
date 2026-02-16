# datasets=(math polaris deepmath)
datasets=(polaris)
# steps=(64 128 192 256 320 384 448 512)
steps=(512)
# bszs=(16 64 128 256)
bszs=(256)
temp=1.0

for ds in "${datasets[@]}";do
for bsz in "${bszs[@]}";do
for step in "${steps[@]}";do
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/RLVR-Peft/Qwen2.5-3B-Instruct-${ds}-GRPO-temp${temp}-bsz${bsz}/global_step_${step}/actor \
    --target_dir checkpoints/RLVR-Peft/Qwen2.5-3B-Instruct-${ds}-GRPO-temp${temp}-bsz${bsz}/global_step_${step}/actor/huggingface 

done
done
done