# datasets=(math polaris deepmath)
datasets=(polaris)
steps=(64 128 192 256 320 384 448 512)
bszs=(1 2 4 8 16 32)
temps=(0.2 0.4 0.6 0.8 1.2)

for ds in "${datasets[@]}";do
for temp in "${temps[@]}";do
# for bsz in "${bszs[@]}";do
for step in "${steps[@]}";do
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/RLVR-Peft/Qwen2.5-3B-Instruct-${ds}-GRPO-temp${temp}/global_step_${step}/actor \
    --target_dir checkpoints/RLVR-Peft/Qwen2.5-3B-Instruct-${ds}-GRPO-temp${temp}/global_step_${step}/actor/huggingface 

done
done
done