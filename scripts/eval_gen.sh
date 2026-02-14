
datasets=(math deepmath polaris)
# steps=(64 128 192 256 320 384 448 512)
steps=(512)

for ds in "${datasets[@]}";do
for step in "${steps[@]}";do
    # python eval/gen_vllm_test.py --model_path checkpoints/RLVR-Peft/Qwen2.5-3B-Instruct-${ds}-GRPO-temp1.0/global_step_${step}/actor/huggingface --n_gpus 4 --output_path gen_outputs/Qwen2.5-3B-Instruct/step${step}/full_${ds}_temp1.0_bsz1024_adamw
    python eval/gen_vllm_test.py --model_path models/Qwen2.5-3B-Instruct --n_gpus 4 --output_path gen_outputs/Qwen2.5-3B-Instruct/base

done
done
