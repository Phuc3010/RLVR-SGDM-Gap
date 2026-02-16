#!/bin/bash
#SBATCH -c 8 # request two cores 
#SBATCH -p kisski-h100,kisski
#SBATCH -o log/eval_gen.out
#SBATCH -e log/error-eval_gen.out
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=eval_gen
#SBATCH --ntasks-per-node=1
#SBATCH -G H100:4

export VLLM_DISABLE_COMPILE_CACHE=1
source ~/.bashrc
conda activate prm_rlvr

datasets=(polaris)
# steps=(64 128 192 256 320 384 448 512)
bszs=(256 512)
steps=(512)

for ds in "${datasets[@]}";do
for bsz in "${bszs[@]}";do
for step in "${steps[@]}";do
    python eval/gen_vllm_test.py --model_path checkpoints/RLVR-Peft/Qwen2.5-3B-Instruct-${ds}-GRPO-temp1.0-bsz${bsz}/global_step_${step}/actor/huggingface --n_gpus 4 --output_path gen_outputs/Qwen2.5-3B-Instruct/step${step}/full_${ds}_bsz${bsz}_step512_adamw
    # python eval/gen_vllm_test.py --model_path models/Qwen2.5-3B-Instruct --n_gpus 4 --output_path gen_outputs/Qwen2.5-3B-Instruct/base

done
done
done
