#!/bin/bash
#SBATCH -c 8 # request two cores 
#SBATCH -p kisski-h100,kisski
#SBATCH -o log/GRPO-Qwen2_5-3b-bsz.out
#SBATCH -e log/error-GRPO-Qwen2_5-3b-bsz.out
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=grpo-qwen2_5-3b
#SBATCH --ntasks-per-node=1
#SBATCH -G H100:4

export VLLM_DISABLE_COMPILE_CACHE=1
source ~/.bashrc
conda activate prm_rlvr

bszs=(16 32 64 128 256 512)
rollout_sz=16
temp=1.0

for bsz in "${bszs[@]}";do
prompt_bsz=$(( bsz / rollout_sz ))
echo "Training with prompt batch size $prompt_bsz"

rm -rf ~/.cache
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=false\
    data.train_files=data/polaris/train.parquet \
    data.val_files=['data/math/test.parquet']\
    data.train_batch_size=${prompt_bsz} \
    data.max_prompt_length=1024 \
    data.max_response_length=3072\
    data.filter_overlong_prompts=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    data.truncation=right \
    actor_rollout_ref.model.path=models/Llama-3.2-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=${prompt_bsz} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=16\
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb']\
    trainer.project_name='RLVR-Peft' \
    trainer.experiment_name=Llama-3.2-3B-Instruct-polaris-GRPO--bsz${bsz}\
    reward_model.reward_manager=deepscaler \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=true \
    trainer.nnodes=1 \
    trainer.save_freq=64 \
    trainer.test_freq=64 \
    trainer.total_training_steps=512\
    trainer.total_epochs=15 $@
done