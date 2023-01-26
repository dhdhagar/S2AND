#!/bin/bash -e

gpu_name=${1:-"gypsum-1080ti"}  # "gypsum-1080ti"
model="e2e"

dataset="pubmed"
for ((dataset_seed = 1; dataset_seed <= 5; dataset_seed++)); do
  JOB_DESC=icml_untrained_${model}_${dataset}_${dataset_seed} && JOB_NAME=${JOB_DESC}_$(date +%s) && \
  sbatch -J ${JOB_NAME} -e jobs/${JOB_NAME}.err -o jobs/${JOB_NAME}.log \
    --partition=${gpu_name} --gres=gpu:1 --mem=100G --time=12:00:00 \
    run_sbatch.sh e2e_scripts/train.py \
    --eval_only_split="test" \
    --dataset=${dataset} \
    --dataset_random_seed=${dataset_seed} \
    --silent \
    --wandb_tags="icml,untrained,${model},${dataset},seed_${dataset_seed}" \
    --save_block_metrics
  echo "    Logs: jobs/${JOB_NAME}.err"
done

dataset="qian"
for ((dataset_seed = 1; dataset_seed <= 5; dataset_seed++)); do
  JOB_DESC=icml_untrained_${model}_${dataset}_${dataset_seed} && JOB_NAME=${JOB_DESC}_$(date +%s) && \
  sbatch -J ${JOB_NAME} -e jobs/${JOB_NAME}.err -o jobs/${JOB_NAME}.log \
    --partition=${gpu_name} --gres=gpu:1 --mem=100G --time=12:00:00 \
    run_sbatch.sh e2e_scripts/train.py \
    --eval_only_split="test" \
    --dataset=${dataset} \
    --dataset_random_seed=${dataset_seed} \
    --silent \
    --wandb_tags="icml,untrained,${model},${dataset},seed_${dataset_seed}" \
    --save_block_metrics
  echo "    Logs: jobs/${JOB_NAME}.err"
done

dataset="zbmath"
for ((dataset_seed = 1; dataset_seed <= 5; dataset_seed++)); do
  JOB_DESC=icml_untrained_${model}_${dataset}_${dataset_seed} && JOB_NAME=${JOB_DESC}_$(date +%s) && \
  sbatch -J ${JOB_NAME} -e jobs/${JOB_NAME}.err -o jobs/${JOB_NAME}.log \
    --partition=${gpu_name} --gres=gpu:1 --mem=100G --time=12:00:00 \
    run_sbatch.sh e2e_scripts/train.py \
    --eval_only_split="test" \
    --dataset=${dataset} \
    --dataset_random_seed=${dataset_seed} \
    --silent \
    --wandb_tags="icml,untrained,${model},${dataset},seed_${dataset_seed}" \
    --save_block_metrics
  echo "    Logs: jobs/${JOB_NAME}.err"
done
