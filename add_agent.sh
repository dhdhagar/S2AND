#!/bin/bash -e

dataset=${1}  # "pubmed"
seed=${2}  # 1
model=${3:-"e2e"}  # Used as prefix and to pick up the right sweep file
sweep_id=${4}  # entity/project/id
n_agents=${5:-1}  # 1
gpu_name=${6:-"gypsum-1080ti"}  # "gypsum-1080ti"

for ((i = 1; i <= ${n_agents}; i++)); do
  JOB_DESC=${model}_${dataset}_sweep${seed}-${i} && JOB_NAME=${JOB_DESC}_$(date +%s) && \
  sbatch -J ${JOB_NAME} -e jobs/${JOB_NAME}.err -o jobs/${JOB_NAME}.log \
    --partition=${gpu_name} --gres=gpu:1 --mem=80G --time=12:00:00 \
    run_sbatch.sh e2e_scripts/train.py \
    --dataset="${dataset}" \
    --dataset_random_seed=${seed} \
    --pairwise_eval_clustering="both" \
    --skip_initial_eval \
    --silent \
    --wandb_sweep_params="wandb_configs/sweeps/${model}.json" \
    --wandb_tags="${model},${dataset},seed_${seed}" \
    --wandb_sweep_id="${sweep_id}"
  echo "    Logs: jobs/${JOB_NAME}.err"
done
