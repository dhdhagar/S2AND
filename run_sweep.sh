#!/bin/bash -e

dataset=${1:-"pubmed"}
n_seeds=${2:-5}
model=${3:-"e2e"}  # Used as prefix and to pick up the right sweep file
gpu_name=${4:-"gypsum-1080ti"}

for ((i = 1; i <= ${n_seeds}; i++)); do
  JOB_DESC=${model}_${dataset}_sweep${i} && JOB_NAME=${JOB_DESC}_$(date +%s) && \
  sbatch -J ${JOB_NAME} -e jobs/${JOB_NAME}.err -o jobs/${JOB_NAME}.log \
    --partition=${gpu_name} --gres=gpu:1 --mem=80G --time=12:00:00 \
    run_sbatch.sh e2e_scripts/train.py \
    --dataset="${dataset}" \
    --dataset_random_seed=${i} \
    --pairwise_eval_clustering="both" \
    --skip_initial_eval \
    --silent \
    --wandb_sweep_name="${model}_${dataset}_${i}" \
    --wandb_sweep_params="wandb_configs/sweeps/${model}.json" \
    --wandb_tags="${model},${dataset},seed_${i}"
  echo "    Logs: jobs/${JOB_NAME}.err"
done
