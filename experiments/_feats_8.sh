#!/bin/bash -e

dataset=${1:-"pubmed"}
n_seed_start=${2:-1}
n_seed_end=${3:-5}
model=${4:-"e2e"}  # Used as prefix and to pick up the right sweep file
gpu_name=${5:-"gypsum-1080ti"}
sweep_prefix=${6:-""}

for ((i = ${n_seed_start}; i <= ${n_seed_end}; i++)); do
  JOB_DESC=${model}_${dataset}_sweep${i} && JOB_NAME=${JOB_DESC}_$(date +%s) && \
  sbatch -J ${JOB_NAME} -e jobs/${JOB_NAME}.err -o jobs/${JOB_NAME}.log \
    --partition=${gpu_name} --gres=gpu:1 --mem=90G --time=12:00:00 \
    run_sbatch.sh e2e_scripts/train.py \
    --dataset="${dataset}" \
    --dataset_random_seed=${i} \
    --skip_initial_eval \
    --silent \
    --wandb_sweep_name="${sweep_prefix}_${model}_${dataset}_${i}" \
    --wandb_sweep_params="wandb_configs/sweeps/${model}.json" \
    --wandb_tags="${model},${dataset},seed_${i},${sweep_prefix}" \
    --keep_feat_idxs=0 --keep_feat_idxs=1 --keep_feat_idxs=2 \
    --keep_feat_idxs=3 --keep_feat_idxs=4 --keep_feat_idxs=5 \
    --keep_feat_idxs=14 --keep_feat_idxs=15
  echo "    Logs: jobs/${JOB_NAME}.err"
done
