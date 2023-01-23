#!/bin/bash -e

dataset=${1:-"pubmed"}
n_seeds=${2:-5}
gpu_name=${3:-"gypsum-1080ti"}

for ((i = 1; i <= ${n_seeds}; i++)); do
  JOB_DESC=${dataset}_sweep${i} && JOB_NAME=${JOB_DESC}_$(date +%s) && \
  sbatch -J ${JOB_NAME} -e jobs/${JOB_NAME}.err -o jobs/${JOB_NAME}.log \
    --partition=${gpu_name} --gres=gpu:1 --mem=80G --time=12:00:00 \
    run_sbatch.sh e2e_scripts/train.py \
    --dataset="${dataset}" \
    --dataset_random_seed=${i} \
    --wandb_sweep_name="main_${dataset}_${i}" \
    --wandb_sweep_params="wandb_configs/sweeps/e2e_main.json" \
    --skip_initial_eval --sdp_eps=1e-1 --silent
  echo "    Logs: jobs/${JOB_NAME}.err"
done
