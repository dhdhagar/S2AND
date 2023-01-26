#!/bin/bash -e

dataset=${1}  # "pubmed"
dataset_seed=${2}  # 1
model=${3}  # "e2e"
run_id=${4}  # entity/project/id
n_runs=${5:-1}  # 1
gpu_name=${6:-"gypsum-1080ti"}  # "gypsum-1080ti"

for ((run_seed = 1; run_seed <= ${n_runs}; run_seed++)); do
  JOB_DESC=icml_${model}_${dataset}_${dataset_seed}-${run_seed} && JOB_NAME=${JOB_DESC}_$(date +%s) && \
  sbatch -J ${JOB_NAME} -e jobs/${JOB_NAME}.err -o jobs/${JOB_NAME}.log \
    --partition=${gpu_name} --gres=gpu:1 --mem=100G --time=12:00:00 \
    run_sbatch.sh e2e_scripts/train.py \
    --pairwise_eval_clustering="both" \
    --skip_initial_eval \
    --silent \
    --load_hyp_from_wandb_run="${run_id}" \
    --run_random_seed=${run_seed} \
    --wandb_tags="icml,${model},${dataset},seed_${dataset_seed},run_seed_${run_seed}" \
    --save_model --save_block_metrics
  echo "    Logs: jobs/${JOB_NAME}.err"
done
