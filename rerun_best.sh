#!/bin/bash -e

entity="dhdhagar"
project="prob-ent-resolution"
run_id=${1}
gpu_name=${2:-"gypsum-1080ti"}
run_tag=${3:-"icml_rebut_best"}

JOB_DESC=rerun_${run_id} && JOB_NAME=${JOB_DESC}_$(date +%s) && \
  sbatch -J ${JOB_NAME} -e jobs/${JOB_NAME}.err -o jobs/${JOB_NAME}.log \
    --partition=${gpu_name} --gres=gpu:1 --mem=120G --time=4:00:00 \
    run_sbatch.sh e2e_scripts/train.py \
    --load_hyp_from_wandb_run="${entity}/${project}/${run_id}" \
    --icml_final_eval \
    --skip_initial_eval \
    --silent \
    --wandb_tags="${run_tag},${run_id}" \
    --save_model
  echo "    Logs: jobs/${JOB_NAME}.err"
