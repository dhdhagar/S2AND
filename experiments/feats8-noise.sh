#!/bin/bash -e

partition=${1:-"cpu"}

sweep_prefix="feats8-noise"
feats8_flags="--keep_feat_idxs=0 --keep_feat_idxs=1 --keep_feat_idxs=2 \
    --keep_feat_idxs=3 --keep_feat_idxs=4 --keep_feat_idxs=5 \
    --keep_feat_idxs=14 --keep_feat_idxs=15"
log_fpath="./experiments/logs/${sweep_prefix}_$(date +%s).txt"
declare -a arr_models=("e2e" "e2e-nosdp" "frac" "frac-nosdp" "mlp")
declare -a arr_datasets=("pubmed" "qian" "arnetminer")
declare -a arr_noise=("1" "2" "3")

for noise in "${arr_noise[@]}"
do
  for dataset in "${arr_datasets[@]}"
  do
    for model in "${arr_models[@]}"
    do
      ./run_sweep.sh ${dataset} 1 5 ${model} ${partition} "${feats8_flags} --noise_std=${noise}" "${sweep_prefix}${noise}" >> ${log_fpath}
    done
  done
done
