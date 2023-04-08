#!/bin/bash -e

sweep_prefix="feats8"
flags="--keep_feat_idxs=0 --keep_feat_idxs=1 --keep_feat_idxs=2 \
    --keep_feat_idxs=3 --keep_feat_idxs=4 --keep_feat_idxs=5 \
    --keep_feat_idxs=14 --keep_feat_idxs=15"
log_fpath="./experiments/logs/${sweep_prefix}_$(date +%s).txt"
declare -a models=("e2e" "e2e-nosdp" "frac" "frac-nosdp" "mlp")

dataset="pubmed"
gpu_name="gypsum-m40"
for model in "${models[@]}"
do
   ./run_sweep.sh ${dataset} 1 5 ${model} ${gpu_name} ${flags} ${sweep_prefix} >> ${log_fpath}
done

dataset="qian"
gpu_name="gypsum-2080ti"
for model in "${models[@]}"
do
   ./run_sweep.sh ${dataset} 1 5 ${model} ${gpu_name} ${flags} ${sweep_prefix} >> ${log_fpath}
done

dataset="arnetminer"
gpu_name="gypsum-1080ti"
for model in "${models[@]}"
do
   ./run_sweep.sh ${dataset} 1 5 ${model} ${gpu_name} ${flags} ${sweep_prefix} >> ${log_fpath}
done
