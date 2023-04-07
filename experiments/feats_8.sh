#!/bin/bash -e

sweep_prefix="feats8"
log_fpath="./experiments/logs/${sweep_prefix}_$(date +%s).txt"

dataset="pubmed"
declare -a models=("e2e" "e2e-nosdp" "frac" "frac-nosdp" "mlp")
gpu_name="gypsum-m40"
for model in "${models[@]}"
do
   ./experiments/_feats_8.sh ${dataset} 1 5 ${model} ${gpu_name} ${sweep_prefix} >> ${log_fpath}
done

dataset="qian"
declare -a models=("e2e" "e2e-nosdp" "frac" "frac-nosdp" "mlp")
gpu_name="gypsum-2080ti"
for model in "${models[@]}"
do
   ./experiments/_feats_8.sh ${dataset} 1 5 ${model} ${gpu_name} ${sweep_prefix} >> ${log_fpath}
done

dataset="arnetminer"
declare -a models=("e2e" "e2e-nosdp" "frac" "frac-nosdp" "mlp")
gpu_name="gypsum-1080ti"
for model in "${models[@]}"
do
   ./experiments/_feats_8.sh ${dataset} 1 5 ${model} ${gpu_name} ${sweep_prefix} >> ${log_fpath}
done
