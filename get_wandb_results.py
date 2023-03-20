import argparse
import json
import logging
import csv
from copy import deepcopy
import numpy as np
import pandas as pd

from IPython import embed

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Parser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument(
            "--data_fpath", type=str
        )
        self.add_argument(
            "--interactive", action="store_true",
        )
        self.add_argument(
            "--get_b3_f1_across", action="store_true",
        )


def get_df_by_dataset(res, dataset):
    new_res = {}
    for _r in res:
        if dataset in _r:
            new_res[_r.replace(f"{dataset}_", '')] = res[_r]
    return pd.DataFrame(new_res).T

if __name__ == '__main__':
    parser = Parser()
    args = parser.parse_args()
    logger.info("Script arguments:")
    logger.info(args.__dict__)

    if args.data_fpath is not None:
        fpath = args.data_fpath
    else:
        # hardcoded during dev
        fpath = 'wandb_export_2023-03-19T14_30_08.659-04_00.csv'

    results = []
    with open(fpath, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
            else:
                results.append(deepcopy(row))
            line_count += 1
        print(f'Processed {line_count} lines.')

    final = {}
    out_keys = {
        'train_time': 'z_run_time',
        'inf_time': 'z_inf_time',
        'b3_f1_hac': 'best_test_b3_f1_hac',
        'b3_f1_cc': 'best_test_b3_f1_cc',
        'b3_f1_cc-fixed': 'best_test_b3_f1_cc-fixed',
        'b3_f1_cc-nosdp': 'best_test_b3_f1_cc-nosdp',
        'b3_f1_cc-nosdp-fixed': 'best_test_b3_f1_cc-nosdp-fixed',
        'vmeasure_hac': 'best_test_vmeasure_hac',
        'vmeasure_cc': 'best_test_vmeasure_cc',
        'vmeasure_cc-fixed': 'best_test_vmeasure_cc-fixed',
        'vmeasure_cc-nosdp': 'best_test_vmeasure_cc-nosdp',
        'vmeasure_cc-nosdp-fixed': 'best_test_vmeasure_cc-nosdp-fixed'
    }

    for r in results:
        try:
            method = f"{'mlp' if r['pairwise_mode']=='true' else 'e2e'}"
            if r['pairwise_mode'] == 'false':
                method += f"{'_nosdp' if r['use_sdp']=='false' else ''}"
                method += f"{'_round' if r['use_rounded_loss'] == 'true' else '_frac'}"
            key = f"{r['dataset']}_{method}"

            if key not in final:
                final[key] = {o: [] for o in out_keys.keys()}

            for _key in out_keys:
                final[key][_key].append(float(r[out_keys[_key]]))
        except:
            continue

    means, stds, comb = {}, {}, {}
    for k in final:
        if k is not means:
            means[k] = {}
            stds[k] = {}
            comb[k] = {}
        for _k in final[k]:
            means[k][_k] = round(np.mean(final[k][_k])*(1 if 'time' in _k else 100), 2)
            stds[k][_k] = round(np.std(final[k][_k])*(1 if 'time' in _k else 100), 2)
            comb[k][_k] = f"{means[k][_k]}±{stds[k][_k]}"

    with open('results-mean.json', 'w') as fh:
        json.dump(means, fh)
    with open('results-std.json', 'w') as fh:
        json.dump(stds, fh)
    with open('results.json', 'w') as fh:
        json.dump(comb, fh)

    res_df = pd.DataFrame(comb)

    if args.get_b3_f1_across:
        # Average b3_f1 numbers of each training method over all inference methods
        print()
        print()
        mean_dfs = {}
        for d in ['pubmed', 'qian', 'zbmath', 'arnetminer', 'kisti']:
            print(f'Dataset: {d}')
            mean_dfs[d] = get_df_by_dataset(means, d).T[
                      ['b3_f1_hac', 'b3_f1_cc', 'b3_f1_cc-fixed', 'b3_f1_cc-nosdp', 'b3_f1_cc-nosdp-fixed']].T.mean()
            print(mean_dfs[d])
            print()

    if args.interactive:
        embed()
