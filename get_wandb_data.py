import wandb
import os
import torch
from collections import defaultdict
import numpy as np
import json
from tqdm import tqdm

from IPython import embed


def get_mean_std(arr):
    _arr = np.array(arr) * 100.
    mean = np.round(np.mean(_arr), 2)
    std = np.round(np.std(_arr), 2)
    plus_minus = u"\u00B1"
    output_string = f"{mean} {plus_minus} {std}"
    return output_string, mean, std

api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("dhdhagar/prob-ent-resolution", filters={"tags": {"$in": ["icml"]}})

TEMP_DIR = './_temp'
os.makedirs(TEMP_DIR, exist_ok=True)

finished = []
failed = 0
other = 0

models = {'e2e', 'e2e-warm', 'frac', 'frac-warm', 'mlp'}

for run in runs:
    # path_id = '/'.join(run.path)
    if 'icml' in run.tags:
        if run.state == 'finished':
            finished.append(run)
        elif run.state in ['failed', 'crashed']:
            failed += 1
        else:
            other += 1

print(f'Total finished: {len(finished)}')
print(f'Total failed: {failed}')
print(f'Other: {other}')
print()


b3_f1 = defaultdict(list)
cc_ratio = defaultdict(list)
auroc = defaultdict(list)

block_sizes = defaultdict(list)
block_auroc = defaultdict(list)
block_b3_f1 = defaultdict(list)
block_cc_ratio = defaultdict(list)

for run in tqdm(finished):
    config = json.loads(run.json_config)
    run_summary = run.summary._json_dict
    model = list(set(run.tags).intersection(models))[0]
    dataset = config['dataset']['value']
    metrics_cc = None
    metrics_hac = None

    for file in run.files():
        fname = file.name
        if fname == 'block_metrics_best_test_cc.pkl':
            file.download(root=TEMP_DIR, replace=True)
            fpath = os.path.join(TEMP_DIR, fname)
            with open(fpath, 'rb') as fh:
                metrics_cc = torch.load(fh)
            os.remove(os.path.join(TEMP_DIR, fname))
        elif fname == 'block_metrics_best_test_hac.pkl':
            file.download(root=TEMP_DIR, replace=True)
            fpath = os.path.join(TEMP_DIR, fname)
            with open(fpath, 'rb') as fh:
                metrics_hac = torch.load(fh)
            os.remove(os.path.join(TEMP_DIR, fname))

    if model == 'mlp':
        key = f'{model}_{dataset}'
        key_cc = f'{model}_cc_{dataset}'
        key_hac = f'{model}_hac_{dataset}'

        b3_f1[key_cc].append(run_summary['best_test_b3_f1_cc'])
        b3_f1[key_hac].append(run_summary['best_test_b3_f1_hac'])
        cc_ratio[key_cc].append(run_summary['best_test_cc_obj_ratio'])
        auroc[key_cc].append(run_summary['best_test_auroc'])
        auroc[key_hac].append(run_summary['best_test_auroc'])

        block_sizes[key_cc].append(metrics_cc['block_sizes'])
        block_sizes[key_hac].append(metrics_hac['block_sizes'])
        block_auroc[key_cc].append(metrics_cc['mlp_auroc'])
        block_auroc[key_hac].append(metrics_hac['mlp_auroc'])
        block_b3_f1[key_cc].append(metrics_cc['b3_f1'])
        block_b3_f1[key_hac].append(metrics_hac['b3_f1'])
        block_cc_ratio[key_cc].append(
            list(np.clip(np.array(metrics_cc['cc_obj_round']) / np.array(metrics_cc['cc_obj_frac']), 0, 1)))
    else:
        key = f'{model}_{dataset}'

        b3_f1[key].append(run_summary['best_test_b3_f1'])
        cc_ratio[key].append(run_summary['best_test_cc_obj_ratio'])
        # TODO: Add total auroc based on blockwise auroc
        all_pairs_sizes = np.array(metrics_cc['block_sizes']) * (np.array(metrics_cc['block_sizes']) - 1) / 2
        run_auroc = np.sum(np.array(metrics_cc['mlp_auroc']) * all_pairs_sizes) / np.sum(all_pairs_sizes)
        auroc[key].append(run_auroc)

        block_sizes[key].append(metrics_cc['block_sizes'])
        block_auroc[key].append(metrics_cc['mlp_auroc'])
        block_b3_f1[key].append(metrics_cc['b3_f1'])
        block_cc_ratio[key].append(
            list(np.clip(np.array(metrics_cc['cc_obj_round']) / np.array(metrics_cc['cc_obj_frac']), 0, 1)))

mean_std_strings = dict(map(lambda x: (x[0], get_mean_std(x[1])), b3_f1.items()))

embed()
