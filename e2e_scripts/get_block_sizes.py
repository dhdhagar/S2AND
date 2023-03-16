import argparse
import glob
import json
import logging
import os
import numpy as np
import pickle
from time import time
from tqdm import tqdm

from IPython import embed

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class Parser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument(
            "--src", type=str
        )
        self.add_argument(
            "--unique", action="store_true",
        )
        self.add_argument(
            "--interactive", action="store_true",
        )


if __name__ == '__main__':
    parser = Parser()
    args = parser.parse_args()
    logger.info("Script arguments:")
    logger.info(args.__dict__)

    root = "data/preprocessed_data"
    save_fpath = f'./data_block_sizes{"_" + int(time()) if args.unique else ""}.json'
    ignore = ['pubmed_OLD']
    n_seeds = 5
    splits = ['train', 'val', 'test']

    result = {}

    for dataset_path in tqdm(glob.glob(os.path.join(root, "*"))):
        dataset = dataset_path.split('/')[-1]
        if dataset in ignore:
            continue
        result[dataset] = {}
        for seed in range(1, n_seeds+1):
            result[dataset][seed] = {}
            _seen_blk = set()
            _full_bkl_sizes = []
            for split in splits:
                _blk_szs = []
                fpath = os.path.join(dataset_path, f'seed{seed}', f'{split}_features.pkl')
                with open(fpath, 'rb') as fh:
                    block_dict = pickle.load(fh)
                    for k in block_dict.keys():
                        assert k not in _seen_blk
                        _seen_blk.add(k)
                        _, _, cluster_ids = block_dict[k]
                        _blk_szs.append(len(cluster_ids))
                result[dataset][seed][split] = {
                    'n_blocks': len(_blk_szs),
                    'min': np.min(_blk_szs),
                    'max': np.max(_blk_szs),
                    'mean': np.mean(_blk_szs),
                    'median': np.median(_blk_szs)
                }
                _full_bkl_sizes += _blk_szs
            result[dataset][seed]['full'] = {
                'n_blocks': len(_full_bkl_sizes),
                'min': np.min(_full_bkl_sizes),
                'max': np.max(_full_bkl_sizes),
                'mean': np.mean(_full_bkl_sizes),
                'median': np.median(_full_bkl_sizes)
            }
        result[dataset]['mean_across_seeds'] = {
            'n_blocks': np.mean([result[dataset][seed]['full']['n_blocks'] for seed in range(1, n_seeds + 1)]),
            'min': np.mean([result[dataset][seed]['full']['min'] for seed in range(1, n_seeds + 1)]),
            'max': np.mean([result[dataset][seed]['full']['max'] for seed in range(1, n_seeds + 1)]),
            'mean': np.mean([result[dataset][seed]['full']['mean'] for seed in range(1, n_seeds + 1)]),
            'median': np.mean([result[dataset][seed]['full']['median'] for seed in range(1, n_seeds + 1)])
        }

    with open(save_fpath, 'w') as fh:
        json.dump(result, fh, cls=NpEncoder)

    if args.interactive:
        embed()
