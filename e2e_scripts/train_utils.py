"""
    Helper functions and constants for e2e_scripts/train.py
"""
import copy
import os
import json
import random
import logging
from collections import defaultdict
from typing import Dict
from typing import Tuple, Optional
import math
import pickle
import torch
import numpy as np
import wandb
from time import time
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from s2and.consts import PREPROCESSED_DATA_DIR
from s2and.data import S2BlocksDataset
from s2and.eval import b3_precision_recall_fscore
from torch import Tensor
from torch.multiprocessing import Process

from IPython import embed


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


# Default hyperparameters
DEFAULT_HYPERPARAMS = {
    # Dataset
    "dataset": "pubmed",
    "dataset_random_seed": 1,
    "subsample_sz_train": 60,
    "subsample_sz_dev": 300,
    # Run config
    "run_random_seed": 17,
    "pairwise_mode": False,
    "early_terminate_epochs": 5,
    # Data config
    "convert_nan": False,
    "nan_value": -1,
    "normalize_data": True,
    "drop_feat_idxs": [],
    "keep_feat_idxs": [],
    "noise_std": 0.,
    # Model config
    "neumiss_deq": False,
    "neumiss_depth": 20,
    "hidden_dim": 512,
    "n_hidden_layers": 2,
    "dropout_p": 0.1,
    "dropout_only_once": True,
    "batchnorm": False,
    "layernorm": True,
    "hidden_config": None,
    "activation": "leaky_relu",
    "negative_slope": 0.01,
    "use_rounded_loss": True,
    "use_sdp": True,
    "e2e_loss": "frob",  # e2e only: "frob", "bce"
    # Solver config
    "sdp_max_iters": 50000,
    "sdp_eps": 1e-3,
    "sdp_scale": True,
    # Training config
    "batch_size": 8000,  # pairwise only; used by e2e if gradient_accumulation is true
    "lr": 1e-3,
    "n_epochs": 5,
    "n_warmstart_epochs": 0,
    "weighted_loss": True,
    "use_lr_scheduler": True,
    "lr_scheduler": "plateau",  # "plateau", "step"
    "lr_factor": 0.4,
    "lr_min": 1e-6,
    "lr_scheduler_patience": 2,
    "lr_step_size": 2,
    "lr_gamma": 0.4,
    "max_grad_norm": 1,  # Off if set to -1
    "weight_decay": 0.01,
    "gradient_accumulation": True,  # e2e only; accumulate over <batch_size> pairwise examples
    "dev_opt_metric": 'b3_f1',  # e2e: {'b3_f1', 'vmeasure'}; pairwise: {'auroc', 'f1'}
    "overfit_batch_idx": -1
}


def read_blockwise_features(pkl):
    blockwise_data: Dict[str, Tuple[np.ndarray, np.ndarray]]
    with open(pkl, "rb") as _pkl_file:
        blockwise_data = pickle.load(_pkl_file)
    return blockwise_data


def _get_feat_idxs(_n_features, _keep_feat_idxs, _drop_feat_idxs):
    _keep = set(range(_n_features) if len(_keep_feat_idxs) == 0 else map(lambda x: int(x), _keep_feat_idxs))
    _drop = set(map(lambda x: int(x), _drop_feat_idxs))
    return np.array(list(_keep - _drop))


def get_dataloaders(dataset, dataset_seed, convert_nan, nan_value, normalize, subsample_sz_train, subsample_sz_dev,
                    pairwise_mode, batch_size, shuffle=False, split=None, drop_feat_idxs=[], keep_feat_idxs=[],
                    noise_std=0.):
    pickle_path = {
        'train': f"{PREPROCESSED_DATA_DIR}/{dataset}/seed{dataset_seed}/train_features.pkl",
        'dev': f"{PREPROCESSED_DATA_DIR}/{dataset}/seed{dataset_seed}/val_features.pkl",
        'test': f"{PREPROCESSED_DATA_DIR}/{dataset}/seed{dataset_seed}/test_features.pkl"
    }
    subsample_sz = {
        'train': subsample_sz_train,
        'dev': subsample_sz_dev,
        'test': -1
    }
    train_scaler = StandardScaler()
    train_X = np.concatenate(list(map(lambda x: x[0], read_blockwise_features(pickle_path['train']).values())))
    train_scaler.fit(train_X)
    # Additionally drop features that are all nan's at training
    all_nan_idxs = list(np.where(train_scaler.n_samples_seen_ == 0)[0]) \
        if type(train_scaler.n_samples_seen_) == np.ndarray else []
    if len(all_nan_idxs) > 0:
        logger.info(f"Dropped {len(all_nan_idxs)} all NaN features: {all_nan_idxs}")
    feat_idxs = _get_feat_idxs(train_X.shape[1], keep_feat_idxs, drop_feat_idxs + all_nan_idxs)
    if len(feat_idxs) != train_X.shape[1]:
        train_scaler.fit(train_X[:, feat_idxs])


    def _get_dataloader(_split):
        dataset = S2BlocksDataset(read_blockwise_features(pickle_path[_split]), convert_nan=convert_nan,
                                  nan_value=nan_value, scale=normalize, scaler=train_scaler,
                                  subsample_sz=subsample_sz[_split],
                                  pairwise_mode=pairwise_mode, sort_desc=(_split in ['dev', 'test']),
                                  feat_idxs=feat_idxs, noise_std=noise_std)
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)
        return dataloader

    if split is None:
        return _get_dataloader('train'), _get_dataloader('dev'), _get_dataloader('test')
    if type(split) is str:
        return _get_dataloader(split)
    if type(split) is list:
        return tuple([_get_dataloader(_split) for _split in split])
    raise ValueError('Invalid argument to split')


def get_feature_count(dataset, dataset_seed):
    data_fpath = f"{PREPROCESSED_DATA_DIR}/{dataset}/seed{dataset_seed}/test_features.pkl"
    block_dict = read_blockwise_features(data_fpath)
    return next(iter(block_dict.values()))[0].shape[1]


def uncompress_target_tensor(compressed_targets, make_symmetric=True, device=None):
    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = round(math.sqrt(2 * compressed_targets.size(dim=0))) + 1
    # Convert the 1D pairwise-similarities list to nxn upper triangular matrix
    ind0, ind1 = torch.triu_indices(n, n, offset=1)
    target = torch.eye(n, device=device)
    target[ind0, ind1] = compressed_targets
    if make_symmetric:
        target[ind1, ind0] = compressed_targets
    return target


# Count parameters in the model
def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_matrix_size_from_triu(triu):
    return round(math.sqrt(2 * len(triu))) + 1


def compute_b3_f1(true_cluster_ids, pred_cluster_ids):
    """
    Compute the B^3 variant of precision, recall and F-score.
    Returns:
        Precision
        Recall
        F1
        Per signature metrics
        Overmerging ratios
        Undermerging ratios
    """
    true_cluster_dict, pred_cluster_dict = defaultdict(list), defaultdict(list)
    for i in range(len(true_cluster_ids)):
        true_cluster_dict[true_cluster_ids[i]].append(i)
        pred_cluster_dict[pred_cluster_ids[i]].append(i)
    return b3_precision_recall_fscore(true_cluster_dict, pred_cluster_dict)


def log_cc_objective_values(scores, split_name, log_prefix, verbose, logger, plot=False):
    frac, round = np.array(scores[2]['sdp']), np.array(scores[2]['round'])
    # Objective across blocks
    total_frac_obj = np.sum(frac)
    total_round_obj = np.sum(round)
    # Mean approximation ratio across blocks
    mean_approx_ratio = min(1., np.mean(round / frac))

    if verbose:
        logger.info(f"{log_prefix}: {split_name}_obj_frac={total_frac_obj}, " +
                    f"{split_name}_obj_round={total_round_obj}, " +
                    f"{split_name}_obj_ratio={mean_approx_ratio}")

    wandb.log({f'{split_name}_obj_frac': total_frac_obj,
               f'{split_name}_obj_round': total_round_obj,
               f'{split_name}_obj_ratio': mean_approx_ratio})

    # TODO: Implement plotting the approx. ratio v/s block sizes


def save_to_wandb_run(file, fname, fpath, logger, error_logger=True):
    if error_logger and os.path.exists(os.path.join(fpath, fname)):
        with open(os.path.join(fpath, fname), 'r') as fh:
            all_errors = json.load(fh)['errors']
            all_ids = set([e['id'] for e in all_errors])
            for new_error in file['errors']:
                if new_error['id'] not in all_ids:
                    all_errors.append(new_error)
            file['errors'] = all_errors
    with open(os.path.join(fpath, fname), 'w') as fh:
        json.dump(file, fh)
    wandb.save(fname)
    logger.info(f"Saved {fname} to {os.path.join(fpath, fname)}")
    return file


class FrobeniusLoss:
    def __init__(self, weight: Optional[Tensor] = None, reduction: str = 'original') -> None:
        self.weight = weight
        self.reduction = reduction

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        n = len(target)
        normalization = 1.
        if self.reduction == 'mean':
            normalization = n * (n - 1)
        elif self.reduction == 'original':  # TODO: Probably want to not use this
            normalization = 2 * n
        if self.weight is None:
            return torch.norm((target - input)) / normalization
        return torch.norm(self.weight * (target - input)) / normalization


def copy_and_load_model(model, run_dir, device, store_only=False):
    _model = copy.deepcopy(model)
    _PATH = os.path.join(run_dir, f'_temp_state_dict_{int(time())}-{random.randint(0, 100)}.pt')
    torch.save(model.state_dict(), _PATH)
    if store_only:
        return _PATH
    _STATE_DICT = torch.load(_PATH, device)
    _model.load_state_dict(_STATE_DICT)
    os.remove(_PATH)
    return _model


def _check_process(_proc, _return_dict, logger, run, overfit_batch_idx, use_lr_scheduler, hyp,
                   scheduler, eval_metric_to_idx, dev_opt_metric, i, best_epoch, best_dev_score,
                   best_dev_scores, best_dev_state_dict, early_terminate_epochs, early_terminate_ctr, sync=False):
    early_terminate = False
    if _proc is not None:
        if _return_dict['_state'] == 'done' or (sync and _return_dict['_state'] != 'finish'):
            _proc.join()
            _return_dict['_state'] = 'finish'
            if _return_dict['_method'] == 'init_eval':
                logger.info(_return_dict['local'])
                run.log(_return_dict['wandb'])
                if overfit_batch_idx == -1:
                    best_dev_scores = _return_dict['dev_scores']
                    best_dev_score = best_dev_scores[eval_metric_to_idx[dev_opt_metric]]
            elif _return_dict['_method'] == 'dev_eval':
                logger.info(_return_dict['local'])
                run.log(_return_dict['wandb'])
                if overfit_batch_idx > -1:
                    if use_lr_scheduler:
                        if hyp['lr_scheduler'] == 'plateau':
                            scheduler.step(_return_dict['train_scores'][eval_metric_to_idx[dev_opt_metric]])
                        elif hyp['lr_scheduler'] == 'step':
                            scheduler.step()
                else:
                    if early_terminate_ctr is not None:
                        early_terminate_ctr -= 1
                    dev_scores = _return_dict['dev_scores']
                    dev_opt_score = dev_scores[eval_metric_to_idx[dev_opt_metric]]
                    if dev_opt_score > best_dev_score:
                        early_terminate_ctr = early_terminate_epochs
                        logger.info(f"New best dev {dev_opt_metric} score @ epoch{i + 1}: {dev_opt_score}")
                        best_epoch = i
                        best_dev_score = dev_opt_score
                        best_dev_scores = dev_scores
                        best_dev_state_dict = torch.load(_return_dict['state_dict_path'])
                    elif early_terminate_ctr < 0:
                        early_terminate = True
                    if use_lr_scheduler:
                        if hyp['lr_scheduler'] == 'plateau':
                            scheduler.step(dev_scores[eval_metric_to_idx[dev_opt_metric]])
                        elif hyp['lr_scheduler'] == 'step':
                            scheduler.step()
    return best_epoch, best_dev_score, best_dev_scores, best_dev_state_dict, early_terminate_ctr, early_terminate


def init_eval(model_class, model_args, state_dict_path, overfit_batch_idx, eval_fn, train_dataloader, device, verbose,
              debug, _errors, eval_metric_to_idx, val_dataloader, return_dict, run_dir):
    return_dict['_state'] = 'start'
    return_dict['_method'] = 'init_eval'
    model = model_class(*model_args)
    model.load_state_dict(torch.load(state_dict_path))
    model.to(device)
    with torch.no_grad():
        model.eval()
        if overfit_batch_idx > -1:
            train_scores = eval_fn(model, train_dataloader, overfit_batch_idx=overfit_batch_idx,
                                   tqdm_label='train', device=device, verbose=verbose, debug=debug,
                                   _errors=_errors, tqdm_position=0, model_args=model_args, run_dir=run_dir)
            return_dict['local'] = f"Initial: train_{list(eval_metric_to_idx)[0]}={train_scores[0]}, " + \
                                   f"train_{list(eval_metric_to_idx)[1]}={train_scores[1]}"
            return_dict['wandb'] = {'epoch': 0, f'train_{list(eval_metric_to_idx)[0]}': train_scores[0],
                                    f'train_{list(eval_metric_to_idx)[1]}': train_scores[1]}
        else:
            dev_scores = eval_fn(model, val_dataloader, tqdm_label='dev 0', device=device, verbose=verbose,
                                 debug=debug, _errors=_errors, tqdm_position=0, model_args=model_args,
                                 run_dir=run_dir)
            return_dict['local'] = f"Initial: dev_{list(eval_metric_to_idx)[0]}={dev_scores[0]}, " + \
                                   f"dev_{list(eval_metric_to_idx)[1]}={dev_scores[1]}"
            return_dict['wandb'] = {'epoch': 0, f'dev_{list(eval_metric_to_idx)[0]}': dev_scores[0],
                                    f'dev_{list(eval_metric_to_idx)[1]}': dev_scores[1]}
            return_dict['dev_scores'] = dev_scores
    del model
    return_dict['_state'] = 'done'
    return return_dict


def dev_eval(model_class, model_args, state_dict_path, overfit_batch_idx, eval_fn, train_dataloader, device, verbose,
             debug, _errors, eval_metric_to_idx, val_dataloader, return_dict, epoch_idx, run_dir):
    return_dict['_state'] = 'start'
    return_dict['_method'] = 'dev_eval'
    return_dict['state_dict_path'] = state_dict_path
    model = model_class(*model_args)
    model.load_state_dict(torch.load(state_dict_path))
    model.to(device)
    with torch.no_grad():
        model.eval()
        if overfit_batch_idx > -1:
            train_scores = eval_fn(model, train_dataloader, overfit_batch_idx=overfit_batch_idx,
                                   tqdm_label='train', device=device, verbose=verbose, debug=debug,
                                   _errors=_errors, model_args=model_args, run_dir=run_dir)
            return_dict['local'] = f"Epoch {epoch_idx + 1}: train_{list(eval_metric_to_idx)[0]}={train_scores[0]}, " + \
                                   f"train_{list(eval_metric_to_idx)[1]}={train_scores[1]}"
            return_dict['wandb'] = {f'train_{list(eval_metric_to_idx)[0]}': train_scores[0],
                                    f'train_{list(eval_metric_to_idx)[1]}': train_scores[1]}
            return_dict['train_scores'] = train_scores
        else:
            dev_scores = eval_fn(model, val_dataloader, tqdm_label=f'dev {epoch_idx + 1}', device=device, verbose=verbose,
                                 debug=debug, _errors=_errors, model_args=model_args, run_dir=run_dir)
            return_dict['local'] = f"Epoch {epoch_idx + 1}: dev_{list(eval_metric_to_idx)[0]}={dev_scores[0]}, " + \
                                   f"dev_{list(eval_metric_to_idx)[1]}={dev_scores[1]}"
            return_dict['wandb'] = {f'dev_{list(eval_metric_to_idx)[0]}': dev_scores[0],
                                    f'dev_{list(eval_metric_to_idx)[1]}': dev_scores[1]}
            return_dict['dev_scores'] = dev_scores
    del model
    return_dict['_state'] = 'done'
    return return_dict


def fork_eval(target, args, model, run_dir, device, logger, sync=False):
    state_dict_path = copy_and_load_model(model, run_dir, device, store_only=True)
    args['model_class'] = model.__class__
    args['state_dict_path'] = state_dict_path
    if sync:
        target(**args)
        proc = Process()
    else:
        proc = Process(target=target, kwargs=args)
        logger.info('Forking eval')
    proc.start()
    return proc
