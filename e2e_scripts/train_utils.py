"""
    Helper functions and constants for e2e_scripts/train.py
"""
import os
import json
from collections import defaultdict
from typing import Dict
from typing import Tuple
import math
import pickle
from torch.utils.data import DataLoader
from s2and.consts import PREPROCESSED_DATA_DIR
from s2and.data import S2BlocksDataset
from s2and.eval import b3_precision_recall_fscore
import torch
import numpy as np
import wandb

from IPython import embed


# Default hyperparameters
DEFAULT_HYPERPARAMS = {
    # Dataset
    "dataset": "pubmed",
    "dataset_random_seed": 1,
    "subsample_sz_train": -1,
    "subsample_sz_dev": -1,
    # Run config
    "run_random_seed": 17,
    "pairwise_mode": False,
    # Data config
    "convert_nan": False,
    "nan_value": -1,
    "drop_feat_nan_pct": -1,
    "normalize_data": True,
    # Model config
    "neumiss_deq": False,
    "neumiss_depth": 20,
    "hidden_dim": 512,
    "n_hidden_layers": 2,
    "dropout_p": 0.1,
    "dropout_only_once": True,
    "batchnorm": True,
    "hidden_config": None,
    "activation": "leaky_relu",
    "negative_slope": 0.01,
    "use_rounded_loss": True,
    "use_sdp": True,
    # Solver config
    "sdp_max_iters": 50000,
    "sdp_eps": 1e-3,
    "sdp_scale": True,
    # Training config
    "batch_size": 10000,  # For pairwise_mode only
    "lr": 4e-3,
    "n_epochs": 5,
    "n_warmstart_epochs": 0,
    "weighted_loss": True,
    # "normalize_loss": True,  # For e2e mode only
    "use_lr_scheduler": True,
    "lr_scheduler": "plateau",  # "step"
    "lr_factor": 0.4,
    "lr_min": 1e-6,
    "lr_scheduler_patience": 2,
    "lr_step_size": 2,
    "lr_gamma": 0.4,
    "weight_decay": 0.01,
    "gradient_accumulation": True,  # if True, accumulate gradient from batch_size number of pairwise examples
    "dev_opt_metric": 'b3_f1',  # e2e: {'b3_f1', 'vmeasure'}; pairwise: {'auroc', 'f1'}
    "overfit_batch_idx": -1
}


def read_blockwise_features(pkl):
    blockwise_data: Dict[str, Tuple[np.ndarray, np.ndarray]]
    with open(pkl, "rb") as _pkl_file:
        blockwise_data = pickle.load(_pkl_file)
    return blockwise_data


def get_dataloaders(dataset, dataset_seed, convert_nan, nan_value, normalize, subsample_sz_train, subsample_sz_dev,
                    pairwise_mode, batch_size):
    train_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset}/seed{dataset_seed}/train_features.pkl"
    val_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset}/seed{dataset_seed}/val_features.pkl"
    test_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset}/seed{dataset_seed}/test_features.pkl"

    train_dataset = S2BlocksDataset(read_blockwise_features(train_pkl), convert_nan=convert_nan, nan_value=nan_value,
                                    scale=normalize, subsample_sz=subsample_sz_train, pairwise_mode=pairwise_mode)
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)

    val_dataset = S2BlocksDataset(read_blockwise_features(val_pkl), convert_nan=convert_nan, nan_value=nan_value,
                                  scale=normalize, scaler=train_dataset.scaler, subsample_sz=subsample_sz_dev,
                                  pairwise_mode=pairwise_mode)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)

    test_dataset = S2BlocksDataset(read_blockwise_features(test_pkl), convert_nan=convert_nan, nan_value=nan_value,
                                   scale=normalize, scaler=train_dataset.scaler, pairwise_mode=pairwise_mode)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader


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


def save_to_wandb_run(file, fname, fpath, logger):
    with open(os.path.join(fpath, fname), 'w') as fh:
        json.dump(file, fh)
    wandb.save(fname)
    logger.info(f"Saved {fname} to {os.path.join(fpath, fname)}")
