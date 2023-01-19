"""
    Functions to evaluate end-to-end clustering and pairwise training
"""

from tqdm import tqdm
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import torch

from e2e_scripts.train_utils import compute_b3_f1

from IPython import embed


def evaluate(model, dataloader, overfit_batch_idx=-1, clustering_fn=None, tqdm_label='', device=None):
    """
    clustering_fn: unused when pairwise_mode is False (only added to keep fn signature identical)
    """
    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = dataloader.dataset[0][0].shape[1]

    all_gold, all_pred = [], []
    max_pred_id = -1
    for (idx, batch) in enumerate(tqdm(dataloader, desc=f'Evaluating {tqdm_label}')):
        if overfit_batch_idx > -1:
            if idx < overfit_batch_idx:
                continue
            if idx > overfit_batch_idx:
                break
        data, target, cluster_ids = batch
        all_gold += list(cluster_ids)
        data = data.reshape(-1, n_features).float()
        if data.shape[0] == 0:
            # Only one signature in block; manually assign a 0-cluster
            pred_cluster_ids = np.array([0])
        else:
            block_size = len(cluster_ids)
            target = target.flatten().float()
            # Forward pass through the e2e model
            data, target = data.to(device), target.to(device)
            _ = model(data, block_size)
            pred_cluster_ids = model.hac_cut_layer.cluster_labels
        pred_cluster_ids += (max_pred_id + 1)
        max_pred_id = max(pred_cluster_ids)
        all_pred += list(pred_cluster_ids)
    vmeasure = v_measure_score(all_pred, all_gold)
    b3_f1 = compute_b3_f1(all_gold, all_pred)[2]
    return b3_f1, vmeasure


def evaluate_pairwise(model, dataloader, overfit_batch_idx=-1, mode="macro", return_pred_only=False,
                      thresh_for_f1=0.5, clustering_fn=None, clustering_threshold=None, val_dataloader=None,
                      tqdm_label='', device=None):
    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = dataloader.dataset[0][0].shape[1]

    if clustering_fn is not None:
        # Then dataloader passed is blockwise
        all_gold, all_pred = [], []
        max_pred_id = -1  # In each iteration, add to all blockwise predicted IDs to distinguish from previous blocks
        for (idx, batch) in enumerate(tqdm(dataloader, desc=f'Evaluating {tqdm_label}')):
            if overfit_batch_idx > -1:
                if idx < overfit_batch_idx:
                    continue
                if idx > overfit_batch_idx:
                    break
            data, _, cluster_ids = batch
            all_gold += list(cluster_ids)
            data = data.reshape(-1, n_features).float()
            if data.shape[0] == 0:
                # Only one signature in block; manually assign a 0-cluster
                pred_cluster_ids = np.array([0])
            else:
                block_size = len(cluster_ids)
                # Forward pass through the e2e model
                data = data.to(device)
                pred_cluster_ids = clustering_fn(model(data), block_size)
            pred_cluster_ids += (max_pred_id + 1)
            max_pred_id = max(pred_cluster_ids)
            all_pred += list(pred_cluster_ids)
        vmeasure = v_measure_score(all_pred, all_gold)
        b3_f1 = compute_b3_f1(all_gold, all_pred)[2]
        return b3_f1, vmeasure

    y_pred, targets = [], []
    for (idx, batch) in enumerate(tqdm(dataloader, desc=f'Evaluating {tqdm_label}')):
        if overfit_batch_idx > -1:
            if idx < overfit_batch_idx:
                continue
            if idx > overfit_batch_idx:
                break
        data, target = batch
        data = data.reshape(-1, n_features).float()
        assert data.shape[0] != 0
        target = target.flatten().float()
        # Forward pass through the pairwise model
        data = data.to(device)
        y_pred.append(torch.sigmoid(model(data)).cpu().numpy())
        targets.append(target)
    y_pred = np.hstack(y_pred)
    targets = np.hstack(targets)

    if return_pred_only:
        return y_pred

    fpr, tpr, _ = roc_curve(targets, y_pred)
    roc_auc = auc(fpr, tpr)
    pr, rc, f1, _ = precision_recall_fscore_support(targets, y_pred >= thresh_for_f1, beta=1.0, average=mode,
                                                    zero_division=0)

    return roc_auc, np.round(f1, 3)
