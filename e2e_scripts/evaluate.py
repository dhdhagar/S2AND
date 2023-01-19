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

    vmeasure, b3_f1, sigs_per_block = [], [], []
    for (idx, batch) in enumerate(tqdm(dataloader, desc=f'Evaluating {tqdm_label}')):
        if overfit_batch_idx > -1:
            if idx < overfit_batch_idx:
                continue
            if idx > overfit_batch_idx:
                break
        data, target, cluster_ids = batch
        data = data.reshape(-1, n_features).float()
        if data.shape[0] == 0:
            # Only one signature in block -> predict correctly
            vmeasure.append(1.)
            b3_f1.append(1.)
            sigs_per_block.append(1)
        else:
            block_size = len(cluster_ids)  # get_matrix_size_from_triu(data)
            cluster_ids = np.reshape(cluster_ids, (block_size,))
            target = target.flatten().float()
            sigs_per_block.append(block_size)

            # Forward pass through the e2e model
            data, target = data.to(device), target.to(device)
            _ = model(data, block_size)
            predicted_cluster_ids = model.hac_cut_layer.cluster_labels  # .detach()

            # Compute clustering metrics
            vmeasure.append(v_measure_score(predicted_cluster_ids, cluster_ids))
            b3_f1_metrics = compute_b3_f1(cluster_ids, predicted_cluster_ids)
            b3_f1.append(b3_f1_metrics[2])

    vmeasure = np.array(vmeasure)
    b3_f1 = np.array(b3_f1)
    sigs_per_block = np.array(sigs_per_block)

    return np.sum(b3_f1 * sigs_per_block) / np.sum(sigs_per_block), \
           np.sum(vmeasure * sigs_per_block) / np.sum(sigs_per_block)


def evaluate_pairwise(model, dataloader, overfit_batch_idx=-1, mode="macro", return_pred_only=False,
                      thresh_for_f1=0.5, clustering_fn=None, clustering_threshold=None, val_dataloader=None,
                      tqdm_label='', device=None):
    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = dataloader.dataset[0][0].shape[1]

    if clustering_fn is not None:
        # Then dataloader passed is blockwise
        vmeasure, b3_f1, sigs_per_block = [], [], []
        for (idx, batch) in enumerate(tqdm(dataloader, desc=f'Evaluating {tqdm_label}')):
            if overfit_batch_idx > -1:
                if idx < overfit_batch_idx:
                    continue
                if idx > overfit_batch_idx:
                    break
            data, target, cluster_ids = batch
            data = data.reshape(-1, n_features).float()
            if data.shape[0] == 0:
                # Only one signature in block -> predict correctly
                vmeasure.append(1.)
                b3_f1.append(1.)
                sigs_per_block.append(1)
            else:
                block_size = len(cluster_ids)  # get_matrix_size_from_triu(data)
                cluster_ids = np.reshape(cluster_ids, (block_size,))
                target = target.flatten().float()
                sigs_per_block.append(block_size)

                # Forward pass through the e2e model
                data, target = data.to(device), target.to(device)
                predicted_cluster_ids = clustering_fn(model(data), block_size)  # .detach()

                # Compute clustering metrics
                vmeasure.append(v_measure_score(predicted_cluster_ids, cluster_ids))
                b3_f1_metrics = compute_b3_f1(cluster_ids, predicted_cluster_ids)
                b3_f1.append(b3_f1_metrics[2])

        vmeasure = np.array(vmeasure)
        b3_f1 = np.array(b3_f1)
        sigs_per_block = np.array(sigs_per_block)

        return np.sum(vmeasure * sigs_per_block) / np.sum(sigs_per_block), \
               np.sum(b3_f1 * sigs_per_block) / np.sum(sigs_per_block)

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
