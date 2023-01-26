"""
    Functions to evaluate end-to-end clustering and pairwise training
"""
import logging

from tqdm import tqdm
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import torch

from e2e_pipeline.cc_inference import CCInference
from e2e_pipeline.hac_inference import HACInference
from e2e_pipeline.sdp_layer import CvxpyException
from e2e_scripts.train_utils import compute_b3_f1

from IPython import embed


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_auroc_f1(targets, y_pred, thresh_for_f1=0.5, average_mode='macro'):
    fpr, tpr, _ = roc_curve(targets, y_pred)
    roc_auc = auc(fpr, tpr)
    pr, rc, f1, _ = precision_recall_fscore_support(targets, y_pred >= thresh_for_f1, beta=1.0, average=average_mode,
                                                    zero_division=0)
    return roc_auc, np.round(f1, 3)


def evaluate(model, dataloader, overfit_batch_idx=-1, clustering_fn=None, val_dataloader=None,
             tqdm_label='', device=None, verbose=False):
    """
    clustering_fn, val_dataloader: unused when pairwise_mode is False (only added to keep fn signature identical)
    """
    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = dataloader.dataset[0][0].shape[1]

    all_gold, all_pred = [], []
    per_block_metrics = {  # Tracks blocks with at least 2 elements
        'block_idxs': [],
        'block_sizes': [],
        'mlp_auroc': [],
        'b3_f1': [],
        'vmeasure': [],
        'cc_obj_frac': [],
        'cc_obj_round': [],
    }
    max_pred_id = -1
    n_exceptions = 0
    for (idx, batch) in enumerate(tqdm(dataloader, desc=f'Evaluating {tqdm_label}', disable=(not verbose))):
        if overfit_batch_idx > -1:
            if idx < overfit_batch_idx:
                continue
            if idx > overfit_batch_idx:
                break
        data, target, cluster_ids = batch
        block_size = len(cluster_ids)
        data = data.reshape(-1, n_features).float()
        target = target.flatten().float()

        if data.shape[0] == 0:
            # Only one signature in block; manually assign a unique cluster
            pred_cluster_ids = [max_pred_id + 1]
            per_block_metrics['cc_obj_round'].append(1e-8)
            per_block_metrics['cc_obj_frac'].append(1e-8)
            per_block_metrics['mlp_auroc'].append(1.)
        else:
            # Forward pass through the e2e model
            data = data.to(device)
            try:
                _ = model(data, block_size)
            except CvxpyException:
                if tqdm_label is not 'dev':
                    raise CvxpyException()
                # If split is dev, skip batch and continue
                n_exceptions += 1
                logger.info(f'Caught CvxpyException {n_exceptions}: skipping batch')
                continue
            pred_cluster_ids = (model.hac_cut_layer.cluster_labels + (max_pred_id + 1)).tolist()
            y_pred = torch.sigmoid(model.mlp_layer.last_output).cpu().numpy()
            block_auroc, _ = get_auroc_f1(target, y_pred)
            per_block_metrics['mlp_auroc'].append(block_auroc)
            per_block_metrics['cc_obj_round'].append(model.hac_cut_layer.objective_value)
            per_block_metrics['cc_obj_frac'].append(model.sdp_layer.objective_value)

        per_block_metrics['block_idxs'].append(idx)
        per_block_metrics['block_sizes'].append(block_size)
        all_gold += list(np.reshape(cluster_ids, (block_size,)))
        all_pred += list(pred_cluster_ids)
        max_pred_id = max(pred_cluster_ids)
        block_vmeasure = v_measure_score(all_gold[-block_size:], pred_cluster_ids)
        per_block_metrics['vmeasure'].append(block_vmeasure)
        block_b3_f1 = compute_b3_f1(all_gold[-block_size:], pred_cluster_ids)[2]
        per_block_metrics['b3_f1'].append(block_b3_f1)

    vmeasure = v_measure_score(all_gold, all_pred)
    b3_f1 = compute_b3_f1(all_gold, all_pred)[2]
    return b3_f1, vmeasure, per_block_metrics


def evaluate_pairwise(model, dataloader, overfit_batch_idx=-1, mode="macro", return_pred_only=False,
                      thresh_for_f1=0.5, clustering_fn=None, clustering_threshold=None, val_dataloader=None,
                      tqdm_label='', device=None, verbose=False):
    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = dataloader.dataset[0][0].shape[1]

    if clustering_fn is not None:
        per_block_metrics = {
            'block_idxs': [],
            'block_sizes': [],
            'mlp_auroc': [],
            'b3_f1': [],
            'vmeasure': []
        }
        if clustering_fn.__class__ is HACInference:
            if clustering_threshold is None:
                clustering_fn.tune_threshold(model, val_dataloader, device)
        elif clustering_fn.__class__ is CCInference:
            per_block_metrics.update({
                'cc_obj_frac': [],
                'cc_obj_round': []
            })
        all_gold, all_pred = [], []
        max_pred_id = -1  # In each iteration, add to all blockwise predicted IDs to distinguish from previous blocks
        n_exceptions = 0
        for (idx, batch) in enumerate(tqdm(dataloader, desc=f'Evaluating {tqdm_label}', disable=(not verbose))):
            if overfit_batch_idx > -1:
                if idx < overfit_batch_idx:
                    continue
                if idx > overfit_batch_idx:
                    break
            data, target, cluster_ids = batch
            block_size = len(cluster_ids)
            data = data.reshape(-1, n_features).float()
            target = target.flatten().float()

            if data.shape[0] == 0:
                # Only one signature in block; manually assign a unique cluster
                pred_cluster_ids = [max_pred_id + 1]
                per_block_metrics['mlp_auroc'].append(1.)
                if clustering_fn.__class__ is CCInference:
                    per_block_metrics['cc_obj_round'].append(1e-8)
                    per_block_metrics['cc_obj_frac'].append(1e-8)
            else:
                # Forward pass through the e2e model
                data = data.to(device)
                try:
                    pred_cluster_ids = clustering_fn(model(data), block_size, min_id=(max_pred_id + 1),
                                                     threshold=clustering_threshold)
                except CvxpyException:
                    if tqdm_label is not 'dev':
                        raise CvxpyException()
                    # If split is dev, skip batch and continue
                    n_exceptions += 1
                    logger.info(f'Caught CvxpyException {n_exceptions}: skipping batch')
                    continue
                y_pred = torch.sigmoid(model.mlp_layer.last_output).cpu().numpy()
                block_auroc, _ = get_auroc_f1(target, y_pred)
                per_block_metrics['mlp_auroc'].append(block_auroc)
                if clustering_fn.__class__ is CCInference:
                    per_block_metrics['cc_obj_round'].append(clustering_fn.hac_cut_layer.objective_value)
                    per_block_metrics['cc_obj_frac'].append(clustering_fn.sdp_layer.objective_value)

            per_block_metrics['block_idxs'].append(idx)
            per_block_metrics['block_sizes'].append(block_size)
            all_gold += list(np.reshape(cluster_ids, (block_size,)))
            all_pred += list(pred_cluster_ids)
            max_pred_id = max(pred_cluster_ids)
            block_vmeasure = v_measure_score(all_gold[-block_size:], pred_cluster_ids)
            per_block_metrics['vmeasure'].append(block_vmeasure)
            block_b3_f1 = compute_b3_f1(all_gold[-block_size:], pred_cluster_ids)[2]
            per_block_metrics['b3_f1'].append(block_b3_f1)

        vmeasure = v_measure_score(all_gold, all_pred)
        b3_f1 = compute_b3_f1(all_gold, all_pred)[2]
        return b3_f1, vmeasure, per_block_metrics

    y_pred, targets = [], []
    for (idx, batch) in enumerate(tqdm(dataloader, desc=f'Evaluating {tqdm_label}', disable=(not verbose))):
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

    roc_auc, f1 = get_auroc_f1(targets, y_pred, thresh_for_f1, average_mode=mode)

    return roc_auc, f1
