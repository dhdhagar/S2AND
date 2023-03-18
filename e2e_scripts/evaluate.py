"""
    Functions to evaluate end-to-end clustering and pairwise training
"""
import logging

from tqdm import tqdm
from time import time
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support
from torch.multiprocessing import Process, Manager
import numpy as np
import torch

from e2e_pipeline.cc_inference import CCInference
from e2e_pipeline.hac_inference import HACInference
from e2e_pipeline.sdp_layer import CvxpyException
from e2e_scripts.train_utils import compute_b3_f1, save_to_wandb_run, copy_and_load_model

from IPython import embed


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def _run_iter(model_class, state_dict_path, _fork_id, _shared_list, **kwargs):
    model = model_class(*kwargs['model_args'])
    model.load_state_dict(torch.load(state_dict_path))
    model.to('cpu')
    model.eval()
    with torch.no_grad():
        res = evaluate(model=model, **kwargs)
    _shared_list.append(res)
    del model


def fork_iter(batch_idx, _fork_id, _shared_list, **kwargs):
    kwargs['model_class'] = kwargs['model'].__class__
    kwargs['state_dict_path'] = copy_and_load_model(kwargs['model'], kwargs['run_dir'], 'cpu', store_only=True)
    del kwargs['model']
    kwargs['overfit_batch_idx'] = batch_idx
    kwargs['tqdm_label'] = f'{kwargs["tqdm_label"]} (fork{_fork_id})'
    kwargs['_fork_id'] = _fork_id
    kwargs['tqdm_position'] = (0 if kwargs['tqdm_position'] is None else kwargs['tqdm_position']) + _fork_id + 1
    kwargs['return_iter'] = True
    kwargs['fork_size'] = -1
    kwargs['_shared_list'] = _shared_list
    kwargs['disable_tqdm'] = True
    kwargs['device'] = 'cpu'
    _proc = Process(target=_run_iter, kwargs=kwargs)
    _proc.start()
    return _proc


def evaluate(model, dataloader, overfit_batch_idx=-1, clustering_fn=None, clustering_threshold=None,
             val_dataloader=None, tqdm_label='', device=None, verbose=False, debug=False, _errors=None,
             run_dir='./', tqdm_position=None, model_args=None, return_iter=False, fork_size=300,
             disable_tqdm=False):
    """
    clustering_fn, clustering_threshold, val_dataloader: unused when pairwise_mode is False
    (only added to keep fn signature identical)
    """
    fn_args = locals()
    fork_enabled = fork_size > -1 and model_args is not None
    if fork_enabled:
        _fork_id = 1
        _shared_list = Manager().list()
        _procs = []
    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = dataloader.dataset[0][0].shape[1]

    all_gold, all_pred = [], []
    cc_obj_vals = {
        'sdp': [],
        'round': [],
        'block_idxs': [],
        'block_sizes': []
    }
    max_pred_id = -1
    n_exceptions = 0
    pbar = tqdm(dataloader, desc=f'Eval {tqdm_label}', position=tqdm_position, disable=disable_tqdm)
    for (idx, batch) in enumerate(pbar):
        if overfit_batch_idx > -1:
            if idx < overfit_batch_idx:
                continue
            if idx > overfit_batch_idx:
                break
        data, _, cluster_ids = batch
        block_size = len(cluster_ids)
        pbar.set_description(f'Eval {tqdm_label} (sz={block_size})')
        data = data.reshape(-1, n_features).float()
        if data.shape[0] == 0:
            # Only one signature in block; manually assign a unique cluster
            pred_cluster_ids = [max_pred_id + 1]
        elif fork_enabled and block_size >= fork_size:
            _proc = fork_iter(idx, _fork_id, _shared_list, **fn_args)
            _fork_id += 1
            _procs.append(_proc)
            continue
        else:
            # Forward pass through the e2e model
            data = data.to(device)
            try:
                _ = model(data, block_size, verbose=verbose)
            except CvxpyException as e:
                logger.info(e)
                _error_obj = {
                    'id': f'e_{int(time())}',
                    'method': 'eval',
                    'model_type': 'e2e',
                    'data_split': tqdm_label,
                    'model_call_args': {
                        'data': data.detach().tolist(),
                        'block_size': block_size
                    },
                    'cvxpy_layer_args': e.data
                }
                if _errors is not None:
                    _errors.append(_error_obj)
                    save_to_wandb_run({'errors': _errors}, 'errors.json', run_dir, logger)
                if not debug:  # if tqdm_label is not 'dev' and not debug:
                    raise CvxpyException(data=_error_obj)
                n_exceptions += 1
                logger.info(f'Caught CvxpyException {n_exceptions}: skipping batch')
                continue
            pred_cluster_ids = (model.hac_cut_layer.cluster_labels + (max_pred_id + 1)).tolist()
            cc_obj_vals['round'].append(model.hac_cut_layer.objective_value)
            cc_obj_vals['sdp'].append(model.sdp_layer.objective_value)
            cc_obj_vals['block_idxs'].append(idx)
            cc_obj_vals['block_sizes'].append(block_size)
        all_gold += list(np.reshape(cluster_ids, (block_size,)))
        max_pred_id = max(pred_cluster_ids)
        all_pred += list(pred_cluster_ids)
        if overfit_batch_idx > -1 and return_iter:
            return {
                'cluster_labels': model.hac_cut_layer.cluster_labels,
                'round_objective_value': model.hac_cut_layer.objective_value,
                'sdp_objective_value': model.sdp_layer.objective_value,
                'block_idx': idx,
                'block_size': block_size,
                'cluster_ids': cluster_ids
            }

    if fork_enabled and len(_procs) > 0:
        for _proc in tqdm(_procs, desc=f'Eval {tqdm_label} (waiting for forks to join)', position=tqdm_position):
            _proc.join()
        assert len(_procs) == len(_shared_list), "All forked eval iterations did not return results"
        for _data in _shared_list:
            pred_cluster_ids = (_data['cluster_labels'] + (max_pred_id + 1)).tolist()
            cc_obj_vals['round'].append(_data['round_objective_value'])
            cc_obj_vals['sdp'].append(_data['sdp_objective_value'])
            cc_obj_vals['block_idxs'].append(_data['block_idx'])
            cc_obj_vals['block_sizes'].append(_data['block_size'])
            all_gold += list(np.reshape(_data['cluster_ids'], (_data['block_size'],)))
            max_pred_id = max(pred_cluster_ids)
            all_pred += list(pred_cluster_ids)

    vmeasure = v_measure_score(all_gold, all_pred)
    b3_f1 = compute_b3_f1(all_gold, all_pred)[2]
    return b3_f1, vmeasure, cc_obj_vals


def evaluate_pairwise(model, dataloader, overfit_batch_idx=-1, mode="macro", return_pred_only=False,
                      thresh_for_f1=0.5, clustering_fn=None, clustering_threshold=None, val_dataloader=None,
                      tqdm_label='', device=None, verbose=False, debug=False, _errors=None, run_dir='./',
                      tqdm_position=None, model_args=None, return_iter=False, fork_size=50, disable_tqdm=False):
    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = dataloader.dataset[0][0].shape[1]

    if clustering_fn is not None:
        # Then dataloader passed is blockwise
        if clustering_fn.__class__ is HACInference:
            if clustering_threshold is None:
                clustering_fn.tune_threshold(model, val_dataloader, device)
        all_gold, all_pred = [], []
        cc_obj_vals = {
            'sdp': [],
            'round': [],
            'block_idxs': [],
            'block_sizes': []
        }
        max_pred_id = -1  # In each iteration, add to all blockwise predicted IDs to distinguish from previous blocks
        n_exceptions = 0
        pbar = tqdm(dataloader, desc=f'Eval {tqdm_label}', position=tqdm_position, disable=disable_tqdm)
        for (idx, batch) in enumerate(pbar):
            if overfit_batch_idx > -1:
                if idx < overfit_batch_idx:
                    continue
                if idx > overfit_batch_idx:
                    break
            data, _, cluster_ids = batch
            block_size = len(cluster_ids)
            pbar.set_description(f'Eval {tqdm_label} (sz={block_size})')
            data = data.reshape(-1, n_features).float()
            if data.shape[0] == 0:
                # Only one signature in block; manually assign a unique cluster
                pred_cluster_ids = [max_pred_id + 1]
            else:
                # Forward pass through the e2e model
                data = data.to(device)
                try:
                    edge_weights = model(data, N=block_size, warmstart=True, verbose=verbose)
                    pred_cluster_ids = clustering_fn(edge_weights, block_size, min_id=(max_pred_id + 1),
                                                     threshold=clustering_threshold)
                except CvxpyException as e:
                    logger.info(e)
                    _error_obj = {
                        'id': f'e_{int(time())}',
                        'method': 'eval',
                        'model_type': 'pairwise_cc',
                        'data_split': tqdm_label,
                        'model_call_args': {
                            'data': data.detach().tolist(),
                            'block_size': block_size
                        },
                        'cvxpy_layer_args': e.data
                    }
                    if _errors is not None:
                        _errors.append(_error_obj)
                        save_to_wandb_run({'errors': _errors}, 'errors.json', run_dir, logger)
                    if not debug:  # if tqdm_label is not 'dev' and not debug:
                        raise CvxpyException(data=_error_obj)
                    n_exceptions += 1
                    logger.info(f'Caught CvxpyException {n_exceptions}: skipping batch')
                    continue
            all_gold += list(np.reshape(cluster_ids, (block_size,)))
            max_pred_id = max(pred_cluster_ids)
            all_pred += list(pred_cluster_ids)
            if clustering_fn.__class__ is CCInference:
                cc_obj_vals['round'].append(clustering_fn.hac_cut_layer.objective_value)
                cc_obj_vals['sdp'].append(clustering_fn.sdp_layer.objective_value)
                cc_obj_vals['block_idxs'].append(idx)
                cc_obj_vals['block_sizes'].append(block_size)
        vmeasure = v_measure_score(all_gold, all_pred)
        b3_f1 = compute_b3_f1(all_gold, all_pred)[2]
        return (b3_f1, vmeasure, cc_obj_vals) if clustering_fn.__class__ is CCInference else (b3_f1, vmeasure)

    y_pred, targets = [], []
    pbar = tqdm(dataloader, desc=f'Eval {tqdm_label}', position=tqdm_position, disable=disable_tqdm)
    for (idx, batch) in enumerate(pbar):
        if overfit_batch_idx > -1:
            if idx < overfit_batch_idx:
                continue
            if idx > overfit_batch_idx:
                break
        data, target = batch
        data = data.reshape(-1, n_features).float()
        pbar.set_description(f'Eval {tqdm_label} (sz={len(data)})')
        assert data.shape[0] != 0
        target = target.flatten().float()
        # Forward pass through the pairwise model
        data = data.to(device)
        y_pred.append(torch.sigmoid(model(data, verbose=verbose)).cpu().numpy())
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
