import glob
import json
import os
import shutil
import time
import logging
import random
import copy

import wandb
import torch
import numpy as np
from tqdm import tqdm
from torch.multiprocessing import set_start_method, Manager

from e2e_pipeline.cc_inference import CCInference
from e2e_pipeline.hac_inference import HACInference
from e2e_pipeline.model import EntResModel
from e2e_pipeline.pairwise_model import PairwiseModel
from e2e_pipeline.sdp_layer import CvxpyException
from e2e_scripts.evaluate import evaluate, evaluate_pairwise
from e2e_scripts.train_utils import DEFAULT_HYPERPARAMS, get_dataloaders, get_matrix_size_from_triu, \
    uncompress_target_tensor, count_parameters, log_cc_objective_values, save_to_wandb_run, FrobeniusLoss, \
    get_feature_count, _check_process, fork_eval, init_eval, dev_eval
from utils.parser import Parser

from IPython import embed

try:
    set_start_method('spawn', force=True)
except RuntimeError:
    pass

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def train(hyperparams={}, verbose=False, project=None, entity=None, tags=None, group=None,
          save_model=False, load_model_from_wandb_run=None, load_model_from_fpath=None,
          skip_final_eval=False, eval_only_split=None, eval_all_only=False, skip_initial_eval=False,
          debug=False, track_errors=True, local=False, sync_dev=False):
    init_args = {
        'config': DEFAULT_HYPERPARAMS
    }
    if project is not None:
        init_args.update({'project': project})
    if entity is not None:
        init_args.update({'entity': entity})
    if tags is not None:
        tags = tags.replace(", ", ",").split(",")
        init_args.update({'tags': tags})
    if group is not None:
        init_args.update({'group': group})
    if local:
        init_args.update({'mode': 'disabled'})

    # Parallel process for validation runs
    _proc = None
    _return_dict = Manager().dict()
    _return_dict['_state'] = 'initial'

    # Start wandb run
    with wandb.init(**init_args) as run:
        if hyperparams.get('run_random_seed', 0) is None:
            hyperparams['run_random_seed'] = random.randint(0, 1000)
            logger.info(f"Generated random run seed: {hyperparams['run_random_seed']}")
        wandb.config.update(hyperparams, allow_val_change=True)
        hyp = wandb.config

        # Limit training epochs by dataset in e2e mode (for tractability)
        max_epochs_by_dataset = {
            'e2e': {
                'aminer': 3,
                'kisti': 3,
                'arnetminer': 5
            },
            'nosdp': {
                'aminer': 3,
                'kisti': 3,
                'arnetminer': 5
            }
        }
        n_epochs_override = None
        if not hyp['pairwise_mode']:
            _training_method = 'e2e' if hyp['use_sdp'] else 'nosdp'
            if hyp['dataset'] in max_epochs_by_dataset[_training_method] and \
                    hyp["n_epochs"] > max_epochs_by_dataset[_training_method][hyp['dataset']]:
                n_epochs_override = max_epochs_by_dataset[_training_method][hyp['dataset']]
                logger.info(f'Limiting number of epochs from {hyp["n_epochs"]} to {n_epochs_override}')

        logger.info("Run hyperparameters:")
        logger.info(hyp)
        # Save hyperparameters as a json file and store in wandb run
        save_to_wandb_run(dict(hyp), 'hyperparameters.json', run.dir, logger, error_logger=False)

        # Track errors
        _errors = [] if track_errors else None

        # Seed everything
        if hyp['run_random_seed'] is not None:
            random.seed(hyp['run_random_seed'])
            np.random.seed(hyp['run_random_seed'])
            torch.manual_seed(hyp['run_random_seed'])

        pairwise_mode = hyp['pairwise_mode']
        weighted_loss = hyp['weighted_loss']
        use_rounded_loss = hyp["use_rounded_loss"]
        e2e_loss = hyp['e2e_loss']
        batch_size = hyp['batch_size'] if pairwise_mode else 1  # Force clustering runs to operate on 1 block only
        n_epochs = n_epochs_override if n_epochs_override is not None else hyp['n_epochs']
        n_warmstart_epochs = hyp['n_warmstart_epochs']
        use_lr_scheduler = hyp['use_lr_scheduler']
        hidden_dim = hyp["hidden_dim"]
        n_hidden_layers = hyp["n_hidden_layers"]
        dropout_p = hyp["dropout_p"]
        dropout_only_once = hyp["dropout_only_once"]
        hidden_config = hyp["hidden_config"]
        activation = hyp["activation"]
        add_batchnorm = hyp["batchnorm"]
        add_layernorm = hyp["layernorm"]
        neumiss_deq = hyp["neumiss_deq"]
        neumiss_depth = hyp["neumiss_depth"]
        add_neumiss = not hyp['convert_nan']
        negative_slope = hyp["negative_slope"]
        use_sdp = hyp["use_sdp"]
        sdp_max_iters = hyp["sdp_max_iters"]
        sdp_eps = hyp["sdp_eps"]
        sdp_scale = hyp["sdp_scale"]
        grad_acc = hyp['batch_size'] if hyp["gradient_accumulation"] else 1
        overfit_batch_idx = hyp['overfit_batch_idx']
        clustering_metrics = {'b3_f1': 0, 'vmeasure': 1}
        pairwise_metrics = {'auroc': 0, 'f1': 1}
        eval_metric_to_idx = clustering_metrics if not pairwise_mode else pairwise_metrics
        dev_opt_metric = hyp['dev_opt_metric'] if hyp['dev_opt_metric'] in eval_metric_to_idx \
            else list(eval_metric_to_idx)[0]
        training_mode = not eval_all_only and eval_only_split is None
        early_terminate_epochs = hyp["early_terminate_epochs"]
        early_terminate_ctr = early_terminate_epochs if early_terminate_epochs > 0 else None

        # Get data loaders (optionally with imputation, normalization)
        if training_mode:
            train_dataloader, val_dataloader, test_dataloader = get_dataloaders(hyp["dataset"],
                                                                                hyp["dataset_random_seed"],
                                                                                hyp["convert_nan"], hyp["nan_value"],
                                                                                hyp["normalize_data"],
                                                                                hyp["subsample_sz_train"],
                                                                                hyp["subsample_sz_dev"],
                                                                                pairwise_mode=pairwise_mode,
                                                                                batch_size=batch_size,
                                                                                drop_feat_idxs=hyp["drop_feat_idxs"],
                                                                                keep_feat_idxs=hyp["keep_feat_idxs"],
                                                                                noise_std=hyp["noise_std"])
            n_features = train_dataloader.dataset[0][0].shape[1]
        else:
            n_features = get_feature_count(hyp["dataset"], hyp["dataset_random_seed"])

        # Create model with hyperparams
        if not pairwise_mode:
            model_args = (n_features, neumiss_depth, dropout_p, dropout_only_once, add_neumiss,
                         neumiss_deq, hidden_dim, n_hidden_layers, add_batchnorm, add_layernorm, activation,
                         negative_slope, hidden_config, sdp_max_iters, sdp_eps, sdp_scale,
                         use_rounded_loss, (e2e_loss == "bce"), use_sdp)
            model = EntResModel(*model_args)
            # Define eval
            eval_fn = evaluate
            pairwise_clustering_fns = [None]  # Unused when pairwise_mode is False

            if training_mode:  # => model will be used for training
                # Define loss
                if e2e_loss not in ["frob", "bce"]:
                    raise ValueError("Invalid value for e2e_loss")
                loss_fn_e2e = FrobeniusLoss() if e2e_loss == 'frob' else torch.nn.BCELoss()

                pos_weight = None
                if weighted_loss:
                    if overfit_batch_idx > -1:
                        n_pos = train_dataloader.dataset[overfit_batch_idx][1].sum()
                        pos_weight = (len(train_dataloader.dataset[overfit_batch_idx][1]) - n_pos) / n_pos
                    else:
                        _n_pos, _n_total = 0., 0.
                        for _i in range(len(train_dataloader.dataset)):
                            _n_pos += train_dataloader.dataset[_i][1].sum()
                            _n_total += len(train_dataloader.dataset[_i][1])
                        pos_weight = (_n_total - _n_pos) / _n_pos if _n_pos > 0 else 1.
                if n_warmstart_epochs > 0:
                    train_dataloader_pairwise = get_dataloaders(hyp["dataset"],
                                                                hyp["dataset_random_seed"],
                                                                hyp["convert_nan"],
                                                                hyp["nan_value"],
                                                                hyp["normalize_data"],
                                                                hyp["subsample_sz_train"],
                                                                hyp["subsample_sz_dev"],
                                                                pairwise_mode=True, batch_size=hyp['batch_size'],
                                                                split='train', drop_feat_idxs=hyp["drop_feat_idxs"],
                                                                keep_feat_idxs=hyp["keep_feat_idxs"], noise_std=hyp["noise_std"])
                    # Define loss
                    loss_fn_pairwise = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        else:
            model_args = (n_features, neumiss_depth, dropout_p, dropout_only_once, add_neumiss,
                          neumiss_deq, hidden_dim, n_hidden_layers, add_batchnorm, add_layernorm, activation,
                          negative_slope, hidden_config)
            model = PairwiseModel(*model_args)
            # Define eval
            eval_fn = evaluate_pairwise
            cc_inference = CCInference(sdp_max_iters, sdp_eps, sdp_scale, use_sdp)
            pairwise_clustering_fns = [cc_inference, HACInference(), cc_inference]
            pairwise_clustering_fns[0].eval()
            pairwise_clustering_fn_labels = ['cc', 'hac', 'cc-fixed']
            val_dataloader_e2e, test_dataloader_e2e = get_dataloaders(hyp["dataset"],
                                                                      hyp["dataset_random_seed"],
                                                                      hyp["convert_nan"],
                                                                      hyp["nan_value"],
                                                                      hyp["normalize_data"],
                                                                      hyp["subsample_sz_train"],
                                                                      hyp["subsample_sz_dev"],
                                                                      pairwise_mode=False, batch_size=1,
                                                                      split=['dev', 'test'],
                                                                      drop_feat_idxs=hyp["drop_feat_idxs"],
                                                                      keep_feat_idxs=hyp["keep_feat_idxs"],
                                                                      noise_std=hyp["noise_std"])
            if training_mode:  # => model will be used for training
                # Define loss
                pos_weight = None
                if weighted_loss:
                    if overfit_batch_idx > -1:
                        n_pos = \
                            train_dataloader.dataset[overfit_batch_idx * batch_size:(overfit_batch_idx + 1) * batch_size][
                                1].sum()
                        pos_weight = torch.tensor((batch_size - n_pos) / n_pos if n_pos > 0 else 1.)
                    else:
                        n_pos = train_dataloader.dataset[:][1].sum()
                        pos_weight = torch.tensor((len(train_dataloader.dataset) - n_pos) / n_pos if n_pos > 0 else 1.)
                loss_fn_pairwise = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        logger.info(f"Model loaded: {model}", )

        # Load stored model, if available
        state_dict = None
        if load_model_from_wandb_run is not None:
            state_dict_fpath = wandb.restore('model_state_dict_best.pt',
                                             run_path=load_model_from_wandb_run).name
            state_dict = torch.load(state_dict_fpath, device)
            os.remove(state_dict_fpath)
        elif load_model_from_fpath is not None:
            state_dict = torch.load(load_model_from_fpath, device)
        if state_dict is not None:
            model.load_state_dict(state_dict)
            logger.info(f'Loaded stored model.')
        model.to(device)

        if eval_all_only:
            # Run all inference variants on the test set and exit
            cc_inference_sdp = CCInference(sdp_max_iters, sdp_eps, sdp_scale, use_sdp=True)
            cc_inference_nosdp = CCInference(sdp_max_iters, sdp_eps, sdp_scale, use_sdp=False)
            inference_fns = [HACInference(),
                             cc_inference_sdp, cc_inference_sdp,
                             cc_inference_nosdp, cc_inference_nosdp]
            inference_fn_labels = ['hac',
                                   'cc', 'cc-fixed',
                                   'cc-nosdp', 'cc-nosdp-fixed']
            cc_inference_sdp.eval()
            cc_inference_nosdp.eval()
            val_dataloader_e2e, test_dataloader_e2e = get_dataloaders(hyp["dataset"],
                                                                      hyp["dataset_random_seed"],
                                                                      hyp["convert_nan"],
                                                                      hyp["nan_value"],
                                                                      hyp["normalize_data"],
                                                                      hyp["subsample_sz_train"],
                                                                      hyp["subsample_sz_dev"],
                                                                      pairwise_mode=False, batch_size=1,
                                                                      split=['dev', 'test'],
                                                                      drop_feat_idxs=hyp["drop_feat_idxs"],
                                                                      keep_feat_idxs=hyp["keep_feat_idxs"],
                                                                      noise_std=hyp["noise_std"])
            start_time = time.time()
            with torch.no_grad():
                model.eval()
                clustering_threshold = None
                for i, inference_fn in enumerate(inference_fns):
                    inf_start_time = time.time()
                    logger.info(f'Inference method: {inference_fn_labels[i]}')
                    clustering_scores = evaluate_pairwise(model, test_dataloader_e2e,
                                                          clustering_fn=inference_fn,
                                                          clustering_threshold=clustering_threshold if i % 2 == 0 else None,
                                                          val_dataloader=val_dataloader_e2e,
                                                          tqdm_label='test clustering', device=device, verbose=verbose,
                                                          debug=debug, _errors=_errors, model_args=model_args)
                    if inference_fn.__class__ is HACInference:
                        clustering_threshold = inference_fn.cut_threshold
                    logger.info(
                        f"Eval: test_{list(clustering_metrics)[0]}_{inference_fn_labels[i]}={clustering_scores[0]}, " +
                        f"test_{list(clustering_metrics)[1]}_{inference_fn_labels[i]}={clustering_scores[1]}")
                    # Log eval metrics
                    wandb.log({f'best_test_{list(clustering_metrics)[0]}_{inference_fn_labels[i]}':
                                   clustering_scores[0],
                               f'best_test_{list(clustering_metrics)[1]}_{inference_fn_labels[i]}':
                                   clustering_scores[1]})
                    if len(clustering_scores) == 3:
                        log_cc_objective_values(scores=clustering_scores,
                                                split_name=f'best_test_{inference_fn_labels[i]}',
                                                log_prefix='Eval', verbose=verbose, logger=logger)
                    inf_end_time = time.time()
                    inf_time = round(inf_end_time - inf_start_time)
                    wandb.log({f"z_inf_time_{inference_fn_labels[i]}": inf_time})
            end_time = time.time()
        elif eval_only_split is not None:
            # Run inference on the specified split and exit
            start_time = time.time()
            with torch.no_grad():
                model.eval()
                eval_dataloader = get_dataloaders(hyp["dataset"], hyp["dataset_random_seed"],
                                                  hyp["convert_nan"], hyp["nan_value"],
                                                  hyp["normalize_data"], hyp["subsample_sz_train"],
                                                  hyp["subsample_sz_dev"], pairwise_mode=pairwise_mode,
                                                  batch_size=batch_size, split=eval_only_split,
                                                  drop_feat_idxs=hyp["drop_feat_idxs"],
                                                  keep_feat_idxs=hyp["keep_feat_idxs"], noise_std=hyp["noise_std"])
                eval_scores = eval_fn(model, eval_dataloader, tqdm_label=eval_only_split, device=device, verbose=verbose,
                                      debug=debug, _errors=_errors, model_args=model_args, run_dir=run.dir)
                logger.info(f"Eval: {eval_only_split}_{list(eval_metric_to_idx)[0]}={eval_scores[0]}, " +
                            f"{eval_only_split}_{list(eval_metric_to_idx)[1]}={eval_scores[1]}")
                # Log eval metrics
                wandb.log({f'best_{eval_only_split}_{list(eval_metric_to_idx)[0]}': eval_scores[0],
                           f'best_{eval_only_split}_{list(eval_metric_to_idx)[1]}': eval_scores[1]})
                if len(eval_scores) == 3:
                    log_cc_objective_values(scores=eval_scores, split_name=eval_only_split, log_prefix='Eval',
                                            verbose=verbose, logger=logger)
                # For pairwise-mode:
                if pairwise_clustering_fns[0] is not None:
                    clustering_threshold = None
                    for i, pairwise_clustering_fn in enumerate(pairwise_clustering_fns):
                        clustering_scores = eval_fn(model, test_dataloader_e2e,  # Clustering only implemented for TEST
                                                    clustering_fn=pairwise_clustering_fn,
                                                    clustering_threshold=clustering_threshold,
                                                    val_dataloader=val_dataloader_e2e,
                                                    tqdm_label='test clustering', device=device, verbose=verbose,
                                                    debug=debug, _errors=_errors, model_args=model_args, run_dir=run.dir)
                        if pairwise_clustering_fn.__class__ is HACInference:
                            clustering_threshold = pairwise_clustering_fn.cut_threshold
                        logger.info(
                            f"Eval: test_{list(clustering_metrics)[0]}_{pairwise_clustering_fn_labels[i]}={clustering_scores[0]}, " +
                            f"test_{list(clustering_metrics)[1]}_{pairwise_clustering_fn_labels[i]}={clustering_scores[1]}")
                        # Log eval metrics
                        wandb.log({f'best_test_{list(clustering_metrics)[0]}_{pairwise_clustering_fn_labels[i]}':
                                       clustering_scores[0],
                                   f'best_test_{list(clustering_metrics)[1]}_{pairwise_clustering_fn_labels[i]}':
                                       clustering_scores[1]})
                        if len(clustering_scores) == 3:
                            log_cc_objective_values(scores=clustering_scores,
                                                    split_name=f'best_test_{pairwise_clustering_fn_labels[i]}',
                                                    log_prefix='Eval', verbose=verbose, logger=logger)
            end_time = time.time()
        else:
            # Train and evaluate
            wandb.watch(model)

            optimizer = torch.optim.AdamW(model.parameters(), lr=hyp['lr'], weight_decay=hyp['weight_decay'])
            if use_lr_scheduler:
                if hyp['lr_scheduler'] == 'plateau':
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                           mode='max',
                                                                           factor=hyp['lr_factor'],
                                                                           min_lr=hyp['lr_min'],
                                                                           patience=hyp['lr_scheduler_patience'],
                                                                           verbose=True)
                elif hyp['lr_scheduler'] == 'step':
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hyp['lr_step_size'],
                                                                gamma=hyp['lr_gamma'], verbose=True)

            best_dev_state_dict = copy.deepcopy(model.state_dict())
            best_dev_score = -1  # Stores the score of only the specified optimization metric
            best_dev_scores = ()  # Contains scores of all metrics
            best_epoch = -1

            if not skip_initial_eval:
                # Get initial model performance on dev (or 'train' for overfitting runs)
                _proc = fork_eval(target=init_eval, args=dict(model_args=model_args,
                                                              overfit_batch_idx=overfit_batch_idx, eval_fn=eval_fn,
                                                              train_dataloader=train_dataloader, device=device,
                                                              verbose=verbose,
                                                              debug=debug, _errors=_errors,
                                                              eval_metric_to_idx=eval_metric_to_idx,
                                                              val_dataloader=val_dataloader, return_dict=_return_dict,
                                                              run_dir=run.dir),
                                  model=model, run_dir=run.dir, device=device, logger=logger)
            if not pairwise_mode and grad_acc > 1:
                grad_acc_steps = []
                _seen_pw = 0
                _seen_blk = 0
                for d in train_dataloader.dataset:
                    _blk_sz = len(d[1])
                    _seen_pw += _blk_sz
                    _seen_blk += 1
                    if _seen_pw >= grad_acc:
                        grad_acc_steps.append(_seen_blk)
                        _seen_pw = 0
                        _seen_blk = 0
                if _seen_blk > 0:
                    grad_acc_steps.append(_seen_blk)
                logger.info(f"grad_acc step schedule: {grad_acc_steps}")

            model.train()
            start_time = time.time()  # Tracks full training runtime
            epoch_idx = -1
            early_terminate = False
            for epoch_idx in range(n_epochs):
                if early_terminate:
                    break
                _train_dataloader = train_dataloader
                loss_fn = loss_fn_e2e if not pairwise_mode else loss_fn_pairwise
                warmstart_mode = not pairwise_mode and epoch_idx < n_warmstart_epochs

                if warmstart_mode:
                    _train_dataloader = train_dataloader_pairwise
                    loss_fn = loss_fn_pairwise

                wandb.log({'epoch': epoch_idx + 1})
                running_loss = []
                n_exceptions = 0

                grad_acc_count = 0
                grad_acc_idx = 0
                optimizer.zero_grad()

                pbar = tqdm(_train_dataloader, desc=f"{'Warm-starting' if warmstart_mode else 'Training'} {epoch_idx + 1}",
                            position=1)
                for (batch_idx, batch) in enumerate(pbar):
                    best_epoch, best_dev_score, best_dev_scores, best_dev_state_dict, \
                        early_terminate_ctr, early_terminate = _check_process(_proc,
                                                                              _return_dict,
                                                                              logger, run,
                                                                              overfit_batch_idx,
                                                                              use_lr_scheduler,
                                                                              hyp, scheduler,
                                                                              eval_metric_to_idx,
                                                                              dev_opt_metric,
                                                                              epoch_idx - 1, best_epoch,
                                                                              best_dev_score,
                                                                              best_dev_scores,
                                                                              best_dev_state_dict,
                                                                              early_terminate_epochs, early_terminate_ctr,
                                                                              early_terminate, sync=sync_dev)
                    if early_terminate:
                        break
                    if overfit_batch_idx > -1:
                        if batch_idx < overfit_batch_idx:
                            continue
                        if batch_idx > overfit_batch_idx:
                            break
                    if not pairwise_mode and not warmstart_mode:
                        data, target, _ = batch
                    else:
                        data, target = batch
                    data = data.reshape(-1, n_features).float()
                    if data.shape[0] == 0:
                        # Block contains only one signature
                        continue
                    if add_batchnorm and data.shape[0] == 1:
                        # Block contains only one signature pair; batchnorm throws error
                        continue
                    block_size = get_matrix_size_from_triu(data)
                    pbar.set_description(f"{'Warm-starting' if warmstart_mode else 'Training'} {epoch_idx + 1} " + \
                                         f"(sz={len(data) if (pairwise_mode or warmstart_mode) else block_size})")
                    target = target.flatten().float()
                    if verbose:
                        logger.info(f"Batch shape: {data.shape}")
                        if not pairwise_mode and not warmstart_mode:
                            logger.info(f"Batch matrix size: {block_size}")

                    # Forward pass through the e2e or pairwise model
                    data, target = data.to(device), target.to(device)
                    try:
                        output = model(data, N=block_size, warmstart=warmstart_mode, verbose=verbose)
                    except CvxpyException as e:
                        logger.info(e)
                        _error_obj = {
                            'id': f'tf_{int(time.time())}',
                            'epoch_idx': epoch_idx,
                            'batch_idx': batch_idx,
                            'method': 'train_forward',
                            'model_type': 'e2e' if not pairwise_mode else 'pairwise',
                            'data_split': 'train',
                            'model_call_args': {
                                'data': data.detach().tolist(),
                                'block_size': block_size
                            },
                            'cvxpy_layer_args': e.data
                        }
                        if _errors is not None:
                            _errors.append(_error_obj)
                            save_to_wandb_run({'errors': _errors}, 'errors.json', run.dir, logger)
                        if debug:
                            n_exceptions += 1
                            logger.info(
                                f'Caught CvxpyException in forward call (count -> {n_exceptions}): skipping batch')
                            continue

                    # Calculate the loss
                    if not pairwise_mode and not warmstart_mode:
                        grad_acc_denom = 1 if grad_acc == 1 else grad_acc_steps[grad_acc_idx]
                        if e2e_loss != "bce":
                            target = uncompress_target_tensor(target, device=device)
                        if verbose:
                            logger.info(f"Gold:\n{target}")
                        if pos_weight is not None:
                            loss_weight = target * pos_weight + (1 - target)
                            loss_fn.weight = loss_weight
                        loss = loss_fn(output.view_as(target), target) / grad_acc_denom
                    else:
                        # Pairwise or warmstart mode
                        if verbose:
                            logger.info(f"Gold:\n{target}")
                        loss = loss_fn(output.view_as(target), target)

                    try:
                        loss.backward()
                        if not pairwise_mode and grad_acc > 1:
                            grad_acc_count += len(data)
                    except Exception as e:
                        logger.info(e)
                        if isinstance(e, CvxpyException):
                            _error_obj = {
                                'id': f'tb_{int(time.time())}',
                                'epoch_idx': epoch_idx,
                                'batch_idx': batch_idx,
                                'method': 'train_backward',
                                'model_type': 'e2e' if not pairwise_mode else 'pairwise',
                                'data_split': 'train',
                                'model_call_args': {
                                    'data': data.detach().tolist(),
                                    'block_size': block_size
                                }
                            }
                            if _errors is not None:
                                _errors.append(_error_obj)
                                save_to_wandb_run({'errors': _errors}, 'errors.json', run.dir, logger)
                            if debug:
                                n_exceptions += 1
                                logger.info(
                                    f'Caught CvxpyException in backward call (count -> {n_exceptions}): skipping batch')
                                continue
                    if pairwise_mode or (batch_idx == len(_train_dataloader) - 1) \
                            or grad_acc == 1 or grad_acc_count >= grad_acc:
                        if hyp["max_grad_norm"] != -1:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), hyp["max_grad_norm"]
                            )
                        optimizer.step()
                        optimizer.zero_grad()
                        if grad_acc > 1:
                            grad_acc_count = 0
                            grad_acc_idx += 1

                    if verbose:
                        logger.info(f"Loss = {loss.item()}")
                    running_loss.append(loss.item())
                    wandb.log({f'train_loss{"_warmstart" if warmstart_mode else ""}': np.mean(running_loss)})

                if warmstart_mode:
                    logger.info(f"Warmstart epoch loss = {np.mean(running_loss)}")
                    wandb.log({f'train_warmstart_epoch_loss': np.mean(running_loss)})
                else:
                    # Sync to get previous epoch's dev eval
                    best_epoch, best_dev_score, best_dev_scores, best_dev_state_dict, \
                        early_terminate_ctr, early_terminate = _check_process(_proc, _return_dict,
                                                                              logger, run,
                                                                              overfit_batch_idx,
                                                                              use_lr_scheduler,
                                                                              hyp, scheduler,
                                                                              eval_metric_to_idx,
                                                                              dev_opt_metric, epoch_idx - 1,
                                                                              best_epoch,
                                                                              best_dev_score,
                                                                              best_dev_scores,
                                                                              best_dev_state_dict,
                                                                              early_terminate_epochs, early_terminate_ctr,
                                                                              early_terminate, sync=True)
                    if early_terminate:
                        break

                    logger.info(f"Epoch loss = {np.mean(running_loss)}")
                    wandb.log({f'train_epoch_loss': np.mean(running_loss)})

                    # Get model performance on dev (or 'train' for overfitting runs)
                    _proc = fork_eval(target=dev_eval,
                                      args=dict(model_args=model_args,
                                                overfit_batch_idx=overfit_batch_idx, eval_fn=eval_fn,
                                                train_dataloader=train_dataloader, device=device,
                                                verbose=verbose, debug=debug, _errors=_errors,
                                                eval_metric_to_idx=eval_metric_to_idx, val_dataloader=val_dataloader,
                                                return_dict=_return_dict, epoch_idx=epoch_idx, run_dir=run.dir),
                                      model=model, run_dir=run.dir, device=device, logger=logger,
                                      sync=(epoch_idx == n_epochs - 1))
            end_time = time.time()
            if early_terminate:
                logger.info(f"Early termination executed: no dev improvement in {early_terminate_epochs} epochs")
            if _proc is not None:
                best_epoch, best_dev_score, best_dev_scores, best_dev_state_dict, \
                    _, _ = _check_process(_proc,
                                          _return_dict,
                                          logger, run,
                                          overfit_batch_idx,
                                          use_lr_scheduler,
                                          hyp, scheduler,
                                          eval_metric_to_idx,
                                          dev_opt_metric,
                                          epoch_idx,
                                          best_epoch,
                                          best_dev_score,
                                          best_dev_scores,
                                          best_dev_state_dict,
                                          early_terminate_epochs, early_terminate_ctr,
                                          early_terminate, sync=True)
            # Save model
            if save_model:
                torch.save(best_dev_state_dict, os.path.join(run.dir, 'model_state_dict_best.pt'))
                wandb.save('model_state_dict_best.pt')
                logger.info(f"Saved best model on dev to {os.path.join(run.dir, 'model_state_dict_best.pt')}")

            # Evaluate the best dev model on test
            if not skip_final_eval:
                if overfit_batch_idx == -1:
                    model.load_state_dict(best_dev_state_dict)
                    with torch.no_grad():
                        model.eval()
                        test_scores = eval_fn(model, test_dataloader, tqdm_label='test', device=device, verbose=verbose,
                                              debug=debug, _errors=_errors, tqdm_position=2, model_args=model_args,
                                              run_dir=run.dir)
                        logger.info(f"Final: test_{list(eval_metric_to_idx)[0]}={test_scores[0]}, " +
                                    f"test_{list(eval_metric_to_idx)[1]}={test_scores[1]}")
                        # Log final metrics
                        _wandb_log = {'best_dev_epoch': best_epoch + 1,
                                      f'best_test_{list(eval_metric_to_idx)[0]}': test_scores[0],
                                      f'best_test_{list(eval_metric_to_idx)[1]}': test_scores[1]
                                      }
                        if len(best_dev_scores) >= 2:
                            _wandb_log.update({
                                f'best_dev_{list(eval_metric_to_idx)[0]}': best_dev_scores[0],
                                f'best_dev_{list(eval_metric_to_idx)[1]}': best_dev_scores[1]
                            })
                        wandb.log(_wandb_log)
                        if len(test_scores) == 3:
                            log_cc_objective_values(scores=test_scores, split_name='best_test', log_prefix='Final',
                                                    verbose=True, logger=logger)
                    # Run all inference variants on the test set
                    cc_inference_sdp = CCInference(sdp_max_iters, sdp_eps, sdp_scale, use_sdp=True)
                    cc_inference_nosdp = CCInference(sdp_max_iters, sdp_eps, sdp_scale, use_sdp=False)
                    inference_fns = [HACInference(),
                                     cc_inference_sdp, cc_inference_sdp,
                                     cc_inference_nosdp, cc_inference_nosdp]
                    inference_fn_labels = ['hac',
                                           'cc', 'cc-fixed',
                                           'cc-nosdp', 'cc-nosdp-fixed']
                    cc_inference_sdp.eval()
                    cc_inference_nosdp.eval()
                    if not pairwise_mode:
                        val_dataloader_e2e, test_dataloader_e2e = val_dataloader, test_dataloader
                    with torch.no_grad():
                        model.eval()
                        clustering_threshold = None
                        total_inf_time = 0
                        for i, inference_fn in enumerate(inference_fns):
                            inf_start_time = time.time()
                            logger.info(f'Inference method: {inference_fn_labels[i]}')
                            clustering_scores = evaluate_pairwise(model, test_dataloader_e2e,
                                                                  clustering_fn=inference_fn,
                                                                  clustering_threshold=clustering_threshold if i % 2 == 0 else None,
                                                                  val_dataloader=val_dataloader_e2e,
                                                                  tqdm_label='test clustering', device=device,
                                                                  verbose=verbose, debug=debug, _errors=_errors,
                                                                  model_args=model_args, run_dir=run.dir)
                            if inference_fn.__class__ is HACInference:
                                clustering_threshold = inference_fn.cut_threshold
                            logger.info(
                                f"Eval: test_{list(clustering_metrics)[0]}_{inference_fn_labels[i]}={clustering_scores[0]}, " +
                                f"test_{list(clustering_metrics)[1]}_{inference_fn_labels[i]}={clustering_scores[1]}")
                            # Log eval metrics
                            wandb.log({f'best_test_{list(clustering_metrics)[0]}_{inference_fn_labels[i]}':
                                           clustering_scores[0],
                                       f'best_test_{list(clustering_metrics)[1]}_{inference_fn_labels[i]}':
                                           clustering_scores[1]})
                            if len(clustering_scores) == 3:
                                log_cc_objective_values(scores=clustering_scores,
                                                        split_name=f'best_test_{inference_fn_labels[i]}',
                                                        log_prefix='Eval', verbose=verbose, logger=logger)
                            inf_end_time = time.time()
                            inf_time = round(inf_end_time - inf_start_time)
                            total_inf_time += inf_time
                            wandb.log({f"z_inf_time_{inference_fn_labels[i]}": inf_time})
                    run.summary["z_inf_time"] = total_inf_time

        run.summary["z_model_parameters"] = count_parameters(model)
        run.summary["z_run_time"] = round(end_time - start_time)  # Train time if in training mode
        run.summary["z_run_dir_path"] = run.dir

        if _errors is not None:
            _all_errors = save_to_wandb_run({'errors': _errors}, 'errors.json', run.dir, logger)
            if len(_all_errors['errors']) > 0:
                logger.warning(f'Errors were encountered during the run. LOGS: {os.path.join(run.dir, "errors.json")}')
        # Cleanup
        for filename in glob.glob(os.path.join(run.dir, "_temp_state_dict*")):
            os.remove(filename)
        logger.info(f"Run directory: {run.dir}")
        logger.info("End of train() call")


if __name__ == '__main__':
    # Read cmd line args
    parser = Parser(add_training_args=True)
    # Handle additional arbitrary arguments
    _, unknown = parser.parse_known_args()
    make_false_args = []
    def _type_or_none_fn(argtype):
        def __type_or_none_fn(value):
            if value.lower() in ['none', 'null']:
                return None
            return argtype(value)
        return __type_or_none_fn
    for arg in unknown:
        if arg.startswith("--"):
            arg_split = arg.split('=')
            argument_name = arg_split[0]
            if argument_name[2:] in DEFAULT_HYPERPARAMS:  # "--argument_name" without the "--"
                argument_type = type(DEFAULT_HYPERPARAMS[argument_name[2:]])
                if argument_type == bool:
                    if len(arg_split) > 1 and arg_split[1].lower() == 'false':
                        parser.add_argument(argument_name, type=str)
                        make_false_args.append(argument_name[2:])
                    else:
                        parser.add_argument(argument_name, action='store_true')
                elif argument_type == list:
                    parser.add_argument(argument_name, action='append')
                else:
                    parser.add_argument(argument_name,
                                        type=_type_or_none_fn(argument_type), default=None)
    args = parser.parse_args().__dict__
    for false_arg in make_false_args:
        args[false_arg] = False
    hyp_args = {k: v for k, v in args.items() if k in DEFAULT_HYPERPARAMS}
    logger.info("Script arguments:")
    logger.info(args)

    if args['cpu']:
        device = torch.device("cpu")
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    logger.info(f"Using device={device}")

    wandb.login()
    if args['wandb_sweep_params'] is not None:
        logger.info("Sweep mode")
        with open(args['wandb_sweep_params'], 'r') as fh:
            sweep_params = json.load(fh)

        sweep_config = {
            'method': args['wandb_sweep_method'],
            'name': args['wandb_sweep_name'],
            'metric': {
                'name': 'best_dev_auroc' if hyp_args.get('pairwise_mode', False) else 'best_dev_b3_f1',
                # args['wandb_sweep_metric_name'],
                'goal': args['wandb_sweep_metric_goal'],
            },
            'parameters': sweep_params,
        }
        if not args['wandb_no_early_terminate']:
            sweep_config.update({
                'early_terminate': {
                    'type': 'hyperband',
                    'min_iter': 4,
                    'eta': 2
                }
            })

        # Init sweep
        sweep_id = args["wandb_sweep_id"]
        if sweep_id is None:
            sweep_id = wandb.sweep(sweep=sweep_config,
                                   project=args['wandb_project'],
                                   entity=args['wandb_entity'])
            logger.info(f"SWEEP_ID={sweep_id}")

        # Start sweep job
        wandb.agent(sweep_id,
                    function=lambda: train(hyperparams=hyp_args,
                                           verbose=not args['silent'],
                                           tags=args['wandb_tags'],
                                           save_model=args['save_model'],
                                           skip_initial_eval=args['skip_initial_eval'],
                                           skip_final_eval=args['skip_final_eval'],
                                           debug=args['debug'],
                                           track_errors=not args['no_error_tracking'],
                                           local=args['local'],
                                           sync_dev=args['sync_dev']),
                    count=args['wandb_max_runs'])

        logger.info("End of sweep")
    else:
        logger.info("Single-run mode")
        try:
            if args['load_hyp_from_wandb_run'] is not None:
                temp_fpath = f'./temp/{args["load_hyp_from_wandb_run"].split("/")[-1]}-{time.time()}'
                os.makedirs(temp_fpath, exist_ok=True)
                cwd = os.getcwd()
                os.chdir(temp_fpath)
                run_params_fpath = wandb.restore('hyperparameters.json', run_path=args['load_hyp_from_wandb_run']).name
                with open(run_params_fpath, 'r') as fh:
                    run_params = json.load(fh)
                os.remove(run_params_fpath)
                shutil.rmtree(temp_fpath)
                os.chdir(cwd)
            else:
                with open(args['wandb_run_params'], 'r') as fh:
                    run_params = json.load(fh)
        except:
            logger.info("Run config could not be loaded; using defaults.")
            run_params = {}
        run_params.update(hyp_args)
        train(hyperparams=run_params,
              verbose=not args['silent'],
              project=args['wandb_project'],
              entity=args['wandb_entity'],
              tags=args['wandb_tags'],
              group=args['wandb_group'],
              save_model=args['save_model'],
              load_model_from_wandb_run=args['load_model_from_wandb_run'],
              load_model_from_fpath=args['load_model_from_fpath'],
              eval_only_split=args['eval_only_split'],
              eval_all_only=args['eval_all_only'],
              skip_initial_eval=args['skip_initial_eval'],
              debug=args['debug'],
              track_errors=not args['no_error_tracking'],
              local=args['local'],
              sync_dev=args['sync_dev'],
              skip_final_eval=args['skip_final_eval'])
        logger.info("End of run")
