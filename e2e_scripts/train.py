import json
import os
import time
import logging
import random
import copy

import wandb
import torch
import numpy as np

from tqdm import tqdm

from e2e_pipeline.cc_inference import CCInference
from e2e_pipeline.hac_inference import HACInference
from e2e_pipeline.model import EntResModel
from e2e_pipeline.pairwise_model import PairwiseModel
from e2e_pipeline.sdp_layer import CvxpyException
from e2e_scripts.evaluate import evaluate, evaluate_pairwise
from e2e_scripts.train_utils import DEFAULT_HYPERPARAMS, get_dataloaders, get_matrix_size_from_triu, \
    uncompress_target_tensor, count_parameters, log_cc_objective_values, save_to_wandb_run
from utils.parser import Parser

from IPython import embed


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def train(hyperparams={}, verbose=False, project=None, entity=None, tags=None, group=None,
          save_model=False, load_model_from_wandb_run=None, load_model_from_fpath=None,
          eval_only_split=None, skip_initial_eval=False, pairwise_eval_clustering=None,
          debug=False, track_errors=False):
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

    # Start wandb run
    with wandb.init(**init_args) as run:
        wandb.config.update(hyperparams, allow_val_change=True)
        hyp = wandb.config
        logger.info("Run hyperparameters:")
        logger.info(hyp)
        # Save hyperparameters as a json file and store in wandb run
        save_to_wandb_run(dict(hyp), 'hyperparameters.json', run.dir, logger)

        # Track errors
        _errors = [] if track_errors else None

        # Seed everything
        if hyp['run_random_seed'] is not None:
            random.seed(hyp['run_random_seed'])
            np.random.seed(hyp['run_random_seed'])
            torch.manual_seed(hyp['run_random_seed'])

        pairwise_mode = hyp['pairwise_mode']
        weighted_loss = hyp['weighted_loss']
        # normalize_loss = hyp['normalize_loss']
        batch_size = hyp['batch_size'] if pairwise_mode else 1  # Force clustering runs to operate on 1 block only
        n_epochs = hyp['n_epochs']
        n_warmstart_epochs = hyp['n_warmstart_epochs']
        use_lr_scheduler = hyp['use_lr_scheduler']
        hidden_dim = hyp["hidden_dim"]
        n_hidden_layers = hyp["n_hidden_layers"]
        dropout_p = hyp["dropout_p"]
        dropout_only_once = hyp["dropout_only_once"]
        hidden_config = hyp["hidden_config"]
        activation = hyp["activation"]
        add_batchnorm = hyp["batchnorm"]
        neumiss_deq = hyp["neumiss_deq"]
        neumiss_depth = hyp["neumiss_depth"]
        add_neumiss = not hyp['convert_nan']
        negative_slope = hyp["negative_slope"]
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

        # Get data loaders (optionally with imputation, normalization)
        train_dataloader, val_dataloader, test_dataloader = get_dataloaders(hyp["dataset"], hyp["dataset_random_seed"],
                                                                            hyp["convert_nan"], hyp["nan_value"],
                                                                            hyp["normalize_data"], hyp["subsample_sz_train"],
                                                                            hyp["subsample_sz_dev"], pairwise_mode,
                                                                            batch_size)
        n_features = train_dataloader.dataset[0][0].shape[1]

        # Create model with hyperparams
        if not pairwise_mode:
            model = EntResModel(n_features, neumiss_depth, dropout_p, dropout_only_once, add_neumiss,
                                neumiss_deq, hidden_dim, n_hidden_layers, add_batchnorm, activation,
                                negative_slope, hidden_config, sdp_max_iters, sdp_eps, sdp_scale,
                                use_rounded_loss=hyp["use_rounded_loss"], return_triu_on_train=True)  # TODO: Control with flag
            # Define loss
            # loss_fn_e2e = lambda pred, gold: torch.norm(gold - pred)
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
                        pos_weight = (_n_total - _n_pos) / _n_pos
            loss_fn_e2e = torch.nn.BCELoss()
            # Define eval
            eval_fn = evaluate
            pairwise_clustering_fns = [None]  # Unused when pairwise_mode is False
            if n_warmstart_epochs > 0:
                train_dataloader_pairwise, _, _ = get_dataloaders(hyp["dataset"],
                                                                  hyp["dataset_random_seed"],
                                                                  hyp["convert_nan"],
                                                                  hyp["nan_value"],
                                                                  hyp["normalize_data"],
                                                                  hyp["subsample_sz_train"],
                                                                  hyp["subsample_sz_dev"],
                                                                  True, hyp['batch_size'])
                # Define loss
                pos_weight = None
                if weighted_loss and overfit_batch_idx == -1:
                    n_pos = train_dataloader_pairwise.dataset[:][1].sum()
                    pos_weight = torch.tensor((len(train_dataloader_pairwise.dataset) - n_pos) / n_pos)
                loss_fn_pairwise = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            model = PairwiseModel(n_features, neumiss_depth, dropout_p, dropout_only_once, add_neumiss,
                                  neumiss_deq, hidden_dim, n_hidden_layers, add_batchnorm, activation,
                                  negative_slope, hidden_config)
            # Define loss
            pos_weight = None
            if weighted_loss:
                if overfit_batch_idx > -1:
                    n_pos = \
                        train_dataloader.dataset[overfit_batch_idx * batch_size:(overfit_batch_idx + 1) * batch_size][
                            1].sum()
                    pos_weight = torch.tensor((batch_size - n_pos) / n_pos)
                else:
                    n_pos = train_dataloader.dataset[:][1].sum()
                    pos_weight = torch.tensor((len(train_dataloader.dataset) - n_pos) / n_pos)
            loss_fn_pairwise = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            # Define eval
            eval_fn = evaluate_pairwise
            pairwise_clustering_fns = [None]
            if pairwise_eval_clustering is not None:
                if pairwise_eval_clustering == 'cc':
                    pairwise_clustering_fns = [CCInference(sdp_max_iters, sdp_eps, sdp_scale)]
                    pairwise_clustering_fns[0].eval()
                    pairwise_clustering_fn_labels = ['cc']
                elif pairwise_eval_clustering == 'hac':
                    pairwise_clustering_fns = [HACInference()]
                    pairwise_clustering_fn_labels = ['hac']
                elif pairwise_eval_clustering == 'both':
                    cc_inference = CCInference(sdp_max_iters, sdp_eps, sdp_scale)
                    pairwise_clustering_fns = [cc_inference, HACInference(), cc_inference]
                    pairwise_clustering_fns[0].eval()
                    pairwise_clustering_fn_labels = ['cc', 'hac', 'cc-fixed']
                else:
                    raise ValueError('Invalid argument passed to --pairwise_eval_clustering')
                _, val_dataloader_e2e, test_dataloader_e2e = get_dataloaders(hyp["dataset"],
                                                                             hyp["dataset_random_seed"],
                                                                             hyp["convert_nan"],
                                                                             hyp["nan_value"],
                                                                             hyp["normalize_data"],
                                                                             hyp["subsample_sz_train"],
                                                                             hyp["subsample_sz_dev"],
                                                                             False, 1)
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

        if eval_only_split is not None:
            # Run inference and exit
            dataloaders = {
                'train': train_dataloader,
                'dev': val_dataloader,
                'test': test_dataloader
            }
            start_time = time.time()
            with torch.no_grad():
                model.eval()

                eval_dataloader = dataloaders[eval_only_split]
                eval_scores = eval_fn(model, eval_dataloader, tqdm_label=eval_only_split, device=device, verbose=verbose,
                                      debug=debug, _errors=_errors)
                if verbose:
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
                                                    debug=debug, _errors=_errors)
                        if pairwise_clustering_fn.__class__ is HACInference:
                            clustering_threshold = pairwise_clustering_fn.cut_threshold
                        if verbose:
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
            # Training
            wandb.watch(model)

            optimizer = torch.optim.AdamW(model.parameters(), lr=hyp['lr'])
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

            best_dev_state_dict = None
            best_dev_score = -1  # Stores the score of only the specified optimization metric
            best_dev_scores = None  # Contains scores of all metrics
            best_epoch = 0

            if not skip_initial_eval:
                # Get initial model performance on dev (or 'train' for overfitting runs)
                with torch.no_grad():
                    model.eval()
                    if overfit_batch_idx > -1:
                        train_scores = eval_fn(model, train_dataloader, overfit_batch_idx=overfit_batch_idx,
                                               tqdm_label='train', device=device, verbose=verbose, debug=debug,
                                               _errors=_errors)
                        if verbose:
                            logger.info(f"Initial: train_{list(eval_metric_to_idx)[0]}={train_scores[0]}, " +
                                        f"train_{list(eval_metric_to_idx)[1]}={train_scores[1]}")
                        wandb.log({'epoch': 0, f'train_{list(eval_metric_to_idx)[0]}': train_scores[0],
                                   f'train_{list(eval_metric_to_idx)[1]}': train_scores[1]})
                    else:
                        dev_scores = eval_fn(model, val_dataloader, tqdm_label='dev', device=device, verbose=verbose,
                                             debug=debug, _errors=_errors)
                        if verbose:
                            logger.info(f"Initial: dev_{list(eval_metric_to_idx)[0]}={dev_scores[0]}, " +
                                        f"dev_{list(eval_metric_to_idx)[1]}={dev_scores[1]}")
                        wandb.log({'epoch': 0, f'dev_{list(eval_metric_to_idx)[0]}': dev_scores[0],
                                   f'dev_{list(eval_metric_to_idx)[1]}': dev_scores[1]})

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

            model.train()
            start_time = time.time()  # Tracks full training runtime
            for i in range(n_epochs):
                _train_dataloader = train_dataloader
                loss_fn = loss_fn_e2e if not pairwise_mode else loss_fn_pairwise
                warmstart_mode = not pairwise_mode and i < n_warmstart_epochs

                if warmstart_mode:
                    _train_dataloader = train_dataloader_pairwise
                    loss_fn = loss_fn_pairwise

                wandb.log({'epoch': i + 1})
                running_loss = []
                n_exceptions = 0

                grad_acc_count = 0
                grad_acc_idx = 0
                optimizer.zero_grad()

                for (idx, batch) in enumerate(tqdm(_train_dataloader,
                                                   desc=f"{'Warm-starting' if warmstart_mode else 'Training'} {i + 1}",
                                                   disable=(not verbose))):
                    if overfit_batch_idx > -1:
                        if idx < overfit_batch_idx:
                            continue
                        if idx > overfit_batch_idx:
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
                        _error_obj = {
                            'method': 'train_forward',
                            'model_type': 'e2e' if not pairwise_mode else 'pairwise',
                            'data_split': 'train',
                            'model_call_args': {
                                'data': data.detach().cpu(),
                                'block_size': block_size
                            },
                            'cvxpy_layer_args': e.data
                        }
                        if _errors is not None:
                            _errors.append(_error_obj)
                        n_exceptions += 1
                        logger.info(f'Caught CvxpyException in forward call (count -> {n_exceptions}): skipping batch')
                        continue

                    # Calculate the loss
                    if not pairwise_mode and not warmstart_mode:
                        # gold_output = uncompress_target_tensor(target, device=device)
                        if verbose:
                            logger.info(f"Gold:\n{target}")  # gold_output
                        if pos_weight is not None:
                            with torch.no_grad():
                                loss_weight = target * pos_weight + (1 - target)
                            loss_fn.weight = loss_weight
                        loss = loss_fn(output.view_as(target), target) / (
                            1 if grad_acc == 1 else grad_acc_steps[grad_acc_idx])
                        # loss = loss_fn(output.view_as(gold_output), gold_output) / (
                        #     (block_size * (block_size - 1)) if normalize_loss else 1) / (  # (2 * block_size)
                        #            1 if grad_acc == 1 else grad_acc_steps[-1])
                    else:
                        if verbose:
                            logger.info(f"Gold:\n{target}")
                        loss = loss_fn(output.view_as(target), target)

                    try:
                        loss.backward()
                        if not pairwise_mode and grad_acc > 1:
                            grad_acc_count += len(data)
                    except:
                        _error_obj = {
                            'method': 'train_backward',
                            'model_type': 'e2e' if not pairwise_mode else 'pairwise',
                            'data_split': 'train',
                            'model_call_args': {
                                'data': data.detach().cpu(),
                                'block_size': block_size
                            }
                        }
                        if _errors is not None:
                            _errors.append(_error_obj)
                        n_exceptions += 1
                        logger.info(f'Caught CvxpyException in backward call (count -> {n_exceptions}): skipping batch')
                        continue
                    if pairwise_mode or (
                            idx == len(_train_dataloader.dataset) - 1) or grad_acc == 1 or grad_acc_count >= grad_acc:
                        optimizer.step()
                        optimizer.zero_grad()
                        if grad_acc > 1:
                            grad_acc_count = 0
                            grad_acc_idx += 1

                    if verbose:
                        logger.info(f"Loss = {loss.item()}")
                    running_loss.append(loss.item())
                    wandb.log({f'train_loss{"_warmstart" if warmstart_mode else ""}': np.mean(running_loss)})

                if verbose:
                    logger.info(f"Epoch loss = {np.mean(running_loss)}")

                # Get model performance on dev (or 'train' for overfitting runs)
                with torch.no_grad():
                    model.eval()
                    if overfit_batch_idx > -1:
                        train_scores = eval_fn(model, train_dataloader, overfit_batch_idx=overfit_batch_idx,
                                               tqdm_label='train', device=device, verbose=verbose, debug=debug,
                                               _errors=_errors)
                        if verbose:
                            logger.info(f"Epoch {i + 1}: train_{list(eval_metric_to_idx)[0]}={train_scores[0]}, " +
                                        f"train_{list(eval_metric_to_idx)[1]}={train_scores[1]}")
                        wandb.log({f'train_{list(eval_metric_to_idx)[0]}': train_scores[0],
                                   f'train_{list(eval_metric_to_idx)[1]}': train_scores[1]})
                        if use_lr_scheduler:
                            if hyp['lr_scheduler'] == 'plateau':
                                scheduler.step(train_scores[eval_metric_to_idx[dev_opt_metric]])
                            elif hyp['lr_scheduler'] == 'step':
                                scheduler.step()
                    else:
                        dev_scores = eval_fn(model, val_dataloader, tqdm_label='dev', device=device, verbose=verbose,
                                             debug=debug, _errors=_errors)
                        if verbose:
                            logger.info(f"Epoch {i + 1}: dev_{list(eval_metric_to_idx)[0]}={dev_scores[0]}, " +
                                        f"dev_{list(eval_metric_to_idx)[1]}={dev_scores[1]}")
                        wandb.log({f'dev_{list(eval_metric_to_idx)[0]}': dev_scores[0],
                                   f'dev_{list(eval_metric_to_idx)[1]}': dev_scores[1],
                                   f'train_epoch_loss': np.mean(running_loss)})
                        dev_opt_score = dev_scores[eval_metric_to_idx[dev_opt_metric]]
                        if dev_opt_score > best_dev_score:
                            if verbose:
                                logger.info(f"New best dev {dev_opt_metric} score @ epoch{i+1}: {dev_opt_score}")
                            best_epoch = i
                            best_dev_score = dev_opt_score
                            best_dev_scores = dev_scores
                            best_dev_state_dict = copy.deepcopy(model.state_dict())
                        if use_lr_scheduler:
                            if hyp['lr_scheduler'] == 'plateau':
                                scheduler.step(dev_scores[eval_metric_to_idx[dev_opt_metric]])
                            elif hyp['lr_scheduler'] == 'step':
                                scheduler.step()
                model.train()
            end_time = time.time()

            if overfit_batch_idx == -1:
                # Evaluate the best dev model on test
                model.load_state_dict(best_dev_state_dict)
                with torch.no_grad():
                    model.eval()
                    test_scores = eval_fn(model, test_dataloader, tqdm_label='test', device=device, verbose=verbose,
                                          debug=debug, _errors=_errors)
                    if verbose:
                        logger.info(f"Final: test_{list(eval_metric_to_idx)[0]}={test_scores[0]}, " +
                                    f"test_{list(eval_metric_to_idx)[1]}={test_scores[1]}")
                    # Log final metrics
                    wandb.log({'best_dev_epoch': best_epoch + 1,
                               f'best_dev_{list(eval_metric_to_idx)[0]}': best_dev_scores[0],
                               f'best_dev_{list(eval_metric_to_idx)[1]}': best_dev_scores[1],
                               f'best_test_{list(eval_metric_to_idx)[0]}': test_scores[0],
                               f'best_test_{list(eval_metric_to_idx)[1]}': test_scores[1]})
                    if len(test_scores) == 3:
                        log_cc_objective_values(scores=test_scores, split_name='best_test', log_prefix='Final',
                                                verbose=verbose, logger=logger)
                    # For pairwise-mode:
                    if pairwise_clustering_fns[0] is not None:
                        clustering_threshold = None
                        for i, pairwise_clustering_fn in enumerate(pairwise_clustering_fns):
                            clustering_scores = eval_fn(model, test_dataloader_e2e,
                                                        clustering_fn=pairwise_clustering_fn,
                                                        clustering_threshold=clustering_threshold,
                                                        val_dataloader=val_dataloader_e2e,
                                                        tqdm_label='test clustering', device=device, verbose=verbose,
                                                        debug=debug, _errors=_errors)
                            if pairwise_clustering_fn.__class__ is HACInference:
                                clustering_threshold = pairwise_clustering_fn.cut_threshold
                            if verbose:
                                logger.info(f"Final: test_{list(clustering_metrics)[0]}_{pairwise_clustering_fn_labels[i]}={clustering_scores[0]}, " +
                                            f"test_{list(clustering_metrics)[1]}_{pairwise_clustering_fn_labels[i]}={clustering_scores[1]}")
                            # Log final metrics
                            wandb.log({f'best_test_{list(clustering_metrics)[0]}_{pairwise_clustering_fn_labels[i]}': clustering_scores[0],
                                       f'best_test_{list(clustering_metrics)[1]}_{pairwise_clustering_fn_labels[i]}': clustering_scores[1]})
                            if len(clustering_scores) == 3:
                                log_cc_objective_values(scores=clustering_scores,
                                                        split_name=f'best_test_{pairwise_clustering_fn_labels[i]}',
                                                        log_prefix='Final', verbose=verbose, logger=logger)


        run.summary["z_model_parameters"] = count_parameters(model)
        run.summary["z_run_time"] = round(end_time - start_time)
        run.summary["z_run_dir_path"] = run.dir

        # Save models
        if save_model:
            torch.save(best_dev_state_dict, os.path.join(run.dir, 'model_state_dict_best.pt'))
            wandb.save('model_state_dict_best.pt')
            logger.info(f"Saved best model on dev to {os.path.join(run.dir, 'model_state_dict_best.pt')}")

        if _errors is not None:
            save_to_wandb_run({'errors': _errors}, 'errors.json', run.dir, logger)

        logger.info(f"Run directory: {run.dir}")
        logger.info("End of train() call")


if __name__ == '__main__':
    # Read cmd line args
    parser = Parser(add_training_args=True)
    # Handle additional arbitrary arguments
    _, unknown = parser.parse_known_args()
    make_false_args = []
    for arg in unknown:
        if arg.startswith("--"):
            arg_split = arg.split('=')
            argument_name = arg_split[0]
            if argument_name[2:] in DEFAULT_HYPERPARAMS:
                argument_type = type(DEFAULT_HYPERPARAMS[argument_name[2:]])
                if argument_type == bool:
                    if len(arg_split) > 1 and arg_split[1].lower() == 'false':
                        parser.add_argument(argument_name, type=str)
                        make_false_args.append(argument_name[2:])
                    else:
                        parser.add_argument(argument_name, action='store_true')
                else:
                    parser.add_argument(argument_name, type=argument_type)
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
                'name': args['wandb_sweep_metric_name'],
                'goal': args['wandb_sweep_metric_goal'],
            },
            'parameters': sweep_params,
        }
        if not args['wandb_no_early_terminate']:
            sweep_config.update({
                'early_terminate': {
                    'type': 'hyperband',
                    'min_iter': 5
                }
            })

        # Init sweep
        sweep_id = args["wandb_sweep_id"]
        if sweep_id is None:
            sweep_id = wandb.sweep(sweep=sweep_config,
                                   project=args['wandb_project'],
                                   entity=args['wandb_entity'])

        # Start sweep job
        wandb.agent(sweep_id,
                    function=lambda: train(hyperparams=hyp_args,
                                           verbose=not args['silent'],
                                           tags=args['wandb_tags'],
                                           save_model=args['save_model'],
                                           skip_initial_eval=args['skip_initial_eval'],
                                           debug=args['debug'],
                                           track_errors=args['track_errors']),
                    count=args['wandb_max_runs'])

        logger.info("End of sweep")
    else:
        logger.info("Single-run mode")
        try:
            if args['load_hyp_from_wandb_run'] is not None:
                run_params_fpath = wandb.restore('hyperparameters.json', run_path=args['load_hyp_from_wandb_run']).name
                with open(run_params_fpath, 'r') as fh:
                    run_params = json.load(fh)
                os.remove(run_params_fpath)
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
              skip_initial_eval=args['skip_initial_eval'],
              pairwise_eval_clustering=args['pairwise_eval_clustering'],
              debug=args['debug'],
              track_errors=args['track_errors'])
        logger.info("End of run")
