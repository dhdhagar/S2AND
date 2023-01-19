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
from e2e_scripts.evaluate import evaluate, evaluate_pairwise
from e2e_scripts.train_utils import DEFAULT_HYPERPARAMS, get_dataloaders, get_matrix_size_from_triu, \
    uncompress_target_tensor, count_parameters
from utils.parser import Parser

from IPython import embed


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def train(hyperparams={}, verbose=False, project=None, entity=None, tags=None, group=None,
          save_model=False, load_model_from_wandb_run=None, load_model_from_fpath=None,
          eval_only_split=None, skip_initial_eval=False, pairwise_mode=False,
          pairwise_eval_clustering=None):
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
        # Save hyperparameters as a json file and store in wandb run
        with open(os.path.join(run.dir, 'hyperparameters.json'), 'w') as fh:
            json.dump(dict(hyp), fh)
        wandb.save('hyperparameters.json')

        # Seed everything
        torch.manual_seed(hyp['run_random_seed'])
        random.seed(hyp['run_random_seed'])
        np.random.seed(hyp['run_random_seed'])

        weighted_loss = hyp['weighted_loss']
        batch_size = hyp['batch_size'] if pairwise_mode else 1  # Force clustering runs to operate on 1 block only
        n_epochs = hyp['n_epochs']
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
        overfit_batch_idx = hyp['overfit_batch_idx']
        clustering_metrics = {'b3_f1': 0, 'vmeasure': 1}
        pairwise_metrics = {'auroc': 0, 'f1': 1}
        eval_metric_to_idx = clustering_metrics if not pairwise_mode else pairwise_metrics
        dev_opt_metric = hyp['dev_opt_metric'] if hyp['dev_opt_metric'] in eval_metric_to_idx \
            else list(eval_metric_to_idx)[0]

        # Get data loaders (optionally with imputation, normalization)
        train_dataloader, val_dataloader, test_dataloader = get_dataloaders(hyp["dataset"], hyp["dataset_random_seed"],
                                                                            hyp["convert_nan"], hyp["nan_value"],
                                                                            hyp["normalize_data"], hyp["subsample_sz"],
                                                                            hyp["subsample_dev"], pairwise_mode,
                                                                            batch_size)
        n_features = train_dataloader.dataset[0][0].shape[1]

        # Create model with hyperparams
        if not pairwise_mode:
            model = EntResModel(n_features, neumiss_depth, dropout_p, dropout_only_once, add_neumiss,
                                neumiss_deq, hidden_dim, n_hidden_layers, add_batchnorm, activation,
                                negative_slope, hidden_config, sdp_max_iters, sdp_eps)
            # Define loss
            loss_fn = lambda pred, gold: torch.norm(gold - pred)
            # Define eval
            eval_fn = evaluate
            pairwise_clustering_fn = None  # Unused when pairwise_mode is False
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
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            # Define eval
            eval_fn = evaluate_pairwise
            pairwise_clustering_fn = None
            if pairwise_eval_clustering is not None:
                if pairwise_eval_clustering == 'cc':
                    pairwise_clustering_fn = CCInference(sdp_max_iters, sdp_eps)
                    pairwise_clustering_fn.eval()
                elif pairwise_eval_clustering == 'hac':
                    pairwise_clustering_fn = HACInference()  # TODO: Implement
                else:
                    raise ValueError('Invalid argument passed to --pairwise_eval_clustering')
                _, _, clustering_test_dataloader = get_dataloaders(hyp["dataset"], hyp["dataset_random_seed"],
                                                                   hyp["convert_nan"], hyp["nan_value"],
                                                                   hyp["normalize_data"], hyp["subsample_sz"],
                                                                   hyp["subsample_dev"], False, 1)
        logger.info(f"Model loaded: {model}", )

        # Load stored model, if available
        state_dict = None
        if load_model_from_wandb_run is not None:
            state_dict_fpath = wandb.restore('model_state_dict_best.pt',
                                             run_path=load_model_from_wandb_run).name
            state_dict = torch.load(state_dict_fpath, device)
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
            with torch.no_grad():
                model.eval()
                if pairwise_clustering_fn is not None:
                    assert eval_only_split == 'test'  # Clustering in --eval_only_split implemented only for test set
                    eval_metric_to_idx = clustering_metrics
                    eval_dataloader = clustering_test_dataloader
                else:
                    eval_dataloader = dataloaders[eval_only_split]
                start_time = time.time()
                eval_scores = eval_fn(model, eval_dataloader, clustering_fn=pairwise_clustering_fn,
                                      tqdm_label=eval_only_split, device=device)
                end_time = time.time()
                if verbose:
                    logger.info(
                        f"Eval: {eval_only_split}_{list(eval_metric_to_idx)[0]}={eval_scores[0]}, " +
                        f"{eval_only_split}_{list(eval_metric_to_idx)[1]}={eval_scores[1]}")
                wandb.log({'epoch': 0, f'{eval_only_split}_{list(eval_metric_to_idx)[0]}': eval_scores[0],
                           f'{eval_only_split}_{list(eval_metric_to_idx)[1]}': eval_scores[1]})
        else:
            # Training
            wandb.watch(model)

            optimizer = torch.optim.AdamW(model.parameters(), lr=hyp['lr'])
            if use_lr_scheduler:
                if hyp['lr_scheduler'] == 'plateau':
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                           mode='min',
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
                                               tqdm_label='train', device=device)
                        if verbose:
                            logger.info(f"Initial: train_{list(eval_metric_to_idx)[0]}={train_scores[0]}, " +
                                        f"train_{list(eval_metric_to_idx)[1]}={train_scores[1]}")
                        wandb.log({'epoch': 0, f'train_{list(eval_metric_to_idx)[0]}': train_scores[0],
                                   f'train_{list(eval_metric_to_idx)[1]}': train_scores[1]})
                    else:
                        dev_scores = eval_fn(model, val_dataloader, tqdm_label='dev', device=device)
                        if verbose:
                            logger.info(f"Initial: dev_{list(eval_metric_to_idx)[0]}={dev_scores[0]}, " +
                                        f"dev_{list(eval_metric_to_idx)[1]}={dev_scores[1]}")
                        wandb.log({'epoch': 0, f'dev_{list(eval_metric_to_idx)[0]}': dev_scores[0],
                                   f'dev_{list(eval_metric_to_idx)[1]}': dev_scores[1]})

            model.train()
            start_time = time.time()  # Tracks full training runtime
            for i in range(n_epochs):
                wandb.log({'epoch': i + 1})
                running_loss = []
                for (idx, batch) in enumerate(tqdm(train_dataloader, desc=f"Training {i + 1}")):
                    if overfit_batch_idx > -1:
                        if idx < overfit_batch_idx:
                            continue
                        if idx > overfit_batch_idx:
                            break
                    if not pairwise_mode:
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
                        if not pairwise_mode:
                            logger.info(f"Batch matrix size: {block_size}")

                    # Forward pass through the e2e or pairwise model
                    data, target = data.to(device), target.to(device)
                    output = model(data, block_size, verbose)

                    # Calculate the loss
                    if not pairwise_mode:
                        gold_output = uncompress_target_tensor(target, device=device)
                        if verbose:
                            logger.info(f"Gold:\n{gold_output}")
                        loss = loss_fn(output.view_as(gold_output), gold_output) / (2 * block_size)
                    else:
                        if verbose:
                            logger.info(f"Gold:\n{target}")
                        loss = loss_fn(output.view_as(target), target)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if verbose:
                        logger.info(f"Loss = {loss.item()}")
                    running_loss.append(loss.item())
                    wandb.log({'train_loss': np.mean(running_loss)})

                if verbose:
                    logger.info(f"Epoch loss = {np.mean(running_loss)}")

                # Get model performance on dev (or 'train' for overfitting runs)
                with torch.no_grad():
                    model.eval()
                    if overfit_batch_idx > -1:
                        train_scores = eval_fn(model, train_dataloader, overfit_batch_idx=overfit_batch_idx,
                                               tqdm_label='train', device=device)
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
                        dev_scores = eval_fn(model, val_dataloader, tqdm_label='dev', device=device)
                        if verbose:
                            logger.info(f"Epoch {i + 1}: dev_{list(eval_metric_to_idx)[0]}={dev_scores[0]}, " +
                                        f"dev_{list(eval_metric_to_idx)[1]}={dev_scores[1]}")
                        wandb.log({f'dev_{list(eval_metric_to_idx)[0]}': dev_scores[0],
                                   f'dev_{list(eval_metric_to_idx)[1]}': dev_scores[1]})
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
                # Evaluate best dev model on test
                model.load_state_dict(best_dev_state_dict)
                with torch.no_grad():
                    model.eval()
                    test_scores = eval_fn(model, test_dataloader, tqdm_label='test', device=device)
                    if verbose:
                        logger.info(f"Final: test_{list(eval_metric_to_idx)[0]}={test_scores[0]}, " +
                                    f"test_{list(eval_metric_to_idx)[1]}={test_scores[1]}")
                    # Log final metrics
                    wandb.log({'best_dev_epoch': best_epoch + 1,
                               f'best_dev_{list(eval_metric_to_idx)[0]}': best_dev_scores[0],
                               f'best_dev_{list(eval_metric_to_idx)[1]}': best_dev_scores[1],
                               f'best_test_{list(eval_metric_to_idx)[0]}': test_scores[0],
                               f'best_test_{list(eval_metric_to_idx)[1]}': test_scores[1]})
                    if pairwise_clustering_fn is not None:
                        clustering_scores = eval_fn(model, clustering_test_dataloader,
                                                    clustering_fn=pairwise_clustering_fn, tqdm_label='test clustering',
                                                    device=device)
                        if verbose:
                            logger.info(f"Final: test_{list(clustering_metrics)[0]}={clustering_scores[0]}, " +
                                        f"test_{list(clustering_metrics)[1]}={clustering_scores[1]}")
                        # Log final metrics
                        wandb.log({f'best_test_{list(clustering_metrics)[0]}': clustering_scores[0],
                                   f'best_test_{list(clustering_metrics)[1]}': clustering_scores[1]})


        run.summary["z_model_parameters"] = count_parameters(model)
        run.summary["z_run_time"] = round(end_time - start_time)
        run.summary["z_run_dir_path"] = run.dir

        # Save models
        if save_model:
            torch.save(best_dev_state_dict, os.path.join(run.dir, 'model_state_dict_best.pt'))
            wandb.save('model_state_dict_best.pt')
            logger.info(f"Saved best model on dev to {os.path.join(run.dir, 'model_state_dict_best.pt')}")

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
                                           save_model=args['save_model'],
                                           skip_initial_eval=args['skip_initial_eval'],
                                           pairwise_mode=args['pairwise_mode']),
                    count=args['wandb_max_runs'])

        logger.info("End of sweep")
    else:
        logger.info("Single-run mode")
        try:
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
              pairwise_mode=args['pairwise_mode'],
              pairwise_eval_clustering=args['pairwise_eval_clustering'])
        logger.info("End of run")
