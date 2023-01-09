import json
import os
import time
from collections import defaultdict
from typing import Dict
from typing import Tuple
import math
import logging
import random
import copy
import pickle

import torch
import wandb
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics.cluster import v_measure_score

from pipeline.model import EntResModel
from s2and.consts import PREPROCESSED_DATA_DIR
from s2and.data import S2BlocksDataset
from s2and.eval import b3_precision_recall_fscore
from utils.parser import Parser

from IPython import embed


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# Default hyperparameters
DEFAULT_HYPERPARAMS = {
    # Dataset
    "dataset": "pubmed",
    "dataset_random_seed": 1,
    # Run config
    "run_random_seed": 17,
    # Data config
    "convert_nan": True,
    "nan_value": -1,
    "drop_feat_nan_pct": -1,
    "normalize_data": False,
    # model config
    "neumiss_deq": False,
    "neumiss_depth": 20,
    "hidden_dim": 1024,
    "n_hidden_layers": 1,
    "dropout_p": 0.1,
    "dropout_only_once": True,
    "batchnorm": True,
    "hidden_config": None,
    "activation": "leaky_relu",
    "negative_slope": 0.01,
    # Training config
    "lr": 1e-5,
    "n_epochs": 2,
    # "weighted_loss": True,
    "use_lr_scheduler": True,
    "lr_scheduler": "plateau",  # "step"
    "lr_factor": 0.6,
    "lr_min": 1e-6,
    "lr_scheduler_patience": 10,
    "lr_step_size": 200,
    "lr_gamma": 0.1,
    "weight_decay": 0.,
    "dev_opt_metric": 'b3_f1',
    "overfit_batch_idx": -1  # 35
}

def read_blockwise_features(pkl):
    blockwise_data: Dict[str, Tuple[np.ndarray, np.ndarray]]
    with open(pkl,"rb") as _pkl_file:
        blockwise_data = pickle.load(_pkl_file)

    print("Total num of blocks:", len(blockwise_data.keys()))
    return blockwise_data


def get_dataloaders(dataset, dataset_seed, convert_nan, nan_value, normalize=False):
    train_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset}/seed{dataset_seed}/train_features.pkl"
    val_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset}/seed{dataset_seed}/val_features.pkl"
    test_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset}/seed{dataset_seed}/test_features.pkl"

    train_dataset = S2BlocksDataset(read_blockwise_features(train_pkl), convert_nan=convert_nan, nan_value=nan_value,
                                    scale=normalize)
    train_dataloader = DataLoader(train_dataset, shuffle=False)

    val_dataset = S2BlocksDataset(read_blockwise_features(val_pkl), convert_nan=convert_nan, nan_value=nan_value,
                                  scale=normalize, scaler=train_dataset.scaler)
    val_dataloader = DataLoader(val_dataset, shuffle=False)

    test_dataset = S2BlocksDataset(read_blockwise_features(test_pkl), convert_nan=convert_nan, nan_value=nan_value,
                                   scale=normalize, scaler=train_dataset.scaler)
    test_dataloader = DataLoader(test_dataset, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

def uncompress_target_tensor(compressed_targets, make_symmetric=True):
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


def evaluate(model, dataloader, overfit_batch_idx=-1):
    n_features = dataloader.dataset[0][0].shape[1]
    v_measure, b3_f1 = [], []
    for (idx, batch) in enumerate(dataloader):
        if overfit_batch_idx > -1:
            if idx < overfit_batch_idx:
                continue
            if idx > overfit_batch_idx:
                break
        data, target, cluster_ids = batch
        data = data.reshape(-1, n_features).float()
        block_size = get_matrix_size_from_triu(data)
        cluster_ids = np.reshape(cluster_ids, (block_size, ))
        target = target.flatten().float()

        # Forward pass through the e2e model
        data, target = data.to(device), target.to(device)
        _ = model(data, block_size)
        predicted_cluster_ids = model.hac_cut_layer.cluster_labels.detach()

        # Compute clustering metrics
        v_measure.append(v_measure_score(predicted_cluster_ids, cluster_ids))
        b3_f1_metrics = compute_b3_f1(cluster_ids, predicted_cluster_ids)
        b3_f1.append(b3_f1_metrics[2])

    return np.mean(v_measure), np.mean(b3_f1)


def train(hyperparams={}, verbose=False, project=None, entity=None, tags=None, group=None,
          save_model=False, load_model_from_wandb_run=None, load_model_from_fpath=None,
          eval_only_split=None):
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

        # Get data loaders (optionally with imputation, normalization)
        train_dataloader, val_dataloader, test_dataloader = get_dataloaders(hyp["dataset"], hyp["dataset_random_seed"],
                                                                            hyp["convert_nan"], hyp["nan_value"],
                                                                            hyp["normalize_data"])

        # Seed everything
        torch.manual_seed(hyp['run_random_seed'])
        random.seed(hyp['run_random_seed'])
        np.random.seed(hyp['run_random_seed'])

        # weighted_loss = hyp['weighted_loss']  # TODO: Think about how to implement this
        dev_opt_metric = hyp['dev_opt_metric']
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
        n_features = train_dataloader.dataset[0][0].shape[1]
        overfit_batch_idx = hyp['overfit_batch_idx']
        eval_metric_to_idx = {'v_measure': 0, 'b3_f1': 1}

        # Create model with hyperparams
        e2e_model = EntResModel(n_features, neumiss_depth, dropout_p, dropout_only_once, add_neumiss,
                                neumiss_deq, hidden_dim, n_hidden_layers, add_batchnorm,
                                activation, negative_slope, hidden_config)
        logger.info(f"Model loaded: {e2e_model}", )
        if verbose:
            logger.info("Learnable parameters:")
            for name, parameter in e2e_model.named_parameters():
                if parameter.requires_grad:
                    logger.info(name)

        # Load stored model, if available
        state_dict = None
        if load_model_from_wandb_run is not None:
            state_dict_fpath = wandb.restore('model_state_dict_best.pt',
                                             run_path=load_model_from_wandb_run).name
            state_dict = torch.load(state_dict_fpath, device)
        elif load_model_from_fpath is not None:
            state_dict = torch.load(load_model_from_fpath, device)
        if state_dict is not None:
            e2e_model.load_state_dict(state_dict)
            logger.info(f'Loaded stored model.')

        if eval_only_split is not None:
            # Run inference and exit
            dataloaders = {
                'train': train_dataloader,
                'dev': val_dataloader,
                'test': test_dataloader
            }
            with torch.no_grad():
                eval_scores = evaluate(e2e_model, dataloaders[eval_only_split])
                if verbose:
                    logger.info(
                        f"Eval: {eval_only_split}_vmeasure={eval_scores[0]}, {eval_only_split}_b3_f1={eval_scores[1]}")
                wandb.log({'epoch': 0, f'{eval_only_split}_vmeasure': eval_scores[0],
                           f'{eval_only_split}_b3_f1': eval_scores[1]})
        else:
            # Training
            e2e_model.to(device)
            wandb.watch(e2e_model)

            optimizer = torch.optim.AdamW(e2e_model.parameters(), lr=hyp['lr'])
            if use_lr_scheduler:
                if hyp['lr_scheduler'] == 'plateau':
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                           mode='min',
                                                                           factor=hyp['lr_factor'],
                                                                           min_lr=hyp['lr_min'],
                                                                           patience=hyp['lr_scheduler_patience'])
                elif hyp['lr_scheduler'] == 'step':
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hyp['lr_step_size'],
                                                                gamma=hyp['lr_gamma'])

            best_dev_state_dict = None
            best_dev_score = -1  # Stores the score of only the specified optimization metric
            best_dev_scores = None  # Contains scores of all metrics
            best_epoch = 0

            # Get initial model performance on dev (or 'train' for overfitting runs)
            with torch.no_grad():
                e2e_model.eval()
                if overfit_batch_idx > -1:
                    train_scores = evaluate(e2e_model, train_dataloader, overfit_batch_idx)
                    if verbose:
                        logger.info(f"Initial: train_vmeasure={train_scores[0]}, train_b3_f1={train_scores[1]}")
                    wandb.log({'epoch': 0, 'train_vmeasure': train_scores[0], 'train_b3_f1': train_scores[1]})
                else:
                    dev_scores = evaluate(e2e_model, val_dataloader)
                    if verbose:
                        logger.info(f"Initial: dev_vmeasure={dev_scores[0]}, dev_b3_f1={dev_scores[1]}")
                    wandb.log({'epoch': 0, 'dev_vmeasure': dev_scores[0], 'dev_b3_f1': dev_scores[1]})

            e2e_model.train()
            start_time = time.time()  # Tracks full training runtime
            for i in range(n_epochs):
                running_loss = []
                for (idx, batch) in enumerate(train_dataloader):
                    if overfit_batch_idx > -1:
                        if idx < overfit_batch_idx:
                            continue
                        if idx > overfit_batch_idx:
                            break
                    data, target, _ = batch
                    data = data.reshape(-1, n_features).float()
                    block_size = get_matrix_size_from_triu(data)
                    target = target.flatten().float()
                    if verbose:
                        logger.info(f"input shape: {data.shape}")
                        logger.info(f"input matrix size: {block_size}")
                        logger.info(f"target shape: {target.shape}")

                    if data.shape[0] == 1:
                        pass  # TODO: Fix this scenario

                    # Forward pass through the e2e model
                    data, target = data.to(device), target.to(device)
                    output = e2e_model(data, block_size, verbose)

                    # Calculate the loss
                    gold_output = uncompress_target_tensor(target)
                    if verbose:
                        logger.info("predicted output:")
                        logger.info(output)
                        logger.info("gold output:")
                        logger.info(gold_output)

                    loss = torch.norm(gold_output - output) / (2 * block_size)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if verbose:
                        logger.info(f"loss = {loss.item()}")
                    running_loss.append(loss.item())

                if verbose:
                    logger.info(f"epoch loss = {np.mean(running_loss)}")

                # Get model performance on dev (or 'train' for overfitting runs)
                with torch.no_grad():
                    e2e_model.eval()
                    if overfit_batch_idx > -1:
                        train_scores = evaluate(e2e_model, train_dataloader, overfit_batch_idx)
                        if verbose:
                            logger.info(f"Epoch {i + 1}: train_vmeasure={train_scores[0]}, train_b3_f1={train_scores[1]}")
                        wandb.log(
                            {'epoch': i + 1, 'train_loss_epoch': np.mean(running_loss), 'train_vmeasure': train_scores[0],
                             'train_b3_f1': train_scores[1]})
                        if use_lr_scheduler:
                            if hyp['lr_scheduler'] == 'plateau':
                                scheduler.step(train_scores[eval_metric_to_idx[dev_opt_metric]])
                            elif hyp['lr_scheduler'] == 'step':
                                scheduler.step()
                    else:
                        dev_scores = evaluate(e2e_model, val_dataloader)
                        if verbose:
                            logger.info(f"Epoch {i + 1}: dev_vmeasure={dev_scores[0]}, dev_b3_f1={dev_scores[1]}")
                        wandb.log({'epoch': i + 1, 'train_loss_epoch': np.mean(running_loss), 'dev_vmeasure': dev_scores[0],
                                   'dev_b3_f1': dev_scores[1]})
                        dev_opt_score = dev_scores[eval_metric_to_idx[dev_opt_metric]]
                        if dev_opt_score > best_dev_score:
                            if verbose:
                                logger.info(f"New best dev {dev_opt_metric} score @ epoch{i+1}: {dev_opt_score}; storing model")
                            best_epoch = i
                            best_dev_score = dev_opt_score
                            best_dev_scores = dev_scores
                            best_dev_state_dict = copy.deepcopy(e2e_model.state_dict())
                        if use_lr_scheduler:
                            if hyp['lr_scheduler'] == 'plateau':
                                scheduler.step(dev_scores[eval_metric_to_idx[dev_opt_metric]])
                            elif hyp['lr_scheduler'] == 'step':
                                scheduler.step()
                e2e_model.train()

            end_time = time.time()

            if overfit_batch_idx == -1:
                # Evaluate best dev model on test
                e2e_model.load_state_dict(best_dev_state_dict)
                e2e_model.eval()
                test_scores = evaluate(e2e_model, test_dataloader)
                if verbose:
                    logger.info(f"Final: test_vmeasure={test_scores[0]}, test_b3_f1={test_scores[1]}")

                # Log final metrics
                wandb.log({'best_dev_epoch': best_epoch + 1, 'best_dev_vmeasure': best_dev_scores[0],
                           'best_dev_b3_f1': best_dev_scores[1], 'best_test_vmeasure': test_scores[0],
                           'best_test_b3_f1': test_scores[1]})

        run.summary["z_model_parameters"] = count_parameters(e2e_model)
        run.summary["z_run_time"] = round(end_time - start_time)
        run.summary["z_run_dir_path"] = run.dir

        # Save models
        if save_model:
            torch.save(best_dev_state_dict, os.path.join(run.dir, 'model_state_dict_best.pt'))
            wandb.save('model_state_dict_best.pt')

        logger.info("End of train() call")


if __name__ == '__main__':
    # Read cmd line args
    parser = Parser(add_training_args=True)
    # Handle additional arbitrary arguments
    _, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith("--"):
            argument_name = arg.split('=')[0]
            if argument_name[2:] in DEFAULT_HYPERPARAMS:
                argument_type = type(DEFAULT_HYPERPARAMS[argument_name[2:]])
                if argument_type == bool:
                    parser.add_argument(argument_name, action='store_true')
                else:
                    parser.add_argument(argument_name, type=argument_type)
    args = parser.parse_args().__dict__
    hyp_args = {k: v for k, v in args.items() if k in DEFAULT_HYPERPARAMS}
    logger.info("Script arguments:")
    logger.info(args)

    if args['cpu']:
        device = torch.device("cpu")
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    print(f"Using device={device}")

    wandb.login()

    if args['wandb_run_params'] is not None:
        logger.info("Single-run mode")
        with open(args['wandb_run_params'], 'r') as fh:
            run_params = json.load(fh)
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
              eval_only_split=args['eval_only_split'])
        logger.info("End of run")
    else:
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
                    function=lambda: train(hyperparams=hyp_args),
                    count=args['wandb_max_runs'])

        logger.info("End of sweep")
