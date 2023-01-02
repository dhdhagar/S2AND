import json
import os
import time
from typing import Dict
from typing import Tuple
import math
import logging
import random

import torch
import wandb
import copy
from torch.utils.data import DataLoader

from pipeline.model import EntResModel
from pipeline.trellis_cut_layer import TrellisCutLayer
from s2and.consts import PREPROCESSED_DATA_DIR
import pickle
import numpy as np
from s2and.data import S2BlocksDataset

from sklearn.metrics.cluster import v_measure_score

from utils.parser import Parser
from IPython import embed

DATA_HOME_DIR = "/Users/pprakash/PycharmProjects/prob-ent-resolution/data/S2AND"
#DATA_HOME_DIR = "/work/pi_mccallum_umass_edu/pragyaprakas_umass_edu/prob-ent-resolution/data"

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# Default hyperparameters
DEFAULT_HYPERPARAMS = {
    "verbose": True,
    # Dataset
    "dataset": "pubmed",
    "dataset_random_seed": 1,
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
    "batchnorm": True,
    "hidden_config": None,
    "activation": "leaky_relu",
    "negative_slope": 0.01,
    # Training config
    "lr": 1e-5,
    "n_epochs": 5,
    "weighted_loss": True,
    "use_lr_scheduler": True,
    "lr_factor": 0.6,
    "lr_min": 1e-6,
    "lr_scheduler_patience": 10,
    "weight_decay": 0.,
    "dev_opt_metric": 'v_measure_score',
    "overfit_one_batch": True
}

def read_blockwise_features(pkl):
    blockwise_data: Dict[str, Tuple[np.ndarray, np.ndarray]]
    with open(pkl,"rb") as _pkl_file:
        blockwise_data = pickle.load(_pkl_file)

    print("Total num of blocks:", len(blockwise_data.keys()))
    return blockwise_data

def load_training_data(dataset, dataset_seed, convert_nan, nan_value):
    train_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset}/seed{dataset_seed}/train_features.pkl"
    val_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset}/seed{dataset_seed}/val_features.pkl"
    #test_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset}/seed{dataset_seed}/test_features.pkl"

    blockwise_features = read_blockwise_features(train_pkl)
    train_Dataset = S2BlocksDataset(blockwise_features, convert_nan=convert_nan, nan_value=nan_value)
    train_Dataloader = DataLoader(train_Dataset, shuffle=False)

    blockwise_features = read_blockwise_features(val_pkl)
    val_Dataset = S2BlocksDataset(blockwise_features, convert_nan=convert_nan, nan_value=nan_value)
    val_Dataloader = DataLoader(val_Dataset, shuffle=False)

    return train_Dataloader, val_Dataloader

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

def evaluate_e2e_model(model, dataloader, eval_metric):
    model.eval()
    f1_score = 0
    n_features = 39
    for (idx, batch) in enumerate(dataloader):
        #TODO: Remove for all batches
        if idx < 27:
            continue
        if idx > 27:
            break
        data, target, clusterIds = batch
        data = data.reshape(-1, n_features).float()
        block_size = get_matrix_size_from_triu(data)
        target = target.flatten().float()

        if data.shape[0] == 1:
            embed()
        else:
            continue

        # Forward pass through the e2e model
        data, target = data.to(device), target.to(device)
        output = model(data, block_size)
        predicted_clusterIds = model.hac_cut_layer.cluster_labels
        logger.info("predicted cluster Ids:", predicted_clusterIds)

        # Calculate the v_measure_score
        if(eval_metric == "v_measure_score"):
            f1_score += v_measure_score(torch.flatten(predicted_clusterIds), torch.flatten(clusterIds))
            logger.info("Cumulative f1 score", f1_score)

    return f1_score



def train_e2e_model(hyperparams={}, verbose=False, project=None, entity=None,
          tags=None, group=None, default_hyperparams=DEFAULT_HYPERPARAMS,
          save_model=False, load_model_from_wandb_run=None, load_model_from_fpath=None):
    init_args = {
        'config': default_hyperparams
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

        # Load data
        train_Dataloader, val_Dataloader = load_training_data(hyp["dataset"], hyp["dataset_random_seed"],
                                                              hyp["convert_nan"], hyp["nan_value"])

        # Seed everything
        torch.manual_seed(hyp['run_random_seed'])
        random.seed(hyp['run_random_seed'])
        np.random.seed(hyp['run_random_seed'])

        weighted_loss = hyp['weighted_loss']
        dev_opt_metric = hyp['dev_opt_metric']
        n_epochs = hyp['n_epochs']
        use_lr_scheduler = hyp['use_lr_scheduler']
        hidden_dim = hyp["hidden_dim"]
        n_hidden_layers = hyp["n_hidden_layers"]
        dropout_p = hyp["dropout_p"]
        hidden_config = hyp["hidden_config"]
        activation = hyp["activation"]
        add_batchnorm = hyp["batchnorm"]
        neumiss_deq = hyp["neumiss_deq"]
        neumiss_depth = hyp["neumiss_depth"]
        add_neumiss = not hyp['convert_nan']
        negative_slope=hyp["negative_slope"]
        n_features = train_Dataloader.dataset[0][0].shape[1]

        # Create model with hyperparams
        e2e_model = EntResModel(n_features, neumiss_depth, dropout_p, add_neumiss,
                                neumiss_deq, hidden_dim, n_hidden_layers, add_batchnorm,
                                activation, negative_slope, hidden_config)
        logger.info("Model loaded: %s", e2e_model)
        logger.info("Learnable parameters:")
        for name, parameter in e2e_model.named_parameters():
            if (parameter.requires_grad):
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

        # TODO: Implement flag and code to run only inference

        # Training Code
        e2e_model.to(device)
        wandb.watch(e2e_model)

        optimizer = torch.optim.AdamW(e2e_model.parameters(), lr=hyp['lr'])

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        #                                                        mode='min',
        #                                                        factor=hyp['lr_factor'],
        #                                                        min_lr=hyp['lr_min'],
        #                                                        patience=hyp['lr_scheduler_patience'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

        e2e_model.train()

        best_model_on_dev = None
        best_dev_f1 = -1
        best_epoch = 0
        batch_idx_to_select = 27  # Based on manual inspection; batch with large size

        start_time = time.time()
        for i in range(n_epochs):
            running_loss = []
            for (idx, batch) in enumerate(train_Dataloader):
                if hyp['overfit_one_batch']:
                    if idx < batch_idx_to_select:
                        continue
                    if idx > batch_idx_to_select:
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
                    embed()
                else:
                    continue

                # Forward pass through the e2e model
                data, target = data.to(device), target.to(device)
                output = e2e_model(data, block_size)

                # Calculate the loss
                gold_output = uncompress_target_tensor(target)
                if verbose:
                    logger.info("predicted output:")
                    logger.info(output)
                    logger.info("gold output:")
                    logger.info(gold_output)

                loss = torch.norm(gold_output - output)/(2*block_size)

                # Zero your gradients for every batch!
                optimizer.zero_grad()
                # Backward pass
                loss.backward()
                optimizer.step()
                if use_lr_scheduler:
                    scheduler.step()

                # # Print grad values for debugging
                # logger.info("Grad values")
                # logger.info(e2e_model.sdp_layer.W_val.grad)
                # mlp_grad = e2e_model.mlp_layer.edge_weights.grad
                # logger.info(uncompress_target_tensor(torch.reshape(mlp_grad.detach(), (-1,))))

                # Gather data and report
                logger.info("loss is %s", loss.item())
                running_loss.append(loss.item())

                # train_f1_metric = get_vmeasure_score(output.detach().numpy(), target.detach().numpy())
                # print("training f1 cluster measure is ", train_f1_metric)

            # Get model performance on dev
            with torch.no_grad():
                e2e_model.eval()
                # dev_f1_metric = evaluate_e2e_model(e2e_model, val_Dataloader, dev_opt_metric)
                # logger.info("Epoch", i + 1, ":", "Dev vmeasure:", dev_f1_metric)
                # if dev_f1_metric > best_metric:
                #     logger.info(f"New best dev {dev_opt_metric}; storing model")
                #     best_epoch = i
                #     best_metric = dev_f1_metric
                #     best_model_on_dev = copy.deepcopy(model)
                if hyp['overfit_one_batch']:
                    train_f1_metric = evaluate_e2e_model(e2e_model, train_Dataloader, dev_opt_metric)
                    logger.info("training f1 cluster vmeasure is ", train_f1_metric)
            e2e_model.train()


            if hyp['overfit_one_batch']:
                wandb.log({'epoch': i + 1, 'train_loss_epoch': np.mean(running_loss)})#, 'train_vmeasure': train_f1_metric})

            end_time = time.time()

            run.summary["z_model_parameters"] = count_parameters(e2e_model)
            run.summary["z_run_time"] = round(end_time - start_time)

            # Save models
            if save_model:
                torch.save(best_model_on_dev.state_dict(), os.path.join(run.dir, 'model_state_dict_best.pt'))
                wandb.save('model_state_dict_best.pt')
                # torch.save(model.state_dict(), os.path.join(run.dir, 'model_state_dict_final.pt'))
                # wandb.save('model_state_dict_final.pt')

        logger.info("End of train() call")



if __name__=='__main__':
    # Read cmd line args
    parser = Parser(add_training_args=True)
    parser.add_training_args()

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

    #wandb.login()

    if args['wandb_run_params'] is not None:
        logger.info("Single-run mode")
        with open(args['wandb_run_params'], 'r') as fh:
            run_params = json.load(fh)
        run_params.update(hyp_args)
        train_e2e_model(hyperparams=run_params,
              verbose=True,
              project=args['wandb_project'],
              entity=args['wandb_entity'],
              tags=args['wandb_tags'],
              group=args['wandb_group'],
              save_model=args['save_model'],
              load_model_from_wandb_run=args['load_model_from_wandb_run'],
              load_model_from_fpath=args['load_model_from_fpath'])
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
                    function=lambda: train_e2e_model(hyperparams=hyp_args),
                    count=args['wandb_max_runs'])

        logger.info("End of sweep")
