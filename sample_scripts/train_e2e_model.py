from typing import Dict
from typing import Tuple
import math
import logging
import time
import torch
import wandb
from torch.utils.data import DataLoader

from pipeline.model import EntResModel
from s2and.consts import PREPROCESSED_DATA_DIR
import pickle
import numpy as np
from s2and.data import S2BlocksDataset

from sklearn.metrics.cluster import v_measure_score

from IPython import embed

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def read_blockwise_features(pkl):
    blockwise_data: Dict[str, Tuple[np.ndarray, np.ndarray]]
    with open(pkl,"rb") as _pkl_file:
        blockwise_data = pickle.load(_pkl_file)

    print("Total num of blocks:", len(blockwise_data.keys()))
    return blockwise_data

def uncompress_target_tensor(compressed_targets, make_symmetric=True):
    n = round(math.sqrt(2 * compressed_targets.size(dim=0))) + 1
    ind0, ind1 = torch.triu_indices(n, n, offset=1)
    target = torch.eye(n, device=device)
    target[ind0, ind1] = compressed_targets
    if make_symmetric:
        target[ind1, ind0] = compressed_targets
    return target

def evaluate_e2e_model(model, dataloader, eval_metric):
    f1_score = 0
    for (idx, batch) in enumerate(dataloader):
        data, target = batch

        # MOVING THE TENSORS TO THE CONFIGURED DEVICE
        data, target = data.to(device), target.to(device)
        # Reshape data to 2-D matrix, and target to 1D
        n = np.shape(data)[1]
        f = np.shape(data)[2]

        batch_size = n
        data = torch.reshape(data, (n, f))
        target = torch.reshape(target, (n,))

        output = model(data)
        # print(output)

        # Calculate the loss and its gradients
        gold_output = uncompress_target_tensor(target)
        if(eval_metric == "v_measure_score"):
            f1_score += v_measure_score(torch.flatten(output), torch.flatten(gold_output))
            print("Cumulative f1 score", f1_score)

        return f1_score
        break

def get_vmeasure_score(outputs, gold_labels):
    # takes as input model output (nxn matrix) and target labels to find cluster v_measure_score
    # Convert outputs to upper triangular matrices
    outputs_triu = np.triu(outputs, 1)
    idxs = np.triu_indices(np.shape(outputs)[0], 1)
    outputs_1d = outputs_triu[idxs]
    print("compressed output", outputs_1d, "max op", np.max(outputs_1d), "targets", gold_labels)

    f1_score = v_measure_score(outputs_1d, gold_labels)

    return f1_score


def get_matrix_size_from_triu(triu):
    return round(math.sqrt(2 * len(triu))) + 1


def train_e2e_model(train_Dataloader, val_Dataloader):
    # Default hyperparameters
    hyperparams = {
        # model config
        "hidden_dim": 512,
        "n_hidden_layers": 1,
        "dropout_p": 0,
        "hidden_config": None,
        "activation": "leaky_relu",
        # Training config
        "lr": 4e-5,
        "n_epochs": 100,
        "weighted_loss": False,
        "use_lr_scheduler": True,
        "lr_factor": 0.6,
        "lr_min": 1e-6,
        "lr_scheduler_patience": 10,
        "weight_decay": 0.,
        "dev_opt_metric": 'v_measure_score',
        "overfit_one_batch": True
    }

    # Start wandb run
    with wandb.init(config=hyperparams) as run:
        hyp = wandb.config
        n_features = train_Dataloader.dataset[0][0].shape[1]
        e2e_model = EntResModel(n_features,
                                hyp['hidden_dim'],
                                hyp['n_hidden_layers'],
                                hyp['dropout_p'],
                                hyp['hidden_config'],
                                hyp['activation'])

        e2e_model.to(device)
        wandb.watch(e2e_model)

        optimizer = torch.optim.AdamW(e2e_model.parameters(), lr=hyp['lr'])
        if hyp['use_lr_scheduler']:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                   mode='min',
                                                                   factor=hyp['lr_factor'],
                                                                   min_lr=hyp['lr_min'],
                                                                   patience=hyp['lr_scheduler_patience'])

        batch_sz_to_select = [50, 75]  # Based on manual inspection; batch with large size
        start_time = time.time()
        for i in range(hyp['n_epochs']):
            epoch_start_time = time.time()
            running_loss = []
            wandb.log({'epoch': i+1})
            for (idx, batch) in enumerate(train_Dataloader):
                batch_start_time = time.time()
                data, target, _ = batch
                data = data.reshape(-1, n_features).float()
                N = get_matrix_size_from_triu(data)

                if hyp['overfit_one_batch']:
                    if N < batch_sz_to_select[0] or N > batch_sz_to_select[1]:
                        continue

                target = target.flatten().float()

                if data.shape[0] == 1:
                    continue  # skip because of batchnorm; TODO: Add a check to see if batchnorm is used; skip only then

                # Forward pass
                data, target = data.to(device), target.to(device)
                output = e2e_model(data, N)

                # Compute loss
                gold_output = uncompress_target_tensor(target)
                loss = torch.norm(gold_output - output) / (2 * N)

                # Zero your gradients for every batch!
                optimizer.zero_grad()
                
                backward_start_time = time.time()
                loss.backward()
                backward_end_time = time.time()
                logger.info(f'loss.backward() runtime = {backward_end_time - backward_start_time} (matrix size={N})')
                optimizer.step()
                
                # Gather data and report
                logger.info("loss is %s", loss.item())
                running_loss.append(loss.item())

                batch_end_time = time.time()
                logger.info(f'Batch runtime = {round(batch_end_time - batch_start_time)}')

                wandb.log({'train_loss': np.mean(running_loss)})
                if hyp['overfit_one_batch']:
                    break

            # Update lr schedule
            if hyp['use_lr_scheduler']:
                scheduler.step(np.mean(running_loss))

            epoch_end_time = time.time()
            logger.info(f'Epoch runtime = {round(epoch_end_time - epoch_start_time)}')

        run.summary["z_run_time"] = round(time.time() - start_time)


if __name__=='__main__':
    dataset = "pubmed"
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device={device}")

    train_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset}/seed1/train_features.pkl"
    val_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset}/seed1/val_features.pkl"
    test_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset}/seed1/test_features.pkl"

    blockwise_features = read_blockwise_features(train_pkl)
    train_Dataset = S2BlocksDataset(blockwise_features)
    train_Dataloader = DataLoader(train_Dataset, shuffle=False)

    blockwise_features = read_blockwise_features(val_pkl)
    val_Dataset = S2BlocksDataset(blockwise_features)
    val_Dataloader = DataLoader(val_Dataset, shuffle=False)

    train_e2e_model(train_Dataloader, val_Dataloader)
