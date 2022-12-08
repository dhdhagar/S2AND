from typing import Dict
from typing import Tuple
import math
import logging

import torch
import wandb
import copy
from torch.utils.data import DataLoader

from pipeline.model import model
from s2and.consts import PREPROCESSED_DATA_DIR
import pickle
import numpy as np
from s2and.data import S2BlocksDataset

from sklearn.metrics.cluster import v_measure_score

#DATA_HOME_DIR = "/Users/pprakash/PycharmProjects/prob-ent-resolution/data/S2AND"
DATA_HOME_DIR = "/work/pi_mccallum_umass_edu/pragyaprakas_umass_edu/prob-ent-resolution/data"

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def read_blockwise_features(pkl):
    blockwise_data: Dict[str, Tuple[np.ndarray, np.ndarray]]
    with open(pkl,"rb") as _pkl_file:
        blockwise_data = pickle.load(_pkl_file)

    print("Total num of blocks:", len(blockwise_data.keys()))
    return blockwise_data

def uncompress_target_tensor(compressed_targets):
    n = round(math.sqrt(2 * compressed_targets.size(dim=0))) + 1
    # Convert the 1D pairwise-similarities list to nxn upper triangular matrix
    ind = torch.triu_indices(n, n, offset=1)
    # TODO: try size [n,n]
    output = (torch.sparse_coo_tensor(ind, compressed_targets, [n, n])).to_dense()
    symm_mat = output + torch.transpose(output, 0, 1)
    return symm_mat

def evaluate_e2e_model(model, val_dataloader, eval_metric):
    f1_score = 0
    for (idx, batch) in enumerate(val_dataloader):
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



def train_e2e_model(e2e_model, train_Dataloader, val_Dataloader):
    # Default hyperparameters
    hyperparams = {
        # Training config
        "lr": 1e-4,
        "n_epochs": 200,
        "weighted_loss": True,
        "use_lr_scheduler": True,
        "lr_factor": 0.6,
        "lr_min": 1e-6,
        "lr_scheduler_patience": 10,
        "weight_decay": 0.,
        "dev_opt_metric": 'v_measure_score',
        "overfit_one_batch": False
    }

    # Start wandb run
    with wandb.init(config=hyperparams):
        hyp = wandb.config
        weighted_loss = hyp['weighted_loss']
        overfit_one_batch = hyp['overfit_one_batch']
        dev_opt_metric = hyp['dev_opt_metric']
        n_epochs = hyp['n_epochs']
        use_lr_scheduler = hyp['use_lr_scheduler']

        model.to(device)
        wandb.watch(model)

        optimizer = torch.optim.SGD(e2e_model.parameters(), lr=hyp['lr'], weight_decay=0.9)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=hyp['lr_factor'],
                                                               min_lr=hyp['lr_min'],
                                                               patience=hyp['lr_scheduler_patience'])

        batch_size = 0
        best_metric = 0
        best_model_on_dev = None
        best_epoch = 0
        for i in range(n_epochs):
            running_loss = []
            wandb.log({'epoch': i + 1})
            for (idx, batch) in enumerate(train_Dataloader):
                # LOADING THE DATA IN A BATCH
                data, target = batch

                # MOVING THE TENSORS TO THE CONFIGURED DEVICE
                data, target = data.to(device), target.to(device)
                # Reshape data to 2-D matrix, and target to 1D
                n = np.shape(data)[1]
                f = np.shape(data)[2]

                batch_size = n
                data = torch.reshape(data, (n, f))
                target = torch.reshape(target, (n,))
                print("Data read, Uncompressed Batch size is: ", target.size())

                # Zero your gradients for every batch!
                optimizer.zero_grad()

                # Forward pass through the e2e model
                output = e2e_model(data)
                # print(output)

                # Calculate the loss and its gradients
                gold_output = uncompress_target_tensor(target)
                loss = torch.norm(gold_output - output)
                loss.backward()
                # print(e2e_model.sdp_layer.W_val.grad)
                # print(e2e_model.mlp_layer.mlp_model._operators[0].weight_3.grad)
                # Gather data and report
                print("loss is ", loss.item())

            # Print epoch validation accuracy
            with torch.no_grad():
                e2e_model.eval()
                dev_f1_metric = evaluate_e2e_model(e2e_model, val_Dataloader, dev_opt_metric)
                logger.info("Epoch", i + 1, ":", "Dev vmeasure:", dev_f1_metric)
                if dev_f1_metric > best_metric:
                    logger.info(f"New best dev {dev_opt_metric}; storing model")
                    best_epoch = i
                    best_metric = dev_f1_metric
                    best_model_on_dev = copy.deepcopy(model)
                if overfit_one_batch:
                    train_f1_metric = evaluate_e2e_model(e2e_model, train_Dataloader, dev_opt_metric)
                    print("training f1 cluster measure is "+train_f1_metric)
            e2e_model.train()

            wandb.log({
                'train_loss_epoch': np.mean(running_loss),
                'dev_vmeasure': dev_f1_metric,
            })
            if overfit_one_batch:
                wandb.log({'train_vmeasure': train_f1_metric})

            # Update lr schedule
            if use_lr_scheduler:
                scheduler.step(dev_f1_metric)  # running_loss




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
    #print(train_Dataloader)

    blockwise_features = read_blockwise_features(val_pkl)
    val_Dataset = S2BlocksDataset(blockwise_features)
    val_Dataloader = DataLoader(val_Dataset, shuffle=False)

    e2e_model = model()
    print("model loaded", e2e_model)
    print("Learnable parameters:")
    for name, parameter in e2e_model.named_parameters():
        if(parameter.requires_grad):
            print(name)

    train_e2e_model(e2e_model, train_Dataloader, val_Dataloader)