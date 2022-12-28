from typing import Dict
from typing import Tuple
import math
import logging

import torch
import wandb
import copy
from torch.utils.data import DataLoader

from pipeline.model import model
from pipeline.trellis_cut_layer import TrellisCutLayer
from s2and.consts import PREPROCESSED_DATA_DIR
import pickle
import numpy as np
from s2and.data import S2BlocksDataset

from sklearn.metrics.cluster import v_measure_score
from torchmetrics.classification import BinaryAUROC

DATA_HOME_DIR = "/Users/pprakash/PycharmProjects/prob-ent-resolution/data/S2AND"
#DATA_HOME_DIR = "/work/pi_mccallum_umass_edu/pragyaprakas_umass_edu/prob-ent-resolution/data"

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
    ind = torch.triu_indices(n, n, offset=1, device=device)
    output = (torch.sparse_coo_tensor(ind, compressed_targets, [n, n])).to_dense()
    # Convert the upper triangular matrix to a symmetric matrix
    symm_mat = output + torch.transpose(output, 0, 1)
    symm_mat += torch.eye(n, device=device) # Set all 1s on the diagonal
    return symm_mat

def train_e2e_model(train_Dataloader, val_Dataloader):
    # Default hyperparameters
    hyperparams = {
        # model config
        "hidden_dim": 1024,
        "n_hidden_layers": 1,
        "dropout_p": 0.1,
        "hidden_config": None,
        "activation": "leaky_relu",
        # Training config
        "lr": 1e-5,
        "n_epochs": 1000,
        "weighted_loss": True,
        "use_lr_scheduler": True,
        "lr_factor": 0.6,
        "lr_min": 1e-6,
        "lr_scheduler_patience": 10,
        "weight_decay": 0.,
        "dev_opt_metric": 'auroc',
        "overfit_one_batch": True
    }

    # Start wandb run
    with wandb.init(config=hyperparams):
        hyp = wandb.config
        weighted_loss = hyp['weighted_loss']
        overfit_one_batch = hyp['overfit_one_batch']
        dev_opt_metric = hyp['dev_opt_metric']
        n_epochs = hyp['n_epochs']
        use_lr_scheduler = hyp['use_lr_scheduler']
        hidden_dim = hyp["hidden_dim"]
        n_hidden_layers = hyp["n_hidden_layers"]
        dropout_p = hyp["dropout_p"]
        hidden_config = hyp["hidden_config"]
        activation = hyp["activation"]

        # pos_weight = None
        # if weighted_loss:
        #     if overfit_one_batch:
        #         pos_weight = (batch_size - y_train_tensor[:batch_size].sum()) / y_train_tensor[:batch_size].sum()
        e2e_model = model(hidden_dim,
                          n_hidden_layers,
                          dropout_p,
                          hidden_config,
                          activation)
        logging.info("model loaded: %s", e2e_model)
        logging.info("Learnable parameters:")
        for name, parameter in e2e_model.named_parameters():
            if (parameter.requires_grad):
                logging.info(name)

        e2e_model.to(device)
        wandb.watch(e2e_model)

        optimizer = torch.optim.AdamW(e2e_model.parameters(), lr=hyp['lr'])

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        #                                                        mode='min',
        #                                                        factor=hyp['lr_factor'],
        #                                                        min_lr=hyp['lr_min'],
        #                                                        patience=hyp['lr_scheduler_patience'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

        batch_size = 0
        best_metric = 0
        best_model_on_dev = None
        best_epoch = 0
        loss_fn = torch.nn.BCEWithLogitsLoss()
        metric = BinaryAUROC(thresholds=None)
        for i in range(n_epochs):
            running_loss = []
            wandb.log({'epoch': i + 1})
            for (idx, batch) in enumerate(train_Dataloader):
                if(idx != 35):
                    continue
                if(idx > 35):
                    break
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
                logging.info("Data read, Uncompressed Batch size is: %s", n)

                # Forward pass through the e2e model
                output = e2e_model(data)

                # Calculate the loss and its gradients
                gold_output = target
                logging.info("gold output")
                logging.info(gold_output)

                # BCE Loss for binary classification
                loss = loss_fn(output.float(), gold_output.float())

                # Zero your gradients for every batch!
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                logging.info("Grad values")
                #logging.info(e2e_model.sdp_layer.W_val.grad)
                mlp_grad = e2e_model.mlp_layer.edge_weights.grad
                logging.info(uncompress_target_tensor(torch.reshape(mlp_grad.detach(), (-1,))))

                # Gather data and report
                logging.info("loss is %s", loss.item())
                running_loss.append(loss.item())

                train_f1_metric = metric(output, gold_output)
                print("training AUROC is ", train_f1_metric)

                # train_f1_metric = get_vmeasure_score(output.detach().numpy(), target.detach().numpy())
                # print("training f1 cluster measure is ", train_f1_metric)
                break

            if overfit_one_batch:
                wandb.log({'train_loss': np.mean(running_loss), 'train_auroc': train_f1_metric})

            if train_f1_metric > best_metric:
                best_metric = train_f1_metric
                best_epoch = i
                best_hyperparameters = {"hidden_dim ": hidden_dim,
        n_hidden_layers = hyp["n_hidden_layers"]
        dropout_p = hyp["dropout_p"]
        hidden_config"}

            # Update lr schedule
            # if use_lr_scheduler:
            #     scheduler.step(train_f1_metric)  # running_loss




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

    train_e2e_model(train_Dataloader, val_Dataloader)