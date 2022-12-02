from typing import Dict
from typing import Tuple
import math

import torch
from torch.utils.data import DataLoader

from pipeline.model import model
from s2and.consts import PREPROCESSED_DATA_DIR
import pickle
import numpy as np
from s2and.data import S2BlocksDataset

#DATA_HOME_DIR = "/Users/pprakash/PycharmProjects/prob-ent-resolution/data/S2AND"
DATA_HOME_DIR = "/work/pi_mccallum_umass_edu/pragyaprakas_umass_edu/prob-ent-resolution/data"

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
    print(train_Dataloader)

    e2e_model = model()
    print("model loaded", e2e_model)
    print("Learnable parameters:")
    for name, parameter in e2e_model.named_parameters():
        if(parameter.requires_grad):
            print(name)

    optimizer = torch.optim.SGD(e2e_model.parameters(), lr=0.001, momentum=0.9)

    # Only train for first block picked up by dataloader:
    # Get the first block size
    batch_size = 0
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
        print("Data read", data.size(), target.size())

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Forward pass through the e2e model
        output = e2e_model(data)
        print(output)

        # Calculate the loss and its gradients
        # TODO: pairwise labels
        gold_output = uncompress_target_tensor(target)
        loss = torch.norm(gold_output - output)
        loss.backward()
        print(e2e_model.sdp_layer.W_val.grad)
        print(e2e_model.mlp_layer.mlp_model._operators[0].weight_3.grad)

        # Gather data and report
        print("loss is ", loss.item())


    #train(e2e_model, train_Dataloader)