from typing import Dict, Tuple
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
import math

from s2and.consts import PREPROCESSED_DATA_DIR
from s2and.data import S2BlocksDataset


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
    output = (torch.sparse_coo_tensor(ind, compressed_targets, [n, n])).to_dense()
    # Convert the upper triangular matrix to a symmetric matrix
    symm_mat = output + torch.transpose(output, 0, 1)
    symm_mat += torch.eye(n) # Set all 1s on the diagonal
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

    for idx, block_id in enumerate(blockwise_features.keys()):
        X, y, cluster_ids = blockwise_features[block_id]
        data = torch.from_numpy(X)
        target = torch.from_numpy(y)

        # MOVING THE TENSORS TO THE CONFIGURED DEVICE
        data, target = data.to(device), target.to(device)
        # Reshape data to 2-D matrix, and target to 1D
        n = np.shape(data)[1]
        f = np.shape(data)[2]

        batch_size = n
        data = torch.reshape(data, (n, f))
        target = torch.reshape(target, (n,))

        # uncompressed size =
        gold_output = uncompress_target_tensor(target)
        print("block size:", gold_output.size(0))
        block_size = gold_output.size(0)

        if(block_size<=30) :
            #check if it has multiple clusters
            print("block_id is", block_id)
            print("idx is", idx)
            print(gold_output)
            print(np.unique(cluster_ids))