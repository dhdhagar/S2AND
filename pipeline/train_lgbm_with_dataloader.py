from typing import Union, Dict
from typing import List
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from s2and.consts import PREPROCESSED_DATA_DIR
import pickle
import numpy as np

DATA_HOME_DIR = "/Users/pprakash/PycharmProjects/prob-ent-resolution/data/S2AND"
#DATA_HOME_DIR = "/work/pi_mccallum_umass_edu/pragyaprakas_umass_edu/prob-ent-resolution/data"

def read_blockwise_features(pkl):
    blockwise_data: Dict[str, Tuple[np.ndarray, np.ndarray]]
    with open(pkl,"rb") as _pkl_file:
        blockwise_data = pickle.load(_pkl_file)

    print(list(blockwise_data.keys())[0])
    print(len(blockwise_data.keys()))
    print(blockwise_data)
    return blockwise_data

class s2BlocksDataset(Dataset):
    def __init__(self, blockwise_data: Dict[str, Tuple[np.ndarray, np.ndarray]]):
        self.blockwise_data = blockwise_data

    def __len__(self):
        return len(self.blockwise_data.keys())


    def __getitem__(self, idx):
        dict_key = list(self.blockwise_data.keys())[idx]
        X, y = self.blockwise_data[dict_key]
        # TODO: Add subsampling logic here

        return (X, y)

def load_pretrained_model():
    with open(f"{DATA_HOME_DIR}/production_model.pickle", "rb") as _pkl_file:
        chckpt = pickle.load(_pkl_file)
        clusterer = chckpt['clusterer']

    # Get Classifier to convert to torch model
    lgbm = clusterer.classifier
    print(lgbm)
    return lgbm

def train_lgbm(model, train_Dataloader):
    # Run fit for all batches in dataloader
    for (idx, batch) in enumerate(train_Dataloader):
        # LOADING THE DATA IN A BATCH
        data, target = batch
        n = np.shape(data)[1]
        if(n<=1):
            continue
        X_train = torch.reshape(data, (n, 39)).numpy()
        Y_train = torch.reshape(target, (n,)).numpy()
        print(np.shape(X_train), np.shape(Y_train))
        print("Running fir for batch "+str(idx))
        model.fit(X_train, Y_train)
    return model

def eval_lgbm(model, test_Dataloader):
    # TODO add evaluate code
    return


if __name__=='__main__':
    dataset = "arnetminer"
    train_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset}/seed1/train_features.pkl"
    val_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset}/seed1/val_features.pkl"
    test_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset}/seed1/test_features.pkl"
    blockwise_features = read_blockwise_features(train_pkl)

    train_Dataset = s2BlocksDataset(blockwise_features)
    train_Dataloader = DataLoader(train_Dataset, shuffle=True)

    lgbm = load_pretrained_model()
    model = train_lgbm(lgbm, train_Dataloader)