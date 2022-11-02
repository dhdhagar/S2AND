from typing import Union, Dict
from typing import List
from typing import Tuple

import hummingbird
import torch
from hummingbird.ml import constants
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

def load_pretrained_model_to_torch():
    with open(f"{DATA_HOME_DIR}/production_model.pickle", "rb") as _pkl_file:
        chckpt = pickle.load(_pkl_file)
        clusterer = chckpt['clusterer']

    # Get Classifier to convert to torch model
    lgbm = clusterer.classifier
    print(lgbm)
    torch_model = hummingbird.ml.convert(clusterer.classifier, "torch", None,
                                             extra_config=
                                             {constants.FINE_TUNE: True,
                                              constants.FINE_TUNE_DROPOUT_PROB: 0.1})
    return torch_model.model

def train(model, train_Dataloader):
    model.to(device)
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)

    def predict_class(model, input):
        return model(input)[1][:, 1]

    def evaluate(model, input, output):
        return (sum(model(input)[0] == output) / len(input)).item()

    # loop through each batch in the DataLoader object
    model.train()
    for (idx, batch) in enumerate(train_Dataloader):
        # LOADING THE DATA IN A BATCH
        data, target = batch

        # MOVING THE TENSORS TO THE CONFIGURED DEVICE
        data, target = data.to(device), target.to(device)

        # FORWARD PASS
        output = predict_class(model, data)
        loss = loss_fn(output, target)

        # BACKWARD AND OPTIMIZE
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # PREDICTIONS
        # Print batch loss
        with torch.no_grad():
            model.eval()
            print("\tBatch", f":{idx}", ":", loss_fn(predict_class(model, data), target).item())
        model.train()

        print("Accuracy on training set is",
              evaluate(model, data.to(device), target.to(device)))



if __name__=='__main__':
    dataset = "arnetminer"
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device={device}")

    train_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset}/seed1/train_features.pkl"
    val_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset}/seed1/val_features.pkl"
    test_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset}/seed1/test_features.pkl"
    blockwise_features = read_blockwise_features(train_pkl)

    train_Dataset = s2BlocksDataset(blockwise_features)
    train_Dataloader = DataLoader(train_Dataset, shuffle=True)

    lgbm_hm = load_pretrained_model_to_torch()
    model = train(lgbm_hm, train_Dataloader)