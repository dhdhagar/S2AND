from typing import Union, Dict
from typing import List
from typing import Tuple

import hummingbird
import torch
from hummingbird.ml import constants
from torch.optim import optimizer
from torch.utils.data import Dataset, DataLoader

from s2and.consts import PREPROCESSED_DATA_DIR
from s2and.featurizer import FeaturizationInfo, store_featurized_pickles, many_pairs_featurize
from os.path import join
from s2and.data import ANDData
import pickle
import numpy as np

#DATA_HOME_DIR = "/Users/pprakash/PycharmProjects/prob-ent-resolution/data/S2AND"
DATA_HOME_DIR = "/work/pi_mccallum_umass_edu/pragyaprakas_umass_edu/prob-ent-resolution/data"

def save_blockwise_featurized_data(dataset_name):
    parent_dir = f"{DATA_HOME_DIR}/{dataset_name}"
    AND_dataset = ANDData(
        signatures=join(parent_dir, f"{dataset_name}_signatures.json"),
        papers=join(parent_dir, f"{dataset_name}_papers.json"),
        mode="train",
        specter_embeddings=join(parent_dir, f"{dataset_name}_specter.pickle"),
        clusters=join(parent_dir, f"{dataset_name}_clusters.json"),
        block_type="s2",
        train_pairs_size=100,
        val_pairs_size=100,
        test_pairs_size=100,
        name=dataset_name,
        n_jobs=2,
    )
    # Uncomment the following line if you wish to preprocess the whole dataset
    #AND_dataset.process_whole_dataset()

    # Load the featurizer, which calculates pairwise similarity scores
    featurization_info = FeaturizationInfo()
    # the cache will make it faster to train multiple times - it stores the features on disk for you
    train_pkl, val_pkl, test_pkl = store_featurized_pickles(AND_dataset, featurization_info,
                                                            n_jobs=2, use_cache=False, nan_value=-1)

    return train_pkl, val_pkl, test_pkl


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


if __name__=='__main__':
    # Creates the pickles that store the preprocessed data
    dataset = "arnetminer"
    save_blockwise_featurized_data(dataset)

    # Check the pickles are created OK
    train_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset}/seed1/train_features.pkl"
    val_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset}/seed1/val_features.pkl"
    test_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset}/seed1/test_features.pkl"
    blockwise_features = read_blockwise_features(train_pkl)

    # Sample Dataloader
    train_Dataset = s2BlocksDataset(blockwise_features)
    train_Dataloader = DataLoader(train_Dataset, shuffle=True)

    for (idx, batch) in enumerate(train_Dataloader):
        # LOADING THE DATA IN A BATCH TO TEST EVERYTHING IS CORRECT
        data, target = batch
        print("batch #", idx, np.shape(data), np.shape(target))

