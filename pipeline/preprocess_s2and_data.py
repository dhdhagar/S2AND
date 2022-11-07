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

def save_blockwise_featurized_data(dataset_name, random_seed, nan_value=-1):
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
        random_seed=random_seed,
    )
    # Uncomment the following line if you wish to preprocess the whole dataset
    AND_dataset.process_whole_dataset()

    # Load the featurizer, which calculates pairwise similarity scores
    featurization_info = FeaturizationInfo()
    # the cache will make it faster to train multiple times - it stores the features on disk for you
    train_pkl, val_pkl, test_pkl = store_featurized_pickles(AND_dataset,
                                                            featurization_info,
                                                            n_jobs=2,
                                                            use_cache=False,
                                                            nan_value=nan_value,
                                                            random_seed=random_seed)

    return train_pkl, val_pkl, test_pkl


def read_blockwise_features(pkl):
    blockwise_data: Dict[str, Tuple[np.ndarray, np.ndarray]]
    with open(pkl,"rb") as _pkl_file:
        blockwise_data = pickle.load(_pkl_file)

    print("Total num of blocks:", len(blockwise_data.keys()))
    return blockwise_data


if __name__=='__main__':
    # Creates the pickles that store the preprocessed data
    # TODO: Create a loop to perform preprocessing for all Datasets
    dataset = "arnetminer"
    random_seeds = {1, 2, 3, 4, 5}
    for seed in random_seeds:
        save_blockwise_featurized_data(dataset, seed)

        # Check the pickles are created OK
        train_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset}/seed{seed}/train_features.pkl"
        val_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset}/seed{seed}/val_features.pkl"
        test_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset}/seed{seed}/test_features.pkl"
        blockwise_features = read_blockwise_features(train_pkl)



