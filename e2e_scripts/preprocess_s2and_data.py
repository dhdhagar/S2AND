"""
Run from command line:
    python e2e_scripts/preprocess_s2and_data.py --data_home_dir="./data" --dataset_name="pubmed"
"""
from typing import Union, Dict
from typing import Tuple

from s2and.consts import PREPROCESSED_DATA_DIR
from s2and.featurizer import FeaturizationInfo, store_featurized_pickles, many_pairs_featurize
from os.path import join
from s2and.data import ANDData
import pickle
import numpy as np
from utils.parser import Parser

from s2and.data import ANDData
import logging
from s2and.featurizer import FeaturizationInfo, featurize

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def save_blockwise_featurized_data(dataset_name, random_seed):
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

    # Load the featurizer, which calculates pairwise similarity scores
    featurization_info = FeaturizationInfo()
    # the cache will make it faster to train multiple times - it stores the features on disk for you
    train_pkl, val_pkl, test_pkl = store_featurized_pickles(AND_dataset,
                                                            featurization_info,
                                                            n_jobs=2,
                                                            use_cache=False,
                                                            random_seed=random_seed)

    return train_pkl, val_pkl, test_pkl


def read_blockwise_features(pkl):
    blockwise_data: Dict[str, Tuple[np.ndarray, np.ndarray]]
    with open(pkl,"rb") as _pkl_file:
        blockwise_data = pickle.load(_pkl_file)

    print("Total num of blocks:", len(blockwise_data.keys()))
    return blockwise_data

def find_total_num_train_pairs(blockwise_data):
    count = 0
    for block_id in blockwise_data.keys():
        count += len(blockwise_data[block_id][0])

    print("Total num of signature pairs", count)

def verify_diff_with_s2and(dataset_name, random_seed):
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
        # train_pairs_size=100000,
        # val_pairs_size=10000,
        # test_pairs_size=10000,
        name=dataset_name,
        n_jobs=2,
        random_seed=random_seed,
    )

    # Load the featurizer, which calculates pairwise similarity scores
    featurization_info = FeaturizationInfo()
    # the cache will make it faster to train multiple times - it stores the features on disk for you
    train, val, test = featurize(AND_dataset, featurization_info, n_jobs=2, use_cache=False, random_seed=random_seed)
    X_train, y_train, _ = train
    X_val, y_val, _ = val

    logger.info("Done loading and featurizing")

    #Verify the 2 sets are equal
    with open("s2and_data_subsample.pkl", "rb") as _pkl_file:
        s2and_set = pickle.load(_pkl_file)

    with open("our_data_subsample.pkl", "rb") as _pkl_file:
        our_set = pickle.load(_pkl_file)

    print("VERIFICATION STATUS: ", s2and_set==our_set)


if __name__=='__main__':
    # Creates the pickles that store the preprocessed data
    # Read cmd line args
    parser = Parser(add_preprocessing_args=True)
    parser.add_preprocessing_args()

    args = parser.parse_args()
    print(args)

    params = args.__dict__
    DATA_HOME_DIR = params["data_home_dir"]
    dataset = params["dataset_name"]

    random_seeds = {1}
    for seed in random_seeds:
        print("Preprocessing started for seed value", seed)
        save_blockwise_featurized_data(dataset, seed)

        # Check the pickles are created OK
        train_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset}/seed{seed}/train_features.pkl"
        val_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset}/seed{seed}/val_features.pkl"
        test_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset}/seed{seed}/test_features.pkl"
        blockwise_features = read_blockwise_features(train_pkl)
        find_total_num_train_pairs(blockwise_features)
        verify_diff_with_s2and(dataset, seed)



