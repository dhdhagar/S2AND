import pickle
from os.path import join
from typing import Dict, Tuple
import numpy as np
from s2and.consts import PREPROCESSED_DATA_DIR
from s2and.data import ANDData
import logging
from s2and.model import PairwiseModeler
from s2and.featurizer import FeaturizationInfo, featurize
from s2and.eval import pairwise_eval, cluster_eval
from s2and.model import Clusterer, FastCluster
from hyperopt import hp
import os

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = "../data"

def load_training_data(train_pkl, val_pkl):
    blockwise_data: Dict[str, Tuple[np.ndarray, np.ndarray]]
    with open(train_pkl, "rb") as _pkl_file:
        blockwise_data = pickle.load(_pkl_file)
    # Combine the blockwise_data to form complete train, test, val sets
    x_vals = [val[0] for val in blockwise_data.values()]
    X_train = np.concatenate(x_vals)

    y_vals = [val[1] for val in blockwise_data.values()]
    y_train = np.concatenate(y_vals)

    blockwise_data_val: Dict[str, Tuple[np.ndarray, np.ndarray]]
    with open(val_pkl, "rb") as _pkl_file:
        blockwise_data_val = pickle.load(_pkl_file)
    # Combine the blockwise_data to form complete train, test, val sets
    x_vals = [val[0] for val in blockwise_data_val.values()]
    X_val = np.concatenate(x_vals)

    y_vals = [val[1] for val in blockwise_data_val.values()]
    y_val = np.concatenate(y_vals)

    logger.info(X_train, y_train)
    logger.info(np.shape(X_train), np.shape(y_train))
    logger.info(X_val, y_val)
    logger.info(np.shape(X_val), np.shape(y_val))
    logger.info("Dataset loaded and prepared for training")

    # Training Featurizer model
    featurization_info = FeaturizationInfo()

    logger.info("Done loading and featurizing")
    return featurization_info, X_train, y_train, X_val, y_val

def train_pairwise_classifier(featurization_info, X_train, y_train, X_val, y_val, save_model=True):
    # calibration fits isotonic regression after the binary classifier is fit
    # monotone constraints help the LightGBM classifier behave sensibly
    pairwise_model = PairwiseModeler(
        n_iter=25, monotone_constraints=featurization_info.lightgbm_monotone_constraints
    )
    # this does hyperparameter selection, which is why we need to pass in the validation set.
    pairwise_model.fit(X_train, y_train, X_val, y_val)
    logger.info("Fitted the Pairwise model")
    if save_model:
        # Create a new directory
        directory = '../s2and_experiments'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Create a new file within the directory
        file_path = os.path.join(directory, f'{dataset_name}_{dataset_seed}_lgbm.pkl')
        # Serialize the model to a file
        with open(file_path, 'wb') as file:
            pickle.dump(pairwise_model, file)
            logger.info("Saved the Pairwise classification model")

    # this will also dump a lot of useful plots (ROC, PR, SHAP) to the figs_path
    # pairwise_metrics = pairwise_eval(
    #     X_val,
    #     y_val,
    #     pairwise_model,
    #     os.path.join(DATA_DIR, "s2and_experiments",  "figs"),
    #     f"{dataset_name}_seed_{dataset_seed}",
    #     featurization_info.get_feature_names()
    # )
    # logger.info(pairwise_metrics)

    return pairwise_model

def train_HAC_clusterer(dataset_name, featurization_info, pairwise_model):
    clusterer = Clusterer(
        featurization_info,
        pairwise_model,
        cluster_model=FastCluster(linkage="average"),
        search_space={"eps": hp.uniform("eps", 0, 1)},
        n_iter=25,
        n_jobs=8,
    )
    clusterer.fit(dataset_name)
    # Save clusterer object to pickle
    
    # the metrics_per_signature are there so we can break out the facets if needed
    metrics, metrics_per_signature = cluster_eval(dataset_name, clusterer)
    logger.info(metrics)


if __name__=='__main__':
    datasets = {"pubmed", "arnetminer", "qian", "zbmath", "kisti"}
    seeds = {1, 2, 3, 4, 5}

    for dataset_name in datasets:
        for dataset_seed in seeds:
            parent_dir = f"../data/{dataset_name}"
            train_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset_name}/seed{dataset_seed}/train_features.pkl"
            val_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset_name}/seed{dataset_seed}/val_features.pkl"
            test_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset_name}/seed{dataset_seed}/test_features.pkl"

            featurization_info, X_train, y_train, X_val, y_val = load_training_data(train_pkl, val_pkl)
            pairwise_model = train_pairwise_classifier(featurization_info, X_train, y_train, X_val, y_val)
            #train_HAC_clusterer(dataset_name, featurization_info, pairwise_model)