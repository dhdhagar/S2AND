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

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def load_training_data(train_pkl, val_pkl):
    blockwise_data: Dict[str, Tuple[np.ndarray, np.ndarray]]
    with open(train_pkl, "rb") as _pkl_file:
        blockwise_data = pickle.load(_pkl_file)
    # Combine the blockwise_data to form complete train, test, val sets
    remove_arr = np.zeros(39)
    X_train = [remove_arr]
    y_train = []
    for block_data in blockwise_data.values():
        x, y, cluster_ids = block_data
        X_train = np.concatenate((X_train, x), axis=0)
        y_train = np.concatenate((y_train, y), axis=0)
    X_train = np.delete(X_train, 0)

    blockwise_data_val: Dict[str, Tuple[np.ndarray, np.ndarray]]
    with open(val_pkl, "rb") as _pkl_file:
        blockwise_data_val = pickle.load(_pkl_file)
    # Combine the blockwise_data to form complete train, test, val sets
    X_val = [remove_arr]
    y_val = []
    for block_data in blockwise_data_val.values():
        x, y, cluster_ids = block_data
        X_val = np.concatenate((X_val, x), axis=0)
        y_val = np.concatenate((y_val, y), axis=0)
    X_val = np.delete(X_val, 0)

    logger.info("Dataset loaded and prepared for training")

    # dataset = ANDData(
    #     signatures=join(parent_dir, f"{dataset_name}_signatures.json"),
    #     papers=join(parent_dir, f"{dataset_name}_papers.json"),
    #     mode="train",
    #     specter_embeddings=join(parent_dir, f"{dataset_name}_specter.pickle"),
    #     clusters=join(parent_dir, f"{dataset_name}_clusters.json"),
    #     block_type="s2",
    #     name=dataset_name,
    #     n_jobs=4,
    # )
    # logger.info("loaded the data")

    # Training Featurizer model
    featurization_info = FeaturizationInfo()

    logger.info("Done loading and featurizing")
    return featurization_info, X_train, y_train, X_val, y_val

def train_pairwise_classifier(featurization_info, X_train, y_train, X_val, y_val):
    # calibration fits isotonic regression after the binary classifier is fit
    # monotone constraints help the LightGBM classifier behave sensibly
    pairwise_model = PairwiseModeler(
        n_iter=25, monotone_constraints=featurization_info.lightgbm_monotone_constraints
    )
    # this does hyperparameter selection, which is why we need to pass in the validation set.
    pairwise_model.fit(X_train, y_train, X_val, y_val)
    logger.info("Fitted the Pairwise model")

    # this will also dump a lot of useful plots (ROC, PR, SHAP) to the figs_path
    pairwise_metrics = pairwise_eval(X_val, y_val, pairwise_model.classifier, figs_path='figs/', title='validation_metrics')
    logger.info(pairwise_metrics)
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
    
    # the metrics_per_signature are there so we can break out the facets if needed
    metrics, metrics_per_signature = cluster_eval(dataset_name, clusterer)
    logger.info(metrics)


if __name__=='__main__':
    dataset_name = "pubmed"
    dataset_seed = 1
    parent_dir = f"../data/{dataset_name}"
    train_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset_name}/seed{dataset_seed}/train_features.pkl"
    val_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset_name}/seed{dataset_seed}/val_features.pkl"
    test_pkl = f"{PREPROCESSED_DATA_DIR}/{dataset_name}/seed{dataset_seed}/test_features.pkl"

    featurization_info, X_train, y_train, X_val, y_val = load_training_data(train_pkl, val_pkl)
    pairwise_model = train_pairwise_classifier(featurization_info, X_train, y_train, X_val, y_val)
    train_HAC_clusterer(dataset_name, featurization_info, pairwise_model)